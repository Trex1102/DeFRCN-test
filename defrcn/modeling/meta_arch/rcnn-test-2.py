
import os
import json
import cv2
import numpy as np
import logging
from pathlib import Path

import torch
from torch import nn

from detectron2.structures import ImageList
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.data import MetadataCatalog

from .build import META_ARCH_REGISTRY
from .gdl import decouple_layer, AffineLayer
from defrcn.modeling.roi_heads import build_roi_heads

__all__ = ["GeneralizedRCNN"]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):

    def __init__(self, cfg):
        """
        GeneralizedRCNN with minimal inference saving:
        - saves only the top-1 predicted bbox per image as visualization + JSON summary.
        """
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)

        # Build modules
        self.backbone = build_backbone(cfg)
        self._SHAPE_ = self.backbone.output_shape()
        self.proposal_generator = build_proposal_generator(cfg, self._SHAPE_)
        self.roi_heads = build_roi_heads(cfg, self._SHAPE_)
        self.normalizer = self.normalize_fn()
        self.affine_rpn = AffineLayer(num_channels=self._SHAPE_['res4'].channels, bias=True)
        self.affine_rcnn = AffineLayer(num_channels=self._SHAPE_['res4'].channels, bias=True)

        # Output directory for saving top-1 prediction visualizations and JSONs
        self.output_dir = Path(getattr(cfg, "OUTPUT_DIR", "./defrcn_inference_out"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Move modules to device
        self.to(self.device)

        # Optionally freeze parts
        if getattr(cfg.MODEL, "BACKBONE", None) and getattr(cfg.MODEL.BACKBONE, "FREEZE", False):
            for p in self.backbone.parameters():
                p.requires_grad = False
            logging.info("froze backbone parameters")

        if getattr(cfg.MODEL, "RPN", None) and getattr(cfg.MODEL.RPN, "FREEZE", False):
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            logging.info("froze proposal generator parameters")

        if getattr(cfg.MODEL, "ROI_HEADS", None) and getattr(cfg.MODEL.ROI_HEADS, "FREEZE_FEAT", False):
            if hasattr(self.roi_heads, "res5"):
                for p in self.roi_heads.res5.parameters():
                    p.requires_grad = False
            logging.info("froze roi_box_head parameters")

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)
        assert "instances" in batched_inputs[0], "Training forward expects 'instances' in inputs"
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        proposal_losses, detector_losses, _, _ = self._forward_once_(batched_inputs, gt_instances)
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs):
        """
        Inference that saves only the top-1 predicted bbox per image (visualization + JSON).
        """
        assert not self.training

        _, _, results, image_sizes = self._forward_once_(batched_inputs, None)

        # Try to get class names from MetadataCatalog (fallback to numeric ids)
        class_names = None
        try:
            ds = None
            if len(self.cfg.DATASETS.TEST) > 0:
                ds = self.cfg.DATASETS.TEST[0]
            elif len(self.cfg.DATASETS.TRAIN) > 0:
                ds = self.cfg.DATASETS.TRAIN[0]
            if ds:
                meta = MetadataCatalog.get(ds)
                class_names = meta.get("thing_classes", None)
        except Exception:
            class_names = None

        processed_results = []
        for idx, (r, input, image_size) in enumerate(zip(results, batched_inputs, image_sizes)):
            height = input.get("height", image_size[0])
            width = input.get("width", image_size[1])
            r = detector_postprocess(r, height, width)
            processed_results.append({"instances": r})

            # Save only the top-1 prediction visualization & JSON summary
            try:
                self._save_top_prediction(idx, input, r, class_names=class_names)
            except Exception as e:
                logging.exception(f"Failed to save top prediction for image idx={idx}: {e}")

        return processed_results

    def _forward_once_(self, batched_inputs, gt_instances=None):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        features_de_rpn = features
        if getattr(self.cfg.MODEL.RPN, "ENABLE_DECOUPLE", False):
            scale = self.cfg.MODEL.RPN.BACKWARD_SCALE
            features_de_rpn = {k: self.affine_rpn(decouple_layer(features[k], scale)) for k in features}
        proposals, proposal_losses = self.proposal_generator(images, features_de_rpn, gt_instances)

        features_de_rcnn = features
        if getattr(self.cfg.MODEL.ROI_HEADS, "ENABLE_DECOUPLE", False):
            scale = self.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE
            features_de_rcnn = {k: self.affine_rcnn(decouple_layer(features[k], scale)) for k in features}
        results, detector_losses = self.roi_heads(images, features_de_rcnn, proposals, gt_instances)

        return proposal_losses, detector_losses, results, images.image_sizes

    def preprocess_image(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def normalize_fn(self):
        assert len(self.cfg.MODEL.PIXEL_MEAN) == len(self.cfg.MODEL.PIXEL_STD)
        num_channels = len(self.cfg.MODEL.PIXEL_MEAN)
        pixel_mean = (torch.Tensor(
            self.cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1))
        pixel_std = (torch.Tensor(
            self.cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1))
        return lambda x: (x - pixel_mean) / pixel_std

    # ----------------- Helpers for saving top-1 prediction -----------------
    def _tensor_to_numpy_image(self, tensor, input_dict=None):
        """
        Convert a CHW torch tensor to HWC uint8 BGR image for OpenCV.
        Prefers to load original file from input_dict['file_name'] if present.
        """
        if input_dict is not None:
            file_name = input_dict.get("file_name", None)
            if file_name:
                img = cv2.imread(str(file_name))
                if img is not None:
                    return img  # BGR uint8

        t = tensor.detach().cpu()
        if t.ndim == 4 and t.shape[0] == 1:
            t = t[0]
        if t.ndim != 3:
            raise ValueError(f"Unexpected image tensor shape: {t.shape}")

        arr = t.numpy()
        arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC

        # If normalized to 0..1
        if arr.max() <= 1.01:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

        # Convert single-channel to 3-channel BGR if needed
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        if arr.shape[2] == 1:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        return arr

    def _boxes_to_list(self, boxes):
        """
        Convert detectron2 Boxes-like object to a Python list of [x1,y1,x2,y2].
        """
        if boxes is None:
            return []
        try:
            b = boxes.tensor.detach().cpu().numpy()
        except Exception:
            b = np.asarray(boxes)
        return b.tolist()

    def _save_top_prediction(self, idx, input, predictions_instances, class_names=None):
        """
        Save only the highest-confidence predicted bbox:
         - one visualization image under self.output_dir/visualizations/
         - one JSON summary next to it
        """
        out_root = self.output_dir
        vis_dir = out_root / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        # Recover image
        img = None
        if "image" in input:
            try:
                img = self._tensor_to_numpy_image(input["image"], input_dict=input)
            except Exception:
                img = None
        if img is None:
            h = input.get("height", 800)
            w = input.get("width", 1333)
            img = np.zeros((h, w, 3), dtype=np.uint8)

        # Extract predictions
        preds_boxes = []
        preds_scores = []
        preds_classes = []
        if predictions_instances is not None:
            if hasattr(predictions_instances, "pred_boxes"):
                preds_boxes = self._boxes_to_list(predictions_instances.pred_boxes)
            if hasattr(predictions_instances, "scores"):
                try:
                    preds_scores = predictions_instances.scores.detach().cpu().tolist()
                except Exception:
                    preds_scores = list(predictions_instances.scores)
            if hasattr(predictions_instances, "pred_classes"):
                try:
                    preds_classes = predictions_instances.pred_classes.detach().cpu().tolist()
                except Exception:
                    preds_classes = list(predictions_instances.pred_classes)

        if len(preds_boxes) == 0:
            logging.info(f"No predictions for image idx={idx}; skipping top1 save.")
            return

        # Choose top-1 by score if available, otherwise first box
        top_idx = 0
        if len(preds_scores) > 0:
            top_idx = int(np.argmax(np.array(preds_scores)))
        x1, y1, x2, y2 = map(int, preds_boxes[top_idx])
        score = preds_scores[top_idx] if top_idx < len(preds_scores) else None
        cls_id = preds_classes[top_idx] if top_idx < len(preds_classes) else None
        cls_name = str(cls_id) if (class_names is None or cls_id is None) else class_names[cls_id]

        vis = img.copy()
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{cls_name}"
        if score is not None:
            label = f"{label} {score:.2f}"
        cv2.putText(vis, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Save visualization image
        vis_name = input.get("file_name", f"img_{idx}.jpg")
        vis_basename = Path(vis_name).stem
        out_vis_path = vis_dir / f"{vis_basename}_{idx}_top1.jpg"
        cv2.imwrite(str(out_vis_path), vis)

        # Save JSON summary for the top prediction
        summary = {
            "image_index": idx,
            "image_file": input.get("file_name", None),
            "top_prediction": {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "score": float(score) if score is not None else None,
                "class_id": int(cls_id) if cls_id is not None else None,
                "class_name": cls_name
            }
        }
        with open(out_root / f"{vis_basename}_{idx}_top1.json", "w") as f:
            json.dump(summary, f, indent=2)

        logging.info(f"Saved top-1 visualization to {out_vis_path}")













