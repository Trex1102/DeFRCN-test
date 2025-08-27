
import os
import json
import cv2
import numpy as np
import logging
from pathlib import Path

import torch
from torch import nn

from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
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
        super().__init__()

        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self._SHAPE_ = self.backbone.output_shape()
        self.proposal_generator = build_proposal_generator(cfg, self._SHAPE_)
        self.roi_heads = build_roi_heads(cfg, self._SHAPE_)
        self.normalizer = self.normalize_fn()
        self.affine_rpn = AffineLayer(num_channels=self._SHAPE_['res4'].channels, bias=True)
        self.affine_rcnn = AffineLayer(num_channels=self._SHAPE_['res4'].channels, bias=True)
        self.to(self.device)

        # output directory for saving inference artifacts
        self.output_dir = Path(getattr(cfg, "OUTPUT_DIR", "./defrcn_inference_out"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("froze backbone parameters")

        if cfg.MODEL.RPN.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            print("froze proposal generator parameters")

        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for p in self.roi_heads.res5.parameters():
                p.requires_grad = False
            print("froze roi_box_head parameters")

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)
        assert "instances" in batched_inputs[0]
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        proposal_losses, detector_losses, _, _ = self._forward_once_(batched_inputs, gt_instances)
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs):
        """
        Modified inference:
        - If batched_inputs contain "instances" (gt), they will be used for comparison and saved.
        - Saves visualization images, per-class crops, and a json summary per image under self.output_dir.
        """
        assert not self.training

        # Build gt_instances list if present in inputs (user may pass instances for evaluation)
        gt_instances = None
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        _, _, results, image_sizes = self._forward_once_(batched_inputs, gt_instances)

        # Try to get class names from MetadataCatalog if dataset is configured
        class_names = None
        try:
            # pick first registered test dataset (if any)
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

            # Save visualization and per-class crops
            try:
                self._save_image_and_annotations(
                    idx, input, r, gt=(gt_instances[idx] if gt_instances is not None else None),
                    class_names=class_names
                )
            except Exception as e:
                logging.exception(f"Failed to save visualization for image idx={idx}: {e}")

        return processed_results

    def _forward_once_(self, batched_inputs, gt_instances=None):

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        features_de_rpn = features
        if self.cfg.MODEL.RPN.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.RPN.BACKWARD_SCALE
            features_de_rpn = {k: self.affine_rpn(decouple_layer(features[k], scale)) for k in features}
        proposals, proposal_losses = self.proposal_generator(images, features_de_rpn, gt_instances)

        features_de_rcnn = features
        if self.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
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

    # ----------------------- Utilities for saving visualizations and crops -----------------------
    def _tensor_to_numpy_image(self, tensor, input_dict=None):
        """
        Convert a CHW torch tensor to HWC uint8 image. Handles common cases:
        - tensor in [0,1] float -> multiply 255
        - tensor already 0..255 -> convert directly
        - if file_name present in input_dict, try cv2.imread(file_name) to get original image
        """
        # Prefer loading original file if path exists
        if input_dict is not None:
            file_name = input_dict.get("file_name", None)
            if file_name:
                img = cv2.imread(str(file_name))
                if img is not None:
                    return img  # BGR uint8

        # Otherwise convert tensor
        t = tensor.detach().cpu()
        if t.ndim == 3:  # C,H,W
            t = t
        elif t.ndim == 4 and t.shape[0] == 1:
            t = t[0]
        else:
            raise ValueError("Unexpected image tensor shape: {}".format(t.shape))

        arr = t.numpy()
        # CHW -> HWC
        arr = np.transpose(arr, (1, 2, 0))

        # If values appear normalized by pixel mean/std (unlikely for original batched_inputs),
        # try to undo normalization if necessary. We assume original batched_inputs image is in 0..255 or 0..1.
        if arr.max() <= 1.01:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

        # if image has 1 channel, convert to 3-channels
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        if arr.shape[2] == 1:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        return arr

    def _boxes_to_list(self, boxes):
        """
        boxes: detectron2.structures.Boxes-like object with .tensor
        returns list of [x1,y1,x2,y2]
        """
        if boxes is None:
            return []
        try:
            b = boxes.tensor.detach().cpu().numpy()
        except Exception:
            # maybe it's already a numpy array
            b = np.asarray(boxes)
        return b.tolist()

    def _classes_from_instances(self, inst):
        """
        Return list of class ids (or names) from an Instances object.
        """
        if inst is None:
            return []
        # Try common field names
        cls_field = None
        for attr in ("pred_classes", "scores", "gt_classes", "gt_ids", "classes", "labels"):
            if hasattr(inst, attr):
                cls_field = attr
                break

        # For predictions: inst.pred_classes is a tensor of ints
        ids = []
        if hasattr(inst, "pred_classes"):
            try:
                ids = inst.pred_classes.detach().cpu().tolist()
            except Exception:
                ids = list(inst.pred_classes)
        elif hasattr(inst, "gt_classes"):
            try:
                ids = inst.gt_classes.detach().cpu().tolist()
            except Exception:
                ids = list(inst.gt_classes)
        elif hasattr(inst, "classes"):
            try:
                ids = inst.classes.detach().cpu().tolist()
            except Exception:
                ids = list(inst.classes)
        else:
            # as fallback, attempt to see if there's an attribute named 'labels'
            if hasattr(inst, "labels"):
                try:
                    ids = inst.labels.detach().cpu().tolist()
                except Exception:
                    ids = list(inst.labels)
        return ids

    def _save_image_and_annotations(self, idx, input, predictions_instances, gt=None, class_names=None):
        """
        Save:
         - overlay image: both preds (red) and GT (green)
         - per-class crop images into predicted/<class_name>/ and gt/<class_name>/
         - JSON summary file with all pred and gt boxes/classes/scores
        """
        # prepare directories
        out_root = self.output_dir
        vis_dir = out_root / "visualizations"
        preds_dir = out_root / "predicted"
        gts_dir = out_root / "gt"
        vis_dir.mkdir(parents=True, exist_ok=True)
        preds_dir.mkdir(parents=True, exist_ok=True)
        gts_dir.mkdir(parents=True, exist_ok=True)

        # Recover a usable image (BGR uint8)
        # `input["image"]` is usually a Tensor CHW
        img = None
        if "image" in input:
            try:
                img = self._tensor_to_numpy_image(input["image"], input_dict=input)
            except Exception:
                img = None

        if img is None:
            # fallback to blank image with declared height / width
            h = input.get("height", 800)
            w = input.get("width", 1333)
            img = np.zeros((h, w, 3), dtype=np.uint8)

        vis = img.copy()

        # Predictions
        preds_boxes = []
        preds_scores = []
        preds_classes = []
        if predictions_instances is not None:
            # Extract boxes, scores, classes
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

        # Ground truth
        gt_boxes = []
        gt_classes = []
        if gt is not None:
            if hasattr(gt, "gt_boxes"):
                gt_boxes = self._boxes_to_list(gt.gt_boxes)
            # try several possible class field names
            if hasattr(gt, "gt_classes"):
                try:
                    gt_classes = gt.gt_classes.detach().cpu().tolist()
                except Exception:
                    gt_classes = list(gt.gt_classes)
            elif hasattr(gt, "classes"):
                try:
                    gt_classes = gt.classes.detach().cpu().tolist()
                except Exception:
                    gt_classes = list(gt.classes)
            elif hasattr(gt, "labels"):
                try:
                    gt_classes = gt.labels.detach().cpu().tolist()
                except Exception:
                    gt_classes = list(gt.labels)

        # Draw GT boxes (green)
        for i, box in enumerate(gt_boxes):
            x1, y1, x2, y2 = map(int, box)
            cls_id = gt_classes[i] if i < len(gt_classes) else None
            cls_name = str(cls_id) if (class_names is None or cls_id is None) else class_names[cls_id]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"GT: {cls_name}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            # save crop to gts_dir/<class_name>/
            try:
                crop = img[y1:y2, x1:x2]
                if crop.size > 0:
                    cls_folder = gts_dir / (cls_name if cls_name is not None else "unknown")
                    cls_folder.mkdir(parents=True, exist_ok=True)
                    fname = f"{idx}_gt_{i}_{cls_name}_{x1}_{y1}_{x2}_{y2}.jpg"
                    cv2.imwrite(str(cls_folder / fname), crop)
            except Exception:
                logging.exception("Failed to save GT crop")

        # Draw prediction boxes (red), annotate score & class, and save crop per predicted class
        for i, box in enumerate(preds_boxes):
            x1, y1, x2, y2 = map(int, box)
            score = preds_scores[i] if i < len(preds_scores) else None
            cls_id = preds_classes[i] if i < len(preds_classes) else None
            cls_name = str(cls_id) if (class_names is None or cls_id is None) else class_names[cls_id]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"P: {cls_name}"
            if score is not None:
                label = f"{label} {score:.2f}"
            cv2.putText(vis, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

            # save crop to preds_dir/<class_name>/
            try:
                crop = img[y1:y2, x1:x2]
                if crop.size > 0:
                    cls_folder = preds_dir / (cls_name if cls_name is not None else "unknown")
                    cls_folder.mkdir(parents=True, exist_ok=True)
                    fname = f"{idx}_pred_{i}_{cls_name}_{x1}_{y1}_{x2}_{y2}_{int(score*100) if score is not None else 0}.jpg"
                    cv2.imwrite(str(cls_folder / fname), crop)
            except Exception:
                logging.exception("Failed to save pred crop")

        # Save visualized overlay
        vis_name = input.get("file_name", f"img_{idx}.jpg")
        vis_basename = Path(vis_name).stem
        out_vis_path = vis_dir / f"{vis_basename}_{idx}_vis.jpg"
        cv2.imwrite(str(out_vis_path), vis)

        # Save JSON summary (preds and gts)
        summary = {
            "image_index": idx,
            "image_file": input.get("file_name", None),
            "predictions": [],
            "ground_truths": []
        }
        for i, box in enumerate(preds_boxes):
            entry = {
                "bbox": [float(x) for x in box],
                "score": float(preds_scores[i]) if i < len(preds_scores) else None,
                "class_id": int(preds_classes[i]) if i < len(preds_classes) else None,
                "class_name": (class_names[preds_classes[i]] if (class_names is not None and i < len(preds_classes) and preds_classes[i] is not None) else (str(preds_classes[i]) if i < len(preds_classes) else None))
            }
            summary["predictions"].append(entry)
        for i, box in enumerate(gt_boxes):
            entry = {
                "bbox": [float(x) for x in box],
                "class_id": int(gt_classes[i]) if i < len(gt_classes) else None,
                "class_name": (class_names[gt_classes[i]] if (class_names is not None and i < len(gt_classes) and gt_classes[i] is not None) else (str(gt_classes[i]) if i < len(gt_classes) else None))
            }
            summary["ground_truths"].append(entry)

        with open(out_root / f"{vis_basename}_{idx}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # done
        logging.info(f"Saved visualization to {out_vis_path}, preds-> {preds_dir}, gts-> {gts_dir}")
