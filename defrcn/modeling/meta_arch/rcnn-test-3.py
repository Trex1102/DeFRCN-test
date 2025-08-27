
import os
import json
import logging
from typing import List

import torch
from torch import nn
import numpy as np
from PIL import Image

from detectron2.structures import ImageList, Boxes, Instances
from detectron2.utils.logger import log_first_n
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

from .build import META_ARCH_REGISTRY
from .gdl import decouple_layer, AffineLayer
from defrcn.modeling.roi_heads import build_roi_heads


__all__ = ["GeneralizedRCNN"]


def xyxy_to_list(box: np.ndarray) -> List[float]:
    # box: [x1, y1, x2, y2]
    return [float(box[0]), float(box[1]), float(box[2]), float(box[3])]


def compute_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    # box format: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH

    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

    union = areaA + areaB - interArea
    if union <= 0:
        return 0.0
    return float(interArea / union)


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

        # where visualizations will be written (can override via cfg.DEFRCN.VIS_OUTPUT_DIR)
        self.vis_output_dir = getattr(cfg, "DEFRCN", None) and getattr(cfg.DEFRCN, "VIS_OUTPUT_DIR", None)
        if not self.vis_output_dir:
            self.vis_output_dir = getattr(cfg, "OUTPUT_DIR", "output")
        self.vis_output_dir = os.path.join(self.vis_output_dir, "defrcn_visuals")
        os.makedirs(self.vis_output_dir, exist_ok=True)

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
        assert not self.training
        _, _, results, image_sizes = self._forward_once_(batched_inputs, None)
        processed_results = []
        for idx, (r, input, image_size) in enumerate(zip(results, batched_inputs, image_sizes)):
            height = int(input.get("height", image_size[0]))
            width = int(input.get("width", image_size[1]))
            r = detector_postprocess(r, height, width)
            processed_results.append({"instances": r})

            # save visualizations + jsons for this single sample
            try:
                self._save_sample_visuals_and_json(
                    input, r, height, width, idx, image_size
                )
            except Exception as e:
                # Don't kill inference for visualization errors - but log once
                logging.exception(f"Failed to save visuals for sample idx={idx}: {e}")

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

    # ------------------- visualization & json helpers -------------------

    def _save_sample_visuals_and_json(self, input_dict, pred_instances: Instances, height: int, width: int, idx: int, image_size):
        """
        Save GT image, predictions image, matched predictions image and JSONs for a single sample.
        - input_dict: one element from batched_inputs (may contain "instances", "file_name", "image", "image_id", ...)
        - pred_instances: Instances returned by detector_postprocess for this image (already in target size coords)
        - height, width: the target size used by detector_postprocess
        - idx: index within the batch (used as fallback for naming)
        - image_size: original image size tuple (unused here but kept for compatibility)
        """
        try:
            # --- metadata & name base ---
            meta = None
            try:
                if len(self.cfg.DATASETS.TEST) > 0:
                    meta = MetadataCatalog.get(self.cfg.DATASETS.TEST[0])
            except Exception:
                meta = None

            def class_name(cl):
                if meta and hasattr(meta, "thing_classes") and meta.thing_classes:
                    try:
                        return meta.thing_classes[int(cl)]
                    except Exception:
                        return str(int(cl))
                else:
                    return str(int(cl)) if cl is not None else None

            file_name = input_dict.get("file_name", None)
            image_id = input_dict.get("image_id", None)
            if file_name:
                base = os.path.splitext(os.path.basename(file_name))[0]
            elif image_id is not None:
                base = f"img_{image_id}"
            else:
                base = f"img_{idx}"

            # --- helper utils ---
            def tensor_to_uint8_image(tensor):
                t = tensor.detach().cpu()
                if t.ndim == 3:
                    npimg = t.numpy().transpose(1, 2, 0)  # HWC
                elif t.ndim == 4 and t.shape[0] == 1:
                    npimg = t[0].numpy().transpose(1, 2, 0)
                else:
                    # Unexpected shape, try to squeeze
                    npimg = np.squeeze(t).numpy().transpose(1, 2, 0)
                if npimg.dtype == np.uint8:
                    return npimg
                # invert normalization. PIXEL_MEAN/STD are expected in same scale as tensors before norm.
                mean = np.array(self.cfg.MODEL.PIXEL_MEAN).reshape(1, 1, -1)
                std = np.array(self.cfg.MODEL.PIXEL_STD).reshape(1, 1, -1)
                npimg = (npimg * std + mean).clip(0, 255).astype(np.uint8)
                return npimg

            # --- obtain raw image, prefer input tensor, else file_name, else blank canvas ---
            raw_img = None
            if "image" in input_dict and input_dict["image"] is not None:
                try:
                    raw_img = tensor_to_uint8_image(input_dict["image"])
                except Exception:
                    raw_img = None

            if raw_img is None and file_name and os.path.exists(file_name):
                try:
                    pil = Image.open(file_name).convert("RGB")
                    raw_img = np.array(pil)
                except Exception:
                    raw_img = None

            if raw_img is None:
                # last resort blank image (size = target size)
                raw_img = np.zeros((height, width, 3), dtype=np.uint8)

            # Ensure it's HWC and 3 channels
            if raw_img.ndim == 2:
                raw_img = np.stack([raw_img] * 3, axis=-1)
            if raw_img.shape[2] == 4:
                raw_img = raw_img[:, :, :3]

            # Resize to detector_postprocess target size (width, height)
            try:
                pil_img = Image.fromarray(raw_img)
                if pil_img.size != (width, height):
                    pil_img = pil_img.resize((width, height), Image.BILINEAR)
                vis_img_np = np.array(pil_img).astype(np.uint8)
            except Exception:
                vis_img_np = raw_img
                if vis_img_np.shape[0] != height or vis_img_np.shape[1] != width:
                    vis_img_np = np.zeros((height, width, 3), dtype=np.uint8)

            H, W = height, width

            # --- extract GT boxes & classes if present in input_dict["instances"] ---
            gt_boxes_np = []
            gt_classes_np = []
            if "instances" in input_dict and input_dict["instances"] is not None:
                try:
                    gt_inst = input_dict["instances"].to("cpu")
                    # Prefer standard attributes
                    if hasattr(gt_inst, "gt_boxes"):
                        gt_boxes_np = gt_inst.gt_boxes.tensor.numpy().astype(float).tolist()
                    elif hasattr(gt_inst, "boxes"):
                        gt_boxes_np = gt_inst.boxes.tensor.numpy().astype(float).tolist()
                    # classes
                    if hasattr(gt_inst, "gt_classes"):
                        gt_classes_np = gt_inst.gt_classes.tensor.numpy().astype(int).tolist()
                    elif hasattr(gt_inst, "gt_classes"):
                        gt_classes_np = gt_inst.gt_classes.tensor.numpy().astype(int).tolist()
                    elif hasattr(gt_inst, "labels"):
                        try:
                            gt_classes_np = gt_inst.labels.tensor.numpy().astype(int).tolist()
                        except Exception:
                            gt_classes_np = []
                except Exception:
                    gt_boxes_np = []
                    gt_classes_np = []

            # --- extract predicted boxes/classes/scores from pred_instances (post-processed coords) ---
            pred_instances_cpu = pred_instances.to("cpu")
            pred_boxes_np = []
            pred_classes_np = []
            pred_scores_np = []
            try:
                if hasattr(pred_instances_cpu, "pred_boxes"):
                    pred_boxes_np = pred_instances_cpu.pred_boxes.tensor.numpy().astype(float).tolist()
            except Exception:
                pred_boxes_np = []
            try:
                if hasattr(pred_instances_cpu, "pred_classes"):
                    pred_classes_np = pred_instances_cpu.pred_classes.numpy().astype(int).tolist()
            except Exception:
                pred_classes_np = []
            try:
                if hasattr(pred_instances_cpu, "scores"):
                    pred_scores_np = pred_instances_cpu.scores.numpy().astype(float).tolist()
            except Exception:
                pred_scores_np = []

            # --- clip boxes to bounds and normalize lists ---
            def clip_box_list(box_list):
                clipped = []
                for b in box_list:
                    x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                    x1 = max(0.0, min(x1, W - 1.0))
                    x2 = max(0.0, min(x2, W - 1.0))
                    y1 = max(0.0, min(y1, H - 1.0))
                    y2 = max(0.0, min(y2, H - 1.0))
                    clipped.append([x1, y1, x2, y2])
                return clipped

            gt_boxes_np = clip_box_list(gt_boxes_np) if len(gt_boxes_np) > 0 else []
            pred_boxes_np = clip_box_list(pred_boxes_np) if len(pred_boxes_np) > 0 else []

            # --- compute matching (best IoU per prediction) ---
            matches = []
            for p_idx, p_box in enumerate(pred_boxes_np):
                best_iou = 0.0
                best_gt = -1
                for g_idx, g_box in enumerate(gt_boxes_np):
                    iou = compute_iou(np.array(p_box), np.array(g_box))
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = g_idx
                matches.append({"pred_idx": p_idx, "best_gt": best_gt, "best_iou": best_iou,
                                "matched": bool(best_iou >= 0.5)})

            # --- prepare JSON objects ---
            gt_json = {
                "id": base,
                "height": H,
                "width": W,
                "ground_truths": [
                    {"bbox": xyxy_to_list(np.array(box)), "class": class_name(c)}
                    for box, c in zip(gt_boxes_np, gt_classes_np)
                ],
            }

            preds_json = {
                "id": base,
                "height": H,
                "width": W,
                "predictions": []
            }
            for p_idx, box in enumerate(pred_boxes_np):
                c = pred_classes_np[p_idx] if p_idx < len(pred_classes_np) else None
                s = pred_scores_np[p_idx] if p_idx < len(pred_scores_np) else None
                m = matches[p_idx]
                preds_json["predictions"].append({
                    "bbox": xyxy_to_list(np.array(box)),
                    "class": class_name(c) if c is not None else None,
                    "score": float(s) if s is not None else None,
                    "matched_gt": int(m["best_gt"]) if m["best_gt"] >= 0 else None,
                    "matched_iou": float(m["best_iou"]),
                    "matched": bool(m["matched"])
                })

            # --- write json files ---
            gt_json_path = os.path.join(self.vis_output_dir, f"{base}_gt.json")
            preds_json_path = os.path.join(self.vis_output_dir, f"{base}_preds.json")
            try:
                with open(gt_json_path, "w") as f:
                    json.dump(gt_json, f, indent=2)
                with open(preds_json_path, "w") as f:
                    json.dump(preds_json, f, indent=2)
            except Exception:
                logging.exception(f"Failed to write JSONs for base={base}")

            # --- drawing utilities (robust to different detectron2 Visualizer API shapes) ---
            def build_instances_from_boxes(box_list, class_list=None, score_list=None):
                inst = Instances((H, W))
                if len(box_list) > 0:
                    inst.pred_boxes = Boxes(torch.tensor(box_list, dtype=torch.float32))
                else:
                    inst.pred_boxes = Boxes(torch.zeros((0, 4), dtype=torch.float32))
                if class_list is not None and len(class_list) == len(box_list):
                    inst.pred_classes = torch.tensor(class_list, dtype=torch.int64)
                else:
                    inst.pred_classes = torch.tensor([0] * len(box_list), dtype=torch.int64)
                if score_list is not None and len(score_list) == len(box_list):
                    inst.scores = torch.tensor(score_list, dtype=torch.float32)
                return inst

            # Utility to extract image from Visualizer return (varies by detectron2 version)
            def extract_vis_image(viz_obj, viz_instance):
                try:
                    out = viz_obj.draw_instance_predictions(viz_instance)
                    if hasattr(out, "get_image"):
                        return out.get_image()
                except Exception:
                    pass
                try:
                    if hasattr(viz_obj, "get_image"):
                        return viz_obj.get_image()
                except Exception:
                    pass
                try:
                    if hasattr(viz_obj, "output") and hasattr(viz_obj.output, "get_image"):
                        return viz_obj.output.get_image()
                except Exception:
                    pass
                # fallback
                return vis_img_np

            # --- draw GT image ---
            gt_image_path = os.path.join(self.vis_output_dir, f"{base}_gt.jpg")
            try:
                vgt = Visualizer(vis_img_np[:, :, ::-1], metadata=meta, scale=1.0, instance_mode=ColorMode.IMAGE)
                gt_inst = build_instances_from_boxes(gt_boxes_np, gt_classes_np, None)
                img_gt = extract_vis_image(vgt, gt_inst)
                Image.fromarray(img_gt).save(gt_image_path)
            except Exception:
                try:
                    Image.fromarray(vis_img_np).save(gt_image_path)
                except Exception:
                    logging.exception(f"Failed to save GT image for base={base}")

            # --- draw ALL predictions image ---
            preds_image_path = os.path.join(self.vis_output_dir, f"{base}_preds.jpg")
            try:
                vp = Visualizer(vis_img_np[:, :, ::-1], metadata=meta, scale=1.0, instance_mode=ColorMode.IMAGE)
                p_inst = build_instances_from_boxes(pred_boxes_np, pred_classes_np, pred_scores_np)
                img_preds = extract_vis_image(vp, p_inst)
                Image.fromarray(img_preds).save(preds_image_path)
            except Exception:
                try:
                    Image.fromarray(vis_img_np).save(preds_image_path)
                except Exception:
                    logging.exception(f"Failed to save preds image for base={base}")

            # --- draw only matched predictions ---
            matched_image_path = os.path.join(self.vis_output_dir, f"{base}_matched_preds.jpg")
            try:
                matched_boxes = [pred_boxes_np[m["pred_idx"]] for m in matches if m["matched"]]
                matched_classes = [pred_classes_np[m["pred_idx"]] for m in matches if m["matched"]]
                matched_scores = [pred_scores_np[m["pred_idx"]] for m in matches if m["matched"]]
                vm = Visualizer(vis_img_np[:, :, ::-1], metadata=meta, scale=1.0, instance_mode=ColorMode.IMAGE)
                m_inst = build_instances_from_boxes(matched_boxes, matched_classes, matched_scores)
                img_match = extract_vis_image(vm, m_inst)
                Image.fromarray(img_match).save(matched_image_path)
            except Exception:
                try:
                    Image.fromarray(vis_img_np).save(matched_image_path)
                except Exception:
                    logging.exception(f"Failed to save matched preds image for base={base}")

            # done
        except Exception:
            # do not crash main inference for visualization failure
            logging.exception(f"Unhandled error saving visuals for sample idx={idx}")


