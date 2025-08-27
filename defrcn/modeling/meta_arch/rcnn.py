# import torch
# import logging
# from torch import nn
# from detectron2.structures import ImageList
# from detectron2.utils.logger import log_first_n
# from detectron2.modeling.backbone import build_backbone
# from detectron2.modeling.postprocessing import detector_postprocess
# from detectron2.modeling.proposal_generator import build_proposal_generator
# from .build import META_ARCH_REGISTRY
# from .gdl import decouple_layer, AffineLayer
# from defrcn.modeling.roi_heads import build_roi_heads

# __all__ = ["GeneralizedRCNN"]


# @META_ARCH_REGISTRY.register()
# class GeneralizedRCNN(nn.Module):

#     def __init__(self, cfg):
#         super().__init__()

#         self.cfg = cfg
#         self.device = torch.device(cfg.MODEL.DEVICE)
#         self.backbone = build_backbone(cfg)
#         self._SHAPE_ = self.backbone.output_shape()
#         self.proposal_generator = build_proposal_generator(cfg, self._SHAPE_)
#         self.roi_heads = build_roi_heads(cfg, self._SHAPE_)
#         self.normalizer = self.normalize_fn()
#         self.affine_rpn = AffineLayer(num_channels=self._SHAPE_['res4'].channels, bias=True)
#         self.affine_rcnn = AffineLayer(num_channels=self._SHAPE_['res4'].channels, bias=True)
#         self.to(self.device)

#         if cfg.MODEL.BACKBONE.FREEZE:
#             for p in self.backbone.parameters():
#                 p.requires_grad = False
#             print("froze backbone parameters")

#         if cfg.MODEL.RPN.FREEZE:
#             for p in self.proposal_generator.parameters():
#                 p.requires_grad = False
#             print("froze proposal generator parameters")

#         if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
#             for p in self.roi_heads.res5.parameters():
#                 p.requires_grad = False
#             print("froze roi_box_head parameters")

#     def forward(self, batched_inputs):
#         if not self.training:
#             return self.inference(batched_inputs)
#         assert "instances" in batched_inputs[0]
#         gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
#         proposal_losses, detector_losses, _, _ = self._forward_once_(batched_inputs, gt_instances)
#         losses = {}
#         losses.update(detector_losses)
#         losses.update(proposal_losses)
#         return losses

#     def inference(self, batched_inputs):
#         assert not self.training
#         _, _, results, image_sizes = self._forward_once_(batched_inputs, None)
#         processed_results = []
#         for r, input, image_size in zip(results, batched_inputs, image_sizes):
#             height = input.get("height", image_size[0])
#             width = input.get("width", image_size[1])
#             r = detector_postprocess(r, height, width)
#             processed_results.append({"instances": r})
#         return processed_results

#     def _forward_once_(self, batched_inputs, gt_instances=None):

#         images = self.preprocess_image(batched_inputs)
#         features = self.backbone(images.tensor)

#         features_de_rpn = features
#         if self.cfg.MODEL.RPN.ENABLE_DECOUPLE:
#             scale = self.cfg.MODEL.RPN.BACKWARD_SCALE
#             features_de_rpn = {k: self.affine_rpn(decouple_layer(features[k], scale)) for k in features}
#         proposals, proposal_losses = self.proposal_generator(images, features_de_rpn, gt_instances)

#         features_de_rcnn = features
#         if self.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
#             scale = self.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE
#             features_de_rcnn = {k: self.affine_rcnn(decouple_layer(features[k], scale)) for k in features}
#         results, detector_losses = self.roi_heads(images, features_de_rcnn, proposals, gt_instances)

#         return proposal_losses, detector_losses, results, images.image_sizes

#     def preprocess_image(self, batched_inputs):
#         images = [x["image"].to(self.device) for x in batched_inputs]
#         images = [self.normalizer(x) for x in images]
#         images = ImageList.from_tensors(images, self.backbone.size_divisibility)
#         return images

#     def normalize_fn(self):
#         assert len(self.cfg.MODEL.PIXEL_MEAN) == len(self.cfg.MODEL.PIXEL_STD)
#         num_channels = len(self.cfg.MODEL.PIXEL_MEAN)
#         pixel_mean = (torch.Tensor(
#             self.cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1))
#         pixel_std = (torch.Tensor(
#             self.cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1))
#         return lambda x: (x - pixel_mean) / pixel_std


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

import re

# helper used only inside this function
_float_re = re.compile(r"[-+]?\d*\.\d+|\d+")

def _extract_first_float(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x)
    m = _float_re.search(s)
    if m:
        try:
            return float(m.group(0))
        except:
            return None
    return None

def _safe_to_float(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x)
    m = _float_re.search(s)
    if m:
        try:
            return float(m.group(0))
        except:
            return None
    try:
        return float(s)
    except:
        return None

def parse_ann_bbox(ann, W, H):
    """
    Parse many VOC-like annotation formats and return (x1,y1,x2,y2) in pixel coords (floats)
    or None if unparseable.
    Handles:
      - ann['bndbox'] dict with xmin,ymin,xmax,ymax
      - ann with keys xmin,ymin,xmax,ymax
      - ann['bbox'] list (xyxy or xywh) -- heuristic converts xywh -> xyxy when needed
      - ann as list/tuple of 4 numbers
      - tolerates strings with extra text
    """
    # 1) bndbox dict
    if isinstance(ann, dict):
        if "bndbox" in ann and isinstance(ann["bndbox"], dict):
            bb = ann["bndbox"]
            xmin = _safe_to_float(bb.get("xmin") or bb.get("x") or bb.get("left"))
            ymin = _safe_to_float(bb.get("ymin") or bb.get("y") or bb.get("top"))
            xmax = _safe_to_float(bb.get("xmax") or bb.get("right"))
            ymax = _safe_to_float(bb.get("ymax") or bb.get("bottom"))
            if None not in (xmin, ymin, xmax, ymax):
                return xmin, ymin, xmax, ymax

        # 2) direct keys
        for kset in (("xmin","ymin","xmax","ymax"), ("x","y","x2","y2")):
            if all(k in ann for k in kset):
                xmin = _safe_to_float(ann[kset[0]])
                ymin = _safe_to_float(ann[kset[1]])
                xmax = _safe_to_float(ann[kset[2]])
                ymax = _safe_to_float(ann[kset[3]])
                if None not in (xmin, ymin, xmax, ymax):
                    return xmin, ymin, xmax, ymax

        # 3) ann['bbox'] (list or tuple)
        if "bbox" in ann:
            b = ann["bbox"]
            if isinstance(b, (list, tuple)) and len(b) == 4:
                b0 = [_safe_to_float(v) for v in b]
                if None in b0:
                    return None
                x0,y0,x1,y1 = b0
                # Heuristic: if x1> x0 and y1> y0 and x1<=W and y1<=H => xyxy
                if (x1 > x0 and y1 > y0 and x1 <= W and y1 <= H):
                    return x0, y0, x1, y1
                # otherwise treat as xywh
                return x0, y0, x0 + x1, y0 + y1

    # 4) if ann itself is list/tuple
    if isinstance(ann, (list, tuple)) and len(ann) == 4:
        b0 = [_safe_to_float(v) for v in ann]
        if None in b0:
            return None
        x0,y0,x1,y1 = b0
        if (x1 > x0 and y1 > y0 and x1 <= W and y1 <= H):
            return x0,y0,x1,y1
        return x0, y0, x0 + x1, y0 + y1

    return None



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

    def _save_sample_visuals_and_json(self, input_dict, pred_instances: Instances, height: int, width: int, idx: int, image_size):
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

            # --- image conversion helper ---
            def tensor_to_uint8_image(tensor):
                t = tensor.detach().cpu()
                if t.ndim == 3:
                    npimg = t.numpy().transpose(1, 2, 0)  # HWC
                elif t.ndim == 4 and t.shape[0] == 1:
                    npimg = t[0].numpy().transpose(1, 2, 0)
                else:
                    npimg = np.squeeze(t).numpy().transpose(1, 2, 0)
                if npimg.dtype == np.uint8:
                    return npimg
                mean = np.array(self.cfg.MODEL.PIXEL_MEAN).reshape(1, 1, -1)
                std = np.array(self.cfg.MODEL.PIXEL_STD).reshape(1, 1, -1)
                npimg = (npimg * std + mean).clip(0, 255).astype(np.uint8)
                return npimg

            # --- get image (tensor or file) ---
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

            # ---------------- Extract GT boxes & classes robustly ----------------
            gt_boxes_np = []
            gt_classes_np = []

            # 1) Try instances (standard Detectron2 Instances)
            if "instances" in input_dict and input_dict["instances"] is not None:
                try:
                    gt_inst = input_dict["instances"].to("cpu")
                    # Boxes
                    try:
                        if hasattr(gt_inst, "gt_boxes"):
                            boxes_tensor = gt_inst.gt_boxes.tensor
                        elif hasattr(gt_inst, "boxes"):
                            boxes_tensor = gt_inst.boxes.tensor
                        elif hasattr(gt_inst, "pred_boxes"):
                            boxes_tensor = gt_inst.pred_boxes.tensor
                        else:
                            boxes_tensor = None
                        if boxes_tensor is not None:
                            gt_boxes_np = boxes_tensor.numpy().astype(float).tolist()
                    except Exception:
                        gt_boxes_np = []

                    # Classes
                    try:
                        if hasattr(gt_inst, "gt_classes"):
                            classes_t = gt_inst.gt_classes
                        elif hasattr(gt_inst, "labels"):
                            classes_t = gt_inst.labels
                        elif hasattr(gt_inst, "classes"):
                            classes_t = gt_inst.classes
                        elif hasattr(gt_inst, "category_ids"):
                            classes_t = gt_inst.category_ids
                        else:
                            classes_t = None
                        if classes_t is not None:
                            if hasattr(classes_t, "numpy"):
                                gt_classes_np = classes_t.numpy().astype(int).tolist()
                            else:
                                gt_classes_np = [int(x) for x in classes_t]
                    except Exception:
                        gt_classes_np = []
                except Exception:
                    gt_boxes_np = []
                    gt_classes_np = []

            if (not gt_boxes_np) and ("annotations" in input_dict and input_dict["annotations"]):
                try:
                    anns = input_dict["annotations"]
                    parsed_boxes = []
                    parsed_classes = []
                    for ann in anns:
                        # DEBUG print raw ann (first): helps identify weird fields causing problems
                        logging.info(f"[ANN RAW] base={base} keys={list(ann.keys())} sample_preview={ {k: ann[k] for k in list(ann.keys())[:5]} }")

                        parsed = parse_ann_bbox(ann, W, H)
                        if parsed is None:
                            logging.warning(f"[ANN SKIP] base={base} could not parse ann {ann}")
                            continue

                        # print parsed before clipping so you can see original coords
                        logging.info(f"[ANN PARSED BEFORE CLIP] base={base} parsed={parsed}")

                        x1, y1, x2, y2 = parsed
                        # now clip safely
                        x1 = max(0.0, min(x1, W - 1.0))
                        x2 = max(0.0, min(x2, W - 1.0))
                        y1 = max(0.0, min(y1, H - 1.0))
                        y2 = max(0.0, min(y2, H - 1.0))

                        # if after clipping the box has zero area, log it and skip or keep depending on you
                        if x2 <= x1 or y2 <= y1:
                            logging.warning(f"[ANN COLLAPSED] base={base} parsed_after_clip={(x1,y1,x2,y2)}; skipping")
                            continue

                        parsed_boxes.append([x1, y1, x2, y2])

                        # category extraction (VOC often has 'name' or 'category_id')
                        cat = ann.get("category_id", None) or ann.get("category", None) or ann.get("name", None) or ann.get("label", None)
                        if cat is None:
                            parsed_classes.append(None)
                        else:
                            if isinstance(cat, str):
                                if meta and hasattr(meta, "thing_classes") and meta.thing_classes:
                                    try:
                                        parsed_classes.append(int(meta.thing_classes.index(cat)))
                                    except Exception:
                                        parsed_classes.append(None)
                                else:
                                    parsed_classes.append(None)
                            else:
                                try:
                                    parsed_classes.append(int(cat))
                                except Exception:
                                    parsed_classes.append(None)

                    if parsed_boxes:
                        gt_boxes_np = parsed_boxes
                        gt_classes_np = [c if c is not None else None for c in parsed_classes]
                except Exception:
                    logging.exception(f"[ANN PARSE FAIL] base={base} exception during annotation parsing")


            # final clip (in case)
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


            # sanitize gt boxes -> ensure floats, clip, drop bad boxes
            _clean_boxes = []
            _clean_classes = []
            for i, raw_box in enumerate(gt_boxes_np):
                # raw_box may be list of mixed types or strings like "499.0JS:499"
                nums = [_extract_first_float(v) for v in raw_box]
                if any(v is None for v in nums) or len(nums) != 4:
                    logging.warning(f"[GT SANITIZE] base={base} skipping box {i} because coords not numeric: {raw_box}")
                    continue
                x1, y1, x2, y2 = nums
                # convert possible xywh -> detect and convert if necessary
                # heuristic: if x2 <= x1 or y2 <= y1, treat as (x,y,w,h)
                if x2 <= x1 or y2 <= y1:
                    # treat as xywh
                    w = x2
                    h = y2
                    x2 = x1 + w
                    y2 = y1 + h
                # Clip to image
                x1 = max(0.0, min(x1, W - 1.0))
                x2 = max(0.0, min(x2, W - 1.0))
                y1 = max(0.0, min(y1, H - 1.0))
                y2 = max(0.0, min(y2, H - 1.0))
                # drop degenerate boxes
                if x2 <= x1 or y2 <= y1:
                    logging.warning(f"[GT SANITIZE] base={base} skipped collapsed box after clip: {(x1,y1,x2,y2)} from raw {raw_box}")
                    continue
                _clean_boxes.append([float(x1), float(y1), float(x2), float(y2)])
                # keep class alignment (pad with None if necessary)
                cls = gt_classes_np[i] if i < len(gt_classes_np) else None
                _clean_classes.append(cls)

            # replace the GT lists with cleaned ones
            gt_boxes_np = _clean_boxes
            gt_classes_np = _clean_classes
            logging.info(f"[GT SANITIZE] base={base} final GT boxes count = {len(gt_boxes_np)}")


            gt_boxes_np = clip_box_list(gt_boxes_np) if len(gt_boxes_np) > 0 else []
            # If there are fewer class entries than boxes, pad with None
            if len(gt_classes_np) < len(gt_boxes_np):
                gt_classes_np = list(gt_classes_np) + [None] * (len(gt_boxes_np) - len(gt_classes_np))

            # Debug log so you can see what we found
            logging.info(f"[vis] base={base} found {len(gt_boxes_np)} GT boxes and {len(pred_instances)} predictions")

            # ---------------- Extract predictions (same as before) ----------------
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
                else:
                    pred_classes_np = []
            except Exception:
                pred_classes_np = []
            try:
                if hasattr(pred_instances_cpu, "scores"):
                    pred_scores_np = pred_instances_cpu.scores.numpy().astype(float).tolist()
            except Exception:
                pred_scores_np = []

            pred_boxes_np = clip_box_list(pred_boxes_np) if len(pred_boxes_np) > 0 else []

            # --- matching (per-pred best IoU; unchanged) ---
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

            # ---------------- build JSONs ----------------
            gt_json = {"id": base, "height": H, "width": W, "ground_truths": []}
            for idx_box, box in enumerate(gt_boxes_np):
                cls_name = None
                if idx_box < len(gt_classes_np) and gt_classes_np[idx_box] is not None:
                    try:
                        cls_name = class_name(gt_classes_np[idx_box])
                    except Exception:
                        cls_name = None
                gt_json["ground_truths"].append({"bbox": xyxy_to_list(np.array(box)), "class": cls_name})

            preds_json = {"id": base, "height": H, "width": W, "predictions": []}
            for p_idx, box in enumerate(pred_boxes_np):
                c = pred_classes_np[p_idx] if p_idx < len(pred_classes_np) else None
                s = pred_scores_np[p_idx] if p_idx < len(pred_scores_np) else None
                m = matches[p_idx] if p_idx < len(matches) else {"best_gt": -1, "best_iou": 0.0, "matched": False}
                preds_json["predictions"].append({
                    "bbox": xyxy_to_list(np.array(box)),
                    "class": class_name(c) if c is not None else None,
                    "score": float(s) if s is not None else None,
                    "matched_gt": int(m["best_gt"]) if m["best_gt"] >= 0 else None,
                    "matched_iou": float(m["best_iou"]),
                    "matched": bool(m["matched"])
                })

            gt_json_path = os.path.join(self.vis_output_dir, f"{base}_gt.json")
            preds_json_path = os.path.join(self.vis_output_dir, f"{base}_preds.json")
            try:
                with open(gt_json_path, "w") as f:
                    json.dump(gt_json, f, indent=2)
                with open(preds_json_path, "w") as f:
                    json.dump(preds_json, f, indent=2)
            except Exception:
                logging.exception(f"Failed to write JSONs for base={base}")

            # ---------------- drawing utilities & saving images (unchanged) ----------------
            def build_instances_from_boxes(box_list, class_list=None, score_list=None):
                inst = Instances((H, W))
                if len(box_list) > 0:
                    boxes_tensor = torch.tensor(box_list, dtype=torch.float32)
                    inst.pred_boxes = Boxes(boxes_tensor)
                    inst.gt_boxes = Boxes(boxes_tensor)
                else:
                    inst.pred_boxes = Boxes(torch.zeros((0, 4), dtype=torch.float32))
                    inst.gt_boxes = Boxes(torch.zeros((0, 4), dtype=torch.float32))
                if class_list is not None and len(class_list) == len(box_list):
                    # replace None by 0 for class tensor, Visualizer expects an int tensor
                    classes_tensor = torch.tensor([0 if c is None else int(c) for c in class_list], dtype=torch.int64)
                    inst.pred_classes = classes_tensor
                    inst.gt_classes = classes_tensor
                else:
                    inst.pred_classes = torch.tensor([], dtype=torch.int64)
                if score_list is not None and len(score_list) == len(box_list):
                    inst.scores = torch.tensor(score_list, dtype=torch.float32)
                return inst

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
                return vis_img_np

            # Save GT image
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

            # Save preds image
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

            # Save matched preds image (only matched preds)
            matched_image_path = os.path.join(self.vis_output_dir, f"{base}_matched_preds.jpg")
            try:
                matched_boxes = [pred_boxes_np[m["pred_idx"]] for m in matches if m["matched"]]
                matched_classes = [pred_classes_np[m["pred_idx"]] for m in matches if m["matched"] and m["pred_idx"] < len(pred_classes_np)]
                matched_scores = [pred_scores_np[m["pred_idx"]] for m in matches if m["matched"] and m["pred_idx"] < len(pred_scores_np)]
                vm = Visualizer(vis_img_np[:, :, ::-1], metadata=meta, scale=1.0, instance_mode=ColorMode.IMAGE)
                m_inst = build_instances_from_boxes(matched_boxes, matched_classes, matched_scores)
                img_match = extract_vis_image(vm, m_inst)
                Image.fromarray(img_match).save(matched_image_path)
            except Exception:
                try:
                    Image.fromarray(vis_img_np).save(matched_image_path)
                except Exception:
                    logging.exception(f"Failed to save matched preds image for base={base}")

        except Exception:
            logging.exception(f"Unhandled error saving visuals for sample idx={idx}")

        exit(1)
            


