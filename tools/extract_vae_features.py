# extract_vae_features_clip.py
import os
import time
import torch
import numpy as np
import tqdm

from detectron2.engine import DefaultPredictor
from detectron2.data import build_detection_test_loader
from detectron2.config import get_cfg
from detectron2.structures import Boxes
import defrcn.data.builtin  # dataset registers automatically


# --------------------------------------------------------
# 1. VOC CLASS NAMES
# --------------------------------------------------------
VOC_CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


# --------------------------------------------------------
# 2. I. BUILD CLIP SEMANTIC EMBEDDINGS
# --------------------------------------------------------
def build_clip_embeddings(class_names, clip_model_name="ViT-B/32", device="cuda"):
    """
    Returns np.array shape (20, clip_dim), normalized.
    """
    try:
        import clip
    except Exception as e:
        raise RuntimeError(
            "Install CLIP: pip install git+https://github.com/openai/CLIP.git"
        ) from e

    model, _ = clip.load(clip_model_name, device=device)
    model.eval()

    texts = [f"a photo of a {c}" for c in class_names]

    with torch.no_grad():
        tokens = clip.tokenize(texts).to(device)
        text_emb = model.encode_text(tokens)
        text_emb = text_emb / text_emb.norm(dim=1, keepdim=True)
        return text_emb.cpu().numpy()  # (20,512) if ViT-B/32


# --------------------------------------------------------
# 2. II. Semantic embedding lookup
# --------------------------------------------------------
def get_semantic_embedding(class_id):
    return CLIP_EMB[class_id]


# --------------------------------------------------------
# 3. PAIRWISE IOU
# --------------------------------------------------------
def pairwise_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    return inter / union


# --------------------------------------------------------
# 4. DETECTRON2 + DEFRCN SETUP
# --------------------------------------------------------
cfg = get_cfg()
cfg.merge_from_file("configs/VAE-RCNN.yaml")
cfg.MODEL.WEIGHTS = "checkpoints/voc/1/defrcn_det_r101_base1/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0

predictor = DefaultPredictor(cfg)
model = predictor.model
model.eval()
device = next(model.parameters()).device

dataset_names = ["voc_2007_trainval_base1", "voc_2012_trainval_base1"]


# --------------------------------------------------------
# 5. CREATE CLIP EMBEDDINGS ONCE
# --------------------------------------------------------
print("Building CLIP semantic embeddings...")
CLIP_EMB = build_clip_embeddings(VOC_CLASS_NAMES, clip_model_name="ViT-B/32", device=device)
print("CLIP embeddings loaded:", CLIP_EMB.shape)


# --------------------------------------------------------
# 6. OUTPUT STORAGE (one file)
# --------------------------------------------------------
out_path = "data/roifeats_base/roi_features_all.npz"
os.makedirs(os.path.dirname(out_path), exist_ok=True)

ALL_FEATS = []
ALL_CLASSES = []
ALL_BOXES = []
ALL_IOU = []
ALL_SEM = []
ALL_IMGID = []


# --------------------------------------------------------
# 7. MAIN EXTRACTION LOOP
# --------------------------------------------------------
@torch.no_grad()
def extract_dataset(dataset_name):

    global ALL_FEATS, ALL_CLASSES, ALL_BOXES, ALL_IOU, ALL_SEM, ALL_IMGID

    data_loader = build_detection_test_loader(cfg, dataset_name)
    print(f"\n==== Extracting {dataset_name} ====\n")

    for batch_idx, inputs in enumerate(tqdm.tqdm(data_loader)):
        images = model.preprocess_image(inputs)
        features = model.backbone(images.tensor)

        # Collect GT info
        gt_info = []
        for inp in inputs:
            inst = inp.get("instances", None)
            if inst is None or len(inst.gt_boxes) == 0:
                gt_info.append((None, None, None))
            else:
                gt_boxes = inst.gt_boxes.tensor.to(device)
                gt_classes = inst.gt_classes.to(device)
                img_id = inp.get("image_id", -1)
                gt_info.append((gt_boxes, gt_classes, img_id))

        # RPN proposals
        proposals, _ = model.rpn(images, features)

        # ROI pool all proposals
        all_props = [p.proposal_boxes for p in proposals]
        feat_list = [features[f] for f in model.roi_heads.in_features]
        pooled = model.roi_heads._shared_roi_transform(feat_list, all_props)

        box_feats = model.roi_heads.box_head(pooled)
        if box_feats.ndim == 4:
            box_feats = box_feats.view(box_feats.size(0), -1)
        box_feats = box_feats.cpu()

        # Split into per image
        offset = 0
        for img_idx, (gt_boxes, gt_classes, img_id) in enumerate(gt_info):

            prop = proposals[img_idx].proposal_boxes.tensor.to(device)
            P = len(prop)

            if gt_boxes is None or P == 0:
                offset += P
                continue

            prop_feats = box_feats[offset:offset+P]
            offset += P

            # IoU(proposals, gt_boxes)
            iou_mat = pairwise_iou(prop, gt_boxes)
            max_iou, max_gt_idx = torch.max(iou_mat, dim=1)

            keep = max_iou >= 0.5
            if keep.sum() == 0:
                continue

            kept_props = prop[keep]
            kept_feats = prop_feats[keep]
            kept_iou = max_iou[keep]
            kept_gt = gt_classes[max_gt_idx[keep]]

            # Store data
            for k in range(len(kept_props)):
                cls_id = int(kept_gt[k].item())
                sem_vec = get_semantic_embedding(cls_id)

                ALL_FEATS.append(kept_feats[k].numpy())
                ALL_CLASSES.append(cls_id)
                ALL_BOXES.append(kept_props[k].cpu().numpy())
                ALL_IOU.append(float(kept_iou[k].item()))
                ALL_SEM.append(sem_vec)
                ALL_IMGID.append(int(img_id))


# --------------------------------------------------------
# 8. RUN + SAVE NPZ
# --------------------------------------------------------
def main():
    for ds in dataset_names:
        extract_dataset(ds)

    print("\nSaving unified NPZ...")
    np.savez_compressed(
        out_path,
        feats=np.array(ALL_FEATS, dtype=np.float32),
        classes=np.array(ALL_CLASSES, dtype=np.int32),
        boxes=np.array(ALL_BOXES, dtype=np.float32),
        iou=np.array(ALL_IOU, dtype=np.float32),
        semantics=np.array(ALL_SEM, dtype=np.float32),
        img_ids=np.array(ALL_IMGID, dtype=np.int32)
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
