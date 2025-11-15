# extract_roifeats.py
import os
import torch
import numpy as np
import tqdm
from detectron2.engine import DefaultPredictor
from detectron2.data import build_detection_test_loader
from detectron2.config import get_cfg
from detectron2.structures import Boxes

# ---------------- CONFIG ----------------
cfg = get_cfg()
cfg.merge_from_file("configs/VAE-RCNN.yaml")
cfg.MODEL.WEIGHTS = "checkpoints/voc/1/defrcn_det_r101_base1/model_final.pth"
# set this if you want to keep very low-score RoIs (optional)
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0

predictor = DefaultPredictor(cfg)
model = predictor.model
model.eval()
device = next(model.parameters()).device

# Provide a list of dataset names (registered with DatasetCatalog).
# Do NOT pass a single comma-separated string.
dataset_names = ["voc_2007_trainval_base1", "voc_2012_trainval_base1"]

out_dir = "data/roifeats_base"
os.makedirs(out_dir, exist_ok=True)

@torch.no_grad()
def extract_dataset(dataset_name):
    data_loader = build_detection_test_loader(cfg, dataset_name)
    print(f"Extracting features for dataset: {dataset_name}")
    for batch_idx, inputs in enumerate(tqdm.tqdm(data_loader)):
        # inputs is a list of dicts (one per image)
        # We process each image in the batch individually (keeps memory low)
        for i, inp in enumerate(inputs):
            # safe image id / filename
            img_id = inp.get("image_id", None)
            if img_id is None:
                img_id = inp.get("file_name", f"{batch_idx}_{i}")
                # make filename-safe
                img_id = os.path.splitext(os.path.basename(img_id))[0]
            save_path = os.path.join(out_dir, f"{dataset_name}__{img_id}.npz")

            # get GT boxes and classes (skip if none)
            instances = inp.get("instances", None)
            if instances is None:
                continue
            gt_boxes = instances.gt_boxes.tensor if hasattr(instances, "gt_boxes") else None
            gt_classes = instances.gt_classes if hasattr(instances, "gt_classes") else None
            if gt_boxes is None or len(gt_boxes) == 0:
                continue

            # Preprocess & backbone forward (use the same preprocessing as Detectron2)
            # model.preprocess_image expects a list of dicts
            img_list = model.preprocess_image([inp])  # returns ImageList
            # backbone expects the batched tensor (ImageList.tensor)
            features = model.backbone(img_list.tensor)  # OrderedDict[str -> Tensor], tensors have batch dim

            # Build Boxes on device
            proposal_boxes = Boxes(gt_boxes.to(device))

            # box_pooler expects list of feature dicts and list of Boxes
            # Passing [features] and [proposal_boxes] follows the pattern used in roi_heads.forward
            pooled = model.roi_heads.box_pooler([features], [proposal_boxes])  # (N, C, H, W)
            # Run the box_head to get the RoI feature vector (same as during training)
            box_features = model.roi_heads.box_head(pooled)  # (N, D)

            # Get classifier logits (optional) using the box_predictor
            try:
                pred_logits, pred_deltas = model.roi_heads.box_predictor(box_features)
                scores = torch.softmax(pred_logits, dim=1).cpu().numpy()
            except Exception:
                scores = None

            # move to CPU and save
            np.savez_compressed(
                save_path,
                feats=box_features.cpu().numpy(),      # (N, D)
                boxes=gt_boxes.cpu().numpy(),          # (N, 4)
                classes=gt_classes.cpu().numpy(),      # (N,)
                scores=scores                          # (N, num_classes) or None
            )

def main():
    for ds in dataset_names:
        extract_dataset(ds)
    print("Extraction finished.")

if __name__ == "__main__":
    main()
