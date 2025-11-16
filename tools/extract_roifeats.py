# extract_roifeats_res5.py
import os
import torch
import numpy as np
import tqdm
from detectron2.engine import DefaultPredictor
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.config import get_cfg
from detectron2.structures import Boxes
import defrcn.data.builtin  # ensure dataset registration side-effects

# ---------- CONFIG ----------
cfg = get_cfg()
cfg.merge_from_file("configs/VAE-RCNN.yaml")
cfg.MODEL.WEIGHTS = "checkpoints/voc/1/defrcn_det_r101_base1/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0

predictor = DefaultPredictor(cfg)
model = predictor.model
model.eval()
device = next(model.parameters()).device

dataset_names = ["voc_2007_trainval_base1", "voc_2012_trainval_base1"]
out_dir = "data/roifeats_base"
os.makedirs(out_dir, exist_ok=True)

@torch.no_grad()
def extract_dataset(dataset_name):
    data_loader = build_detection_train_loader(cfg)
    # print(f"Extracting dataset {dataset_name} with {len(data_loader)} batches")
    for batch_idx, inputs in enumerate(tqdm.tqdm(data_loader)):
        # Preprocess and compute backbone features for the entire batch
        images = model.preprocess_image(inputs)          # ImageList on device
        features = model.backbone(images.tensor)         # OrderedDict of feature maps

        # Build list of Boxes per image using GT boxes (we extract GT RoIs)
        boxes_per_image = []
        counts = []
        for inp in inputs:
            inst = inp.get("instances", None)
            if inst is None or (not hasattr(inst, "gt_boxes")) or len(inst.gt_boxes) == 0:
                boxes_per_image.append(Boxes(torch.empty((0,4), device=device)))
                counts.append(0)
            else:
                gt_boxes = inst.gt_boxes.tensor.to(device)
                boxes_per_image.append(Boxes(gt_boxes))
                counts.append(len(gt_boxes))

        # Skip if this batch has no GT boxes
        if sum(counts) == 0:
            continue

        # IMPORTANT: Res5ROIHeads._shared_roi_transform expects list(features[f] for f in in_features)
        # so build that list (in_features is an attribute of roi_heads)
        in_features = model.roi_heads.in_features  # e.g. ['res4']
        feature_list = [features[f] for f in in_features]

        # Call the same transform used in Res5ROIHeads
        # This returns a tensor shaped (total_rois, C, H, W)
        pooled = model.roi_heads._shared_roi_transform(feature_list, boxes_per_image)

        # Now run the Res5 block and get the pooled 1x1 feature mean exactly as in forward:
        # Res5 already applied inside _shared_roi_transform for Res5ROIHeads; if not, the implementation above does res5 after pooler.
        # In your pasted class, _shared_roi_transform already does pooler -> res5
        # So pooled is the output (N, C, H, W)
        # Pool to 1x1 by taking spatial mean:
        try:
            # pooled may be on gpu; take mean across H,W as in model
            feature_pooled = pooled.mean(dim=[2, 3])  # (N_total, C)
        except Exception as e:
            raise RuntimeError("Pooled feature has unexpected shape or device. Details: " + str(e))

        # Compute classifier logits / scores using box_predictor
        # box_predictor usually returns (logits, bbox_deltas)
        logits = None
        scores = None
        try:
            with torch.no_grad():
                logits, deltas = model.roi_heads.box_predictor(feature_pooled)
                scores = torch.softmax(logits, dim=1).cpu().numpy()
        except Exception:
            # If box_predictor signature differs, try fallback calling with list input or other shapes
            try:
                logits, deltas = model.roi_heads.box_predictor(feature_pooled)
                scores = torch.softmax(logits, dim=1).cpu().numpy()
            except Exception as e:
                # give up computing scores but keep features
                scores = None

        # split features back into per-image groups using counts
        offset = 0
        for i, cnt in enumerate(counts):
            if cnt == 0:
                continue
            feats_i = feature_pooled[offset: offset + cnt].cpu().numpy()   # (cnt, C)
            inst = inputs[i].get("instances", None)
            if inst is None or (not hasattr(inst, "gt_boxes")) or len(inst.gt_boxes) == 0:
                offset += cnt
                continue
            gt_boxes = inst.gt_boxes.tensor.cpu().numpy()
            gt_classes = inst.gt_classes.cpu().numpy() if hasattr(inst, "gt_classes") else np.array([-1]*len(gt_boxes))

            img_id = inputs[i].get("image_id", None)
            if img_id is None:
                fname = inputs[i].get("file_name", None)
                if fname is not None:
                    img_id = os.path.splitext(os.path.basename(fname))[0]
                else:
                    img_id = f"{batch_idx}_{i}"

            save_path = os.path.join(out_dir, f"{dataset_name}__{img_id}.npz")
            np.savez_compressed(
                save_path,
                feats=feats_i.astype(np.float32),
                boxes=gt_boxes.astype(np.float32),
                classes=gt_classes.astype(np.int32),
                scores=scores[offset: offset + cnt] if (scores is not None) else None
            )
            offset += cnt

    print(f"Finished extracting dataset {dataset_name}")

def main():
    for ds in dataset_names:
        extract_dataset(ds)
    print("All datasets processed.")

if __name__ == "__main__":
    main()
