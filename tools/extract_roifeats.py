# extract_roifeats_res5_fixed.py
import os
import time
import torch
import numpy as np
import tqdm
from detectron2.engine import DefaultPredictor
from detectron2.data import build_detection_test_loader
from detectron2.config import get_cfg
from detectron2.structures import Boxes
import defrcn.data.builtin  # ensure dataset registration side-effects

# ---------- CONFIG (exact paths / values you provided) ----------
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
    """
    Extract RoI box-head features for GT boxes in dataset_name and save per-image .npz files
    into out_dir with filenames like: {dataset_name}__{img_id}.npz
    """
    
    data_loader = build_detection_test_loader(cfg, dataset_name)

    print(f"Extracting dataset {dataset_name} with approximately {len(data_loader)} batches")
    start_time_all = time.time()
    total_saved = 0
    t_last_print = time.time()

    for batch_idx, inputs in enumerate(tqdm.tqdm(data_loader, desc=f"Extract {dataset_name}")):
        t0 = time.time()
        
        images = model.preprocess_image(inputs)          
        features = model.backbone(images.tensor)         
        t_backbone = time.time()

        
        boxes_per_image = []
        counts = []
        img_ids = []
        for inp in inputs:
            inst = inp.get("instances", None)
            if inst is None or (not hasattr(inst, "gt_boxes")) or len(inst.gt_boxes) == 0:
                boxes_per_image.append(Boxes(torch.empty((0,4), device=device)))
                counts.append(0)
                img_ids.append(None)
            else:
                gt_boxes = inst.gt_boxes.tensor.to(device)
                boxes_per_image.append(Boxes(gt_boxes))
                counts.append(len(gt_boxes))

                img_id = inp.get("image_id", None)
                if img_id is None:
                    fname = inp.get("file_name", None)
                    if fname is not None:
                        img_id = os.path.splitext(os.path.basename(fname))[0]
                    else:
                        img_id = f"{batch_idx}_{len(img_ids)}"
                img_ids.append(str(img_id))

        
        if sum(counts) == 0:
            # small print occasionally
            if (batch_idx % 200) == 0:
                print(f"[{dataset_name}] batch {batch_idx}: no GT boxes, skipping")
            continue

        
        in_features = model.roi_heads.in_features  # e.g. ['res4']
        feature_list = [features[f] for f in in_features]

        
        try:
            pooled = model.roi_heads._shared_roi_transform(feature_list, boxes_per_image)
        except Exception:
            # fallback if API differs
            pooled = model.roi_heads.box_pooler(feature_list, boxes_per_image)

        t_pool = time.time()

        
        try:
            box_features = model.roi_heads.box_head(pooled)  
        except Exception:
            # fallback: flatten then call head
            pooled_flat = pooled.flatten(start_dim=1)
            box_features = model.roi_heads.box_head(pooled_flat)

        
        if box_features.ndim == 4 and box_features.shape[2] == 1 and box_features.shape[3] == 1:
            box_features = box_features.view(box_features.shape[0], -1)

        t_box_head = time.time()

        
        scores = None
        try:
            with torch.no_grad():
                logits, deltas = model.roi_heads.box_predictor(box_features)
                scores = torch.softmax(logits, dim=1).cpu().numpy()
        except Exception:
            # if predictor signature differs, continue but without scores
            scores = None

        # Split features back into per-image groups using counts
        box_features_cpu = box_features.cpu().numpy()
        offset = 0
        for i, cnt in enumerate(counts):
            if cnt == 0:
                continue
            feats_i = box_features_cpu[offset: offset + cnt]   # (cnt, C)
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

            # prepare scores slice carefully
            scores_slice = None
            if (scores is not None) and (offset + cnt <= scores.shape[0]):
                scores_slice = scores[offset: offset + cnt]
            else:
                scores_slice = None

            # Save compressed npz per image (feats, boxes, classes, optional scores)
            np.savez_compressed(
                save_path,
                feats=feats_i.astype(np.float32),
                boxes=gt_boxes.astype(np.float32),
                classes=gt_classes.astype(np.int32),
                scores=scores_slice.astype(np.float32) if (scores_slice is not None) else None
            )
            total_saved += cnt
            offset += cnt

        t_end = time.time()

        # periodic logging
        if (batch_idx % 50) == 0:
            print(
                f"[{dataset_name}] batch {batch_idx}: backbone {t_backbone - t0:.3f}s, pool {t_pool - t_backbone:.3f}s, "
                f"box_head {t_box_head - t_pool:.3f}s, save_time {t_end - t_box_head:.3f}s, total_saved {total_saved}"
            )

    total_time = time.time() - start_time_all
    print(f"Finished extracting dataset {dataset_name}. Total ROIs saved: {total_saved}. Time: {total_time:.1f}s")

def main():
    for ds in dataset_names:
        extract_dataset(ds)
    print("All datasets processed.")

if __name__ == "__main__":
    main()
