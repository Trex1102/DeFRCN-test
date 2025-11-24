# extract_roifeats_res5_fixed.py
import os
import time
import torch
import numpy as np
import tqdm
from detectron2.engine import DefaultPredictor
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.config import get_cfg
from detectron2.structures import Boxes
import defrcn.data.builtin  # ensure dataset registration side-effects
from detectron2.data import DatasetMapper

# ---------- CONFIG (exact paths / values you provided) ----------
cfg = get_cfg()
cfg.merge_from_file("configs/VAE-RCNN.yaml")
cfg.MODEL.WEIGHTS = "checkpoints/voc/1/defrcn_det_r101_base1/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0

predictor = DefaultPredictor(cfg)
model = predictor.model
model.eval()
device = next(model.parameters()).device

dataset_names = ["voc_2007_trainval_base1","voc_2012_trainval_base1"]


out_dir = "data/roifeats_base"
os.makedirs(out_dir, exist_ok=True)

@torch.no_grad()
def extract_dataset(dataset_name):
    print(f"Extracting dataset {dataset_name}...")
    
    # 1. Setup Mapper (No augmentation, use GT)
    mapper = DatasetMapper(cfg, is_train=True, augmentations=[])
    data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
    
    start_time_all = time.time()
    
    all_feats = []
    all_boxes = []
    all_classes = []
    all_scores = []
    all_img_ids = []  

    for batch_idx, inputs in enumerate(tqdm.tqdm(data_loader, desc=f"Extract {dataset_name}")):
        
        images = model.preprocess_image(inputs)          
        features = model.backbone(images.tensor)         

        # --- DEBUG: Uncomment if error persists to see available keys ---
        # if batch_idx == 0:
        #     print(f"Available feature keys: {features.keys()}") 
        #     for k, v in features.items():
        #         print(f"Key {k} shape: {v.shape}")

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

        if sum(counts) == 0:
            continue

        # ---------------------------------------------------------
        # CRITICAL FIX: Force usage of 'res4' for Res5ROIHeads
        # ---------------------------------------------------------
        feature_list = []
        
        # We prefer 'res4' (1024 channels) because the Head is 'res5'
        if "res4" in features:
            feature_list = [features["res4"]]
        elif "res5" in features:
            # Fallback: If ONLY res5 exists (2048ch), we cannot run the Head again.
            # We must skip the head and just use the pooled features.
            feature_list = [features["res5"]]
        else:
            # Fallback to config default
            in_features = model.roi_heads.in_features
            feature_list = [features[f] for f in in_features]
        
        # 1. ROI Align (Crop)
        if hasattr(model.roi_heads, "_shared_roi_transform"):
            pooled = model.roi_heads._shared_roi_transform(feature_list, boxes_per_image)
        else:
            pooled = model.roi_heads.box_pooler(feature_list, boxes_per_image)

        # 2. Apply Head (Conditional based on input channels)
        # Check channel dimension (dim 1)
        current_channels = pooled.shape[1]

        if hasattr(model.roi_heads, "res5") and current_channels == 1024:
            # Normal Case: Input is res4 (1024), pass through res5 Head
            x = model.roi_heads.res5(pooled)  
            box_features = x.mean(dim=[2, 3]) 
        else:
            # Fallback Case: Input is already res5 (2048), just pool it
            # We cannot pass 2048 ch into a layer expecting 1024.
            if pooled.ndim == 4:
                box_features = pooled.mean(dim=[2, 3])
            else:
                box_features = pooled

        # Ensure 2D shape [N, C]
        if box_features.ndim > 2:
            box_features = box_features.flatten(start_dim=1)

        # 3. Get Scores
        scores = None
        try:
            with torch.no_grad():
                logits, deltas = model.roi_heads.box_predictor(box_features)
                scores = torch.softmax(logits, dim=1).cpu().numpy()
        except Exception:
            scores = None

        # --- Accumulate Data ---
        box_features_cpu = box_features.cpu().numpy()
        offset = 0
        
        for i, cnt in enumerate(counts):
            if cnt == 0: continue
                
            feats_i = box_features_cpu[offset: offset + cnt]
            all_feats.append(feats_i)
            
            inst = inputs[i]["instances"]
            gt_boxes = inst.gt_boxes.tensor.cpu().numpy()
            if hasattr(inst, "gt_classes"):
                gt_classes = inst.gt_classes.cpu().numpy()
            else:
                gt_classes = np.zeros(cnt, dtype=np.int32)
            
            all_boxes.append(gt_boxes)
            all_classes.append(gt_classes)

            if scores is not None:
                scores_slice = scores[offset: offset + cnt]
                all_scores.append(scores_slice)

            img_id = inputs[i].get("image_id", None)
            if img_id is None:
                fname = inputs[i].get("file_name", None)
                img_id = os.path.splitext(os.path.basename(fname))[0] if fname else f"{batch_idx}_{i}"
            
            all_img_ids.append(np.array([str(img_id)] * cnt))

            offset += cnt

    # --- Save ---
    if len(all_feats) == 0:
        print(f"No features extracted for {dataset_name}.")
        return

    print("Concatenating arrays...")
    final_feats = np.concatenate(all_feats, axis=0)       
    final_boxes = np.concatenate(all_boxes, axis=0)       
    final_classes = np.concatenate(all_classes, axis=0)   
    final_ids = np.concatenate(all_img_ids, axis=0)       
    
    final_scores = None
    if len(all_scores) > 0:
        final_scores = np.concatenate(all_scores, axis=0) 

    save_path = os.path.join(out_dir, f"{dataset_name}_combined.npz")
    
    print(f"Saving to {save_path} ...")
    np.savez_compressed(
        save_path,
        feats=final_feats.astype(np.float32),
        boxes=final_boxes.astype(np.float32),
        classes=final_classes.astype(np.int32),
        image_ids=final_ids,
        scores=final_scores.astype(np.float32) if final_scores is not None else None
    )

    total_time = time.time() - start_time_all
    print(f"Finished. Total objects: {final_feats.shape[0]}. Time: {total_time:.1f}s")

def main():
    for ds in dataset_names:
        extract_dataset(ds)
    print("All datasets processed.")

if __name__ == "__main__":
    main()
