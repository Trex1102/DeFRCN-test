# extract_roifeats.py
import os
import torch
import numpy as np
import tqdm
from detectron2.engine import DefaultPredictor
from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
from detectron2.structures import Boxes


cfg = get_cfg()
cfg.merge_from_file("path/to/defrcn/config.yaml")
cfg.MODEL.WEIGHTS = "path/to/defrcn_checkpoint.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
predictor = DefaultPredictor(cfg)
model = predictor.model
model.eval()


dataset_name = "voc_2007_trainval_base1,voc_2012_trainval_base1"   # register or use existing
data_loader = build_detection_test_loader(cfg, dataset_name)

# ---- Where to save ----
out_dir = "data/roifeats_base"
os.makedirs(out_dir, exist_ok=True)
# we will save per-image npy files (or switch to LMDB/LMDB for scale)
# file naming: <image_id>.npz containing features + boxes + labels + confs

@torch.no_grad()
def extract():
    for idx, inputs in enumerate(tqdm.tqdm(data_loader)):
        
        outputs = model(inputs)

        for i, inp in enumerate(inputs):
            img_id = inp["image_id"] if "image_id" in inp else str(idx)+"_"+str(i)
            

            gt_boxes = inp["instances"].gt_boxes.tensor.clone().to(model.device)  # Nx4
            gt_classes = inp["instances"].gt_classes.cpu().numpy()  # Nx
            if len(gt_boxes)==0:
                continue


            images = model.backbone(inp["image"].to(model.device).unsqueeze(0))
            

            features = images  

            proposal_boxes = Boxes(gt_boxes)  


            pooled = model.roi_heads.box_pooler([features], [proposal_boxes])  

            box_features = model.roi_heads.box_head(pooled) 

            pred_logits, pred_bbox_deltas = model.roi_heads.box_predictor(box_features)
            scores = torch.softmax(pred_logits, dim=1).cpu().numpy()
            np.savez_compressed(
                os.path.join(out_dir, f"{img_id}.npz"),
                feats=box_features.cpu().numpy(),  # NxD
                boxes=gt_boxes.cpu().numpy(),      # Nx4
                classes=gt_classes,                # Nx
                scores=scores                      # NxC
            )

if __name__=="__main__":
    extract()
