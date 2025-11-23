import torch
import clip
import numpy as np
import os
from detectron2.data import MetadataCatalog
import defrcn.data.builtin  # Register datasets

# --- CONFIG ---
DATASET_NAME = "voc_2012_trainval_base1"
CLIP_MODEL_NAME = "ViT-B/32"  # Outputs 512-dim vectors
OUTPUT_PATH = "data/roifeats_base/clip_prototypes.npy"
# --------------

def generate_prototypes():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model: {CLIP_MODEL_NAME} on {device}...")
    model, _ = clip.load(CLIP_MODEL_NAME, device=device)
    
    # 1. Get Class Names
    try:
        metadata = MetadataCatalog.get(DATASET_NAME)
        class_names = metadata.thing_classes
        print(f"Found {len(class_names)} classes: {class_names}")
    except KeyError:
        print("Error: Dataset not registered.")
        return

    # 2. Create Prompts (Feature Engineering)
    # We average multiple templates for better robustness
    templates = [
        "a photo of a {}.",
        "a picture of a {}.",
        "a set of {}.",
        "images of {}."
    ]
    
    all_embeddings = []
    
    print("Generating embeddings...")
    with torch.no_grad():
        for name in class_names:
            texts = [t.format(name) for t in templates]
            text_tokens = clip.tokenize(texts).to(device)
            
            # Encode
            text_features = model.encode_text(text_tokens)
            
            # Normalize and Mean
            text_features /= text_features.norm(dim=-1, keepdim=True)
            embedding = text_features.mean(dim=0)
            embedding /= embedding.norm()
            
            all_embeddings.append(embedding.cpu().numpy())

    # Stack into matrix [N_classes, 512]
    prototypes = np.stack(all_embeddings)
    
    print(f"Saving prototypes with shape {prototypes.shape} to {OUTPUT_PATH}")
    np.save(OUTPUT_PATH, prototypes)

if __name__ == "__main__":
    generate_prototypes()