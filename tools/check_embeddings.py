import torch
import numpy as np
import clip  # pip install git+https://github.com/openai/CLIP.git

# 1. Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_PATH = "data/roifeats_base/clip_prototypes_finetune.npy" # Path to your npy
loaded_prototypes = np.load(CLIP_PATH)
loaded_prototypes = torch.from_numpy(loaded_prototypes).to(device).float()

# 2. Define the ACTUAL class names your detector expects
# These must be in the order Detectron2 sees them (0, 1, 2...)
# Example for VOC Split 1 (check your dataset config!):
class_names = ["aeroplane", "bicycle", "boat", "bottle", "car", 
               "cat", "chair", "diningtable", "dog", "horse", 
               "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
               "bird", "bus", "cow", "motorbike"] 

# 3. Load CLIP Model
model, preprocess = clip.load("ViT-B/32", device=device)

# 4. Generate Fresh Embeddings
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_names]).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Normalize your loaded prototypes too
    loaded_prototypes /= loaded_prototypes.norm(dim=-1, keepdim=True)

# 5. Calculate Similarity Matrix
# [Num_Classes, Num_Loaded_Prototypes]
similarity = (text_features @ loaded_prototypes.T).cpu().numpy()

# 6. Check Alignment
print(f"\n--- Alignment Check ---")
for i, name in enumerate(class_names):
    # Find which index in the loaded file matches this class name best
    best_match_idx = np.argmax(similarity[i])
    score = similarity[i, best_match_idx]
    
    status = "OK" if i == best_match_idx else "MISMATCH"
    print(f"Class '{name}' (ID {i}) matches Loaded Index {best_match_idx} (Score: {score:.3f}) -> {status}")

    if i != best_match_idx:
        print(f"   >>> CRITICAL WARNING: Your code thinks Class {i} is '{name}', "
              f"but it is loading the vector for Index {i} which looks like something else.")