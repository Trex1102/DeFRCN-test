import torch
import clip
import numpy as np
from detectron2.data import MetadataCatalog


# --- CONFIG ---
DATASET_NAME = "voc_2012_trainval_all1"
CLIP_MODEL_NAME = "ViT-B/32"
OUTPUT_PATH = "data/roifeats_base/clip_prototypes_finetune.npy"
# --------------

# Standard Pascal VOC 20 classes (fixed canonical order)
PASCAL_VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


def get_voc_classes(metadata):
    """Ensure class ordering strictly matches Pascal VOC canonical order."""
    if hasattr(metadata, "thing_classes"):
        meta_classes = metadata.thing_classes
        if set(meta_classes) == set(PASCAL_VOC_CLASSES):
            print("Using class order from Detectron2 Metadata (VOC).")
            return meta_classes
        else:
            print("⚠ WARNING: Metadata class order mismatch. Falling back to canonical Pascal VOC order.")
    else:
        print("⚠ WARNING: No thing_classes found. Using canonical Pascal VOC order.")

    return PASCAL_VOC_CLASSES


def generate_prototypes():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model {CLIP_MODEL_NAME} on {device}...")
    model, _ = clip.load(CLIP_MODEL_NAME, device=device)

    # 1. Load dataset metadata
    try:
        metadata = MetadataCatalog.get(DATASET_NAME)
    except KeyError:
        print(f"Error: Dataset {DATASET_NAME} is not registered.")
        return

    # 2. Get Pascal VOC classes in correct order
    class_names = get_voc_classes(metadata)
    print(f"Final class order ({len(class_names)}): {class_names}")

    # 3. Prompt templates
    templates = [
        "a photo of a {}.",
        "a picture of a {}.",
        "a close-up photo of a {}.",
        "a cropped image of a {}."
    ]

    all_embeddings = []

    print("Generating CLIP text embeddings...")
    with torch.no_grad():
        for cname in class_names:

            prompts = [t.format(cname) for t in templates]
            tokens = clip.tokenize(prompts).to(device)

            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            mean_embed = text_features.mean(dim=0)
            mean_embed = mean_embed / mean_embed.norm()

            all_embeddings.append(mean_embed.cpu().numpy())

    prototypes = np.stack(all_embeddings)

    print(f"Saving prototypes: shape={prototypes.shape} → {OUTPUT_PATH}")
    np.save(OUTPUT_PATH, prototypes)

    print("Done.")


if __name__ == "__main__":
    generate_prototypes()
