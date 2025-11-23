import numpy as np
import matplotlib.pyplot as plt
import os
from detectron2.data import MetadataCatalog

# ---------- CONFIG ----------
# MUST match the dataset name you used for extraction
DATASET_NAME = "voc_2012_trainval_base1" 
FILE_PATH = "data/roifeats_base/voc_2007_trainval_base1_combined.npz"
# ----------------------------

def analyze_dataset():
    if not os.path.exists(FILE_PATH):
        print(f"Error: File not found at {FILE_PATH}")
        return

    print(f"Loading {FILE_PATH}...")
    data = np.load(FILE_PATH)
    classes = data['classes']
    
    # --- FIX: Get the ACTUAL class names for this specific split ---
    try:
        metadata = MetadataCatalog.get(DATASET_NAME)
        actual_class_names = metadata.thing_classes
        print(f"Loaded {len(actual_class_names)} class names from metadata.")
    except KeyError:
        print(f"Error: Dataset '{DATASET_NAME}' not registered. Make sure you import defrcn/detectron2.")
        return

    # Count occurrences
    unique_ids, counts = np.unique(classes, return_counts=True)
    
    plot_names = []
    plot_counts = []

    print(f"\n{'ID':<5} {'Name':<20} {'Count':<10}")
    print("-" * 40)

    for cls_id, count in zip(unique_ids, counts):
        if 0 <= cls_id < len(actual_class_names):
            name = actual_class_names[cls_id]
        else:
            name = f"Unknown ({cls_id})"
            
        plot_names.append(name)
        plot_counts.append(count)
        print(f"{cls_id:<5} {name:<20} {count:<10}")

    # --- Plotting ---
    plt.figure(figsize=(14, 8)) # Increased height for readability
    bars = plt.bar(plot_names, plot_counts, color='steelblue', edgecolor='black')
    
    # Add counts on top
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 50, int(yval), 
                 ha='center', va='bottom', fontsize=9, rotation=0)

    plt.title(f'Corrected Class Distribution: {DATASET_NAME}')
    plt.xlabel('Class Name')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig("class_distribution_corrected.png")
    print(f"\nPlot saved to class_distribution_corrected.png")
    plt.show()

if __name__ == "__main__":
    # Ensure dataset is registered (DeFRCN specific)
    try:
        import defrcn.data.builtin
    except ImportError:
        print("Warning: Could not import defrcn.data.builtin. Metadata might be incomplete.")
        
    analyze_dataset()