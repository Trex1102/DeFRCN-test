import numpy as np
import matplotlib.pyplot as plt
import os

# Define Pascal VOC Class Names (in standard order)
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

def analyze_dataset(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    print(f"Loading {file_path}...")
    data = np.load(file_path)
    
    # Extract relevant arrays
    classes = data['classes']
    feats = data['feats']
    
    print(f"Total Objects Loaded: {len(classes)}")
    print(f"Feature Dimension: {feats.shape[1]}")
    print("-" * 40)

    # Count occurrences of each class ID
    unique_ids, counts = np.unique(classes, return_counts=True)
    
    # Prepare data for plotting
    class_names = []
    class_counts = []
    
    print(f"{'Class ID':<10} {'Name':<15} {'Count':<10}")
    print("-" * 40)

    for cls_id, count in zip(unique_ids, counts):
        # Handle potential background class or offsets if they exist
        if 0 <= cls_id < len(VOC_CLASSES):
            name = VOC_CLASSES[cls_id]
        else:
            name = f"Unknown ({cls_id})"
            
        class_names.append(name)
        class_counts.append(count)
        print(f"{cls_id:<10} {name:<15} {count:<10}")

    # --- Plotting ---
    plt.figure(figsize=(14, 6))
    bars = plt.bar(class_names, class_counts, color='skyblue', edgecolor='black')
    
    # Add counts on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval), 
                 ha='center', va='bottom', fontsize=9)

    plt.title(f'Class Distribution in {os.path.basename(file_path)}')
    plt.xlabel('Class Name')
    plt.ylabel('Number of Ground Truth Objects')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save plot instead of showing if running on a server without display
    plot_filename = "class_distribution.png"
    plt.savefig(plot_filename)
    print(f"\nPlot saved to {plot_filename}")
    plt.show()

if __name__ == "__main__":
    # Update this path to match the file you just created
    FILE_TO_ANALYZE = "data/roifeats_base/voc_2007_trainval_base1_combined.npz"
    
    analyze_dataset(FILE_TO_ANALYZE)