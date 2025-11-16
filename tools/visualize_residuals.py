import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random

# ------------------------
# Load residuals and labels
# ------------------------
residuals = np.load("data/residuals/residuals.npy")       # path to residuals
labels = np.load("data/residuals/residuals_labels.npy")   # path to labels
print("Residuals shape:", residuals.shape, "Labels shape:", labels.shape)

# ------------------------
# Optional: sample for speed (t-SNE is slow on large datasets)
# ------------------------
sample_size = 5000
if residuals.shape[0] > sample_size:
    idx = np.random.choice(residuals.shape[0], sample_size, replace=False)
    res_sample = residuals[idx]
    labels_sample = labels[idx]
else:
    res_sample = residuals
    labels_sample = labels

# ------------------------
# t-SNE embedding
# ------------------------
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
res_2d = tsne.fit_transform(res_sample)

# ------------------------
# Plot
# ------------------------
plt.figure(figsize=(8,8))
num_classes = len(np.unique(labels_sample))
# Generate a color map
cmap = plt.get_cmap("tab20", num_classes)

for cls in np.unique(labels_sample):
    mask = labels_sample == cls
    plt.scatter(res_2d[mask,0], res_2d[mask,1], s=10, alpha=0.6, label=f"Class {cls}", color=cmap(cls))

plt.title("t-SNE of Residuals Colored by Class")
plt.xlabel("t-SNE dim 1")
plt.ylabel("t-SNE dim 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.show()
