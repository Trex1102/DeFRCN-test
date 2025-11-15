# build_residuals.py
import os, glob
import numpy as np
from sklearn.decomposition import PCA
import faiss  # pip install faiss-cpu or faiss-gpu

# 1) Load all features into per-class lists
base_feats_dir = "data/roifeats_base"
class_features = {}  # class_id -> list of features (N, D) and metadata
for npzfile in glob.glob(os.path.join(base_feats_dir, "*.npz")):
    data = np.load(npzfile)
    feats = data["feats"]  # (n, D)
    classes = data["classes"]
    boxes = data["boxes"]
    # iterate per row
    for f,c,b in zip(feats, classes, boxes):
        class_features.setdefault(int(c), []).append((f, npzfile, b))

# 2) Convert to arrays and optionally sample per class cap
max_per_class = 20000
for c, lst in list(class_features.items()):
    if len(lst) > max_per_class:
        # random subsample
        idxs = np.random.choice(len(lst), max_per_class, replace=False)
        class_features[c] = [lst[i] for i in idxs]

# 3) Build nearest neighbors using FAISS (per-class)
residuals = []
for c, lst in class_features.items():
    feats = np.stack([x[0] for x in lst]).astype('float32')  # (Nc, D)
    Nc, D = feats.shape
    # optional normalization
    # feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    index = faiss.IndexFlatL2(D)
    index.add(feats)
    # query each to get nearest neighbor (excluding itself)
    k = 2
    Ddist, I = index.search(feats, k)  # returns self + nearest
    for i in range(Nc):
        nn_idx = I[i,1]  # nearest neighbor not itself
        f_i = feats[i]
        f_j = feats[nn_idx]
        delta = (f_j - f_i).astype('float32')
        residuals.append((delta, c))  # store delta + class label (for optional conditioning)

# 4) Alternatively compute prototype residuals
prototype_res = []
for c, lst in class_features.items():
    feats = np.stack([x[0] for x in lst]).astype('float32')
    proto = feats.mean(axis=0)
    for f in feats:
        delta = (f - proto).astype('float32')
        prototype_res.append((delta, c))

# 5) Merge / balance / filter
all_res = residuals + prototype_res
# filter by norm outliers:
deltas = np.stack([r[0] for r in all_res])
norms = np.linalg.norm(deltas, axis=1)
mask = (norms < np.percentile(norms, 99))  # drop top 1%
deltas = deltas[mask]
labels = np.array([r[1] for r in all_res])[mask]

# optional PCA
pca_dim = 64
pca = PCA(n_components=pca_dim)
deltas_pca = pca.fit_transform(deltas)  # shape (M, pca_dim)

# save
np.save("data/residuals.npy", deltas.astype('float32'))
np.save("data/residuals_labels.npy", labels.astype('int32'))
# If using PCA:
np.save("data/residuals_pca.npy", deltas_pca.astype('float32'))
