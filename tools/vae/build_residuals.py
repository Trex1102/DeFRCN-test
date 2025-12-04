import os
import glob
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import math
import random
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# Try to use FAISS if available for speed; otherwise fall back to sklearn
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

def load_npz_dir(in_dir, verbose=True):
    """
    Load all per-image .npz files in `in_dir` and aggregate features by class.
    Expects files contain arrays: feats (N,D), boxes (N,4), classes (N,)
    """
    pattern = os.path.join(in_dir, "*.npz")
    files = sorted(glob.glob(pattern))
    if verbose:
        print(f"Found {len(files)} .npz files in {in_dir}")

    class_features = defaultdict(list)  
    total_rois = 0
    for fp in tqdm(files, desc="Loading npz files"):
        try:
            data = np.load(fp, allow_pickle=True)
        except Exception as e:
            print(f"Warning: failed to load {fp}: {e}")
            continue
        feats = data.get("feats", None)
        classes = data.get("classes", None)
        boxes = data.get("boxes", None)
        if feats is None or classes is None:
            continue
        feats = np.asarray(feats, dtype=np.float32)
        classes = np.asarray(classes, dtype=np.int32)
        n = feats.shape[0]
        total_rois += n
        for i in range(n):
            cls = int(classes[i])
            class_features[cls].append({
                "feat": feats[i],
                "file": os.path.basename(fp),
                "box": None if boxes is None else np.asarray(boxes[i], dtype=np.float32)
            })
    if verbose:
        print(f"Loaded {total_rois} ROIs across {len(class_features)} classes")
    return class_features

def compute_mean_std_all(class_features):
    """Compute global per-dimension mean/std over all features."""
    # two-pass to avoid huge memory: compute mean then var
    dims = None
    total = 0
    mean = None
    for cls, lst in class_features.items():
        for entry in lst:
            f = entry["feat"]
            if mean is None:
                dims = f.shape[0]
                mean = np.zeros(dims, dtype=np.float64)
            mean += f
            total += 1
    mean /= max(1, total)
    # compute std
    m2 = np.zeros_like(mean)
    for cls, lst in class_features.items():
        for entry in lst:
            f = entry["feat"]
            diff = f - mean
            m2 += diff * diff
    var = m2 / max(1, total)
    std = np.sqrt(var)
    # avoid zero std
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)

def normalize_class_features(class_features, mean, std):
    """Inplace normalize features in class_features by mean/std"""
    for cls, lst in class_features.items():
        for entry in lst:
            entry["feat"] = (entry["feat"] - mean) / std

def sample_pairs_nn(class_feats_arr, k=1, use_faiss=_HAS_FAISS):
    """
    Given array feats (N, D) return list of index pairs (i, nn) where nn is nearest neighbor
    Excludes self; returns at most one pair per origin i (you can run with k>1 to get more)
    """
    N, D = class_feats_arr.shape
    if N < 2:
        return []

    if use_faiss:
        # faiss IndexFlatL2 expects float32
        index = faiss.IndexFlatL2(D)
        index.add(class_feats_arr.astype(np.float32))
        kq = min(N, k + 1)
        Ddist, I = index.search(class_feats_arr.astype(np.float32), kq)
        pairs = []
        for i in range(N):
            # find first neighbor that is not self
            for j in range(1, kq):
                nn = I[i, j]
                if nn != i:
                    pairs.append((i, int(nn)))
                    break
        return pairs
    else:
        # sklearn nearest neighbors (L2)
        nbrs = NearestNeighbors(n_neighbors=min(N, k + 1), algorithm="auto", metric="euclidean").fit(class_feats_arr)
        distances, indices = nbrs.kneighbors(class_feats_arr)
        pairs = []
        for i in range(N):
            for j in range(1, indices.shape[1]):
                nn = int(indices[i, j])
                if nn != i:
                    pairs.append((i, nn))
                    break
        return pairs

def sample_pairs_random(class_feats_arr, nsamples):
    """
    Generate nsamples random same-class pairs (i,j) without replacement if possible.
    """
    N = class_feats_arr.shape[0]
    if N < 2:
        return []
    pairs = []
    # If nsamples is large relative to N*(N-1)/2, just enumerate all pairs and sample
    max_pairs = N * (N - 1) // 2
    if nsamples >= max_pairs:
        for i in range(N):
            for j in range(i + 1, N):
                pairs.append((i, j))
        random.shuffle(pairs)
        return pairs
    # Otherwise sample with replacement but try to avoid trivial repeats
    for _ in range(nsamples):
        i = random.randrange(N)
        j = random.randrange(N - 1)
        if j >= i:
            j += 1
        pairs.append((i, j))
    return pairs

def build_residuals_from_pairs(class_features, out_dir, strategy="mix", nn_k=1,
                               random_per_class=100, max_per_class=20000,
                               min_sim=0.0, pca_dim=0, verbose=True):
    """
    Build residuals and anchors based on strategy.
    Returns residuals (M,D), anchors (M,D), labels (M,)
    """
    residuals = []
    anchors = []
    labels = []

    for cls, lst in tqdm(class_features.items(), desc="Processing classes"):
        if len(lst) == 0:
            continue
        # optionally cap per-class features for speed
        if len(lst) > max_per_class:
            lst = random.sample(lst, max_per_class)

        feats = np.stack([e["feat"] for e in lst], axis=0).astype(np.float32)  # (Nc, D)
        Nc, D = feats.shape

        if strategy in ("nn", "mix"):
            # NN pairs
            pairs = sample_pairs_nn(feats, k=nn_k)
            for (i, j) in pairs:
                f_i = feats[i]; f_j = feats[j]
                # optional similarity filter (cosine)
                if min_sim > 0.0:
                    denom = (np.linalg.norm(f_i) * np.linalg.norm(f_j))
                    sim = (f_i.dot(f_j) / denom) if denom > 0 else 0.0
                    if sim < min_sim:
                        continue
                residuals.append((f_j - f_i).astype(np.float32))
                anchors.append(f_i.astype(np.float32))
                labels.append(cls)

        if strategy in ("random", "mix"):
            # Random pairs
            pairs = sample_pairs_random(feats, nsamples=random_per_class)
            for (i, j) in pairs:
                f_i = feats[i]; f_j = feats[j]
                if min_sim > 0.0:
                    denom = (np.linalg.norm(f_i) * np.linalg.norm(f_j))
                    sim = (f_i.dot(f_j) / denom) if denom > 0 else 0.0
                    if sim < min_sim:
                        continue
                residuals.append((f_j - f_i).astype(np.float32))
                anchors.append(f_i.astype(np.float32))
                labels.append(cls)

        if strategy in ("proto", "mix"):
            # Prototype residuals (f - prototype)
            proto = feats.mean(axis=0)
            for i in range(Nc):
                f_i = feats[i]
                # optionally skip near-zero deltas:
                delta = (f_i - proto).astype(np.float32)
                if np.linalg.norm(delta) < 1e-6:
                    continue
                residuals.append(delta)
                anchors.append(proto.astype(np.float32))   # anchor = prototype (so adding delta gives f_i)
                labels.append(cls)

    if len(residuals) == 0:
        raise RuntimeError("No residuals produced - check filters / input features")

    residuals = np.stack(residuals, axis=0)
    anchors = np.stack(anchors, axis=0)
    labels = np.array(labels, dtype=np.int32)

    if verbose:
        print(f"Built {residuals.shape[0]} residuals (dim {residuals.shape[1]}) across {len(np.unique(labels))} classes")

    # Optional PCA
    pca = None
    if pca_dim and pca_dim > 0 and pca_dim < residuals.shape[1]:
        print(f"Fitting PCA -> reduce {residuals.shape[1]} -> {pca_dim}")
        pca = PCA(n_components=pca_dim, svd_solver="auto", whiten=False)
        residuals_pca = pca.fit_transform(residuals)
        residuals_out = residuals_pca.astype(np.float32)
    else:
        residuals_out = residuals

    # Basic outlier removal: clip top percentile of norms
    norms = np.linalg.norm(residuals_out, axis=1)
    thr = np.percentile(norms, 99.0)
    keep_mask = norms <= thr
    if verbose:
        n_before = residuals_out.shape[0]
    residuals_out = residuals_out[keep_mask]
    anchors = anchors[keep_mask]
    labels = labels[keep_mask]
    if verbose:
        print(f"Removed {n_before - residuals_out.shape[0]} extreme residuals (norm > 99th pct)")

    # Save
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "residuals.npy"), residuals_out.astype(np.float32))
    np.save(os.path.join(out_dir, "residuals_anchors.npy"), anchors.astype(np.float32))
    np.save(os.path.join(out_dir, "residuals_labels.npy"), labels.astype(np.int32))
    if pca is not None:
        np.save(os.path.join(out_dir, "pca_components.npy"), pca.components_.astype(np.float32))
        np.save(os.path.join(out_dir, "pca_mean.npy"), pca.mean_.astype(np.float32))
    return residuals_out, anchors, labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="data/roifeats_base", help="Directory with per-image .npz files")
    parser.add_argument("--out_dir", type=str, default="data/residuals", help="Where to save residuals and stats")
    parser.add_argument("--strategy", type=str, default="mix", choices=["nn","random","proto","mix"],
                        help="Pairing strategy")
    parser.add_argument("--nn-k", type=int, default=1, help="k for nearest neighbors (per element)")
    parser.add_argument("--random-per-class", type=int, default=50, help="random pairs per class when random used")
    parser.add_argument("--max-per-class", type=int, default=20000, help="cap on features per class when building index")
    parser.add_argument("--min-sim", type=float, default=0.3, help="min cosine similarity filter for pairs (0..1)")
    parser.add_argument("--pca", type=int, default=0, help="PCA dim to reduce residuals to (0 = no PCA)")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Loading features from:", args.in_dir)
    class_features = load_npz_dir(args.in_dir, verbose=True)

    # Optionally cap per class here - implemented in builder
    # Compute mean/std for normalization
    print("Computing global feature mean/std...")
    mean, std = compute_mean_std_all(class_features)
    print("mean/std shapes:", mean.shape, std.shape)
    # Save mean/std
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "feat_mean.npy"), mean)
    np.save(os.path.join(args.out_dir, "feat_std.npy"), std)

    # Normalize features in-place
    print("Normalizing features (in-memory)...")
    normalize_class_features(class_features, mean, std)

    # Build residuals
    print("Building residuals with strategy:", args.strategy)
    residuals_out, anchors_out, labels_out = build_residuals_from_pairs(
        class_features,
        out_dir=args.out_dir,
        strategy=args.strategy,
        nn_k=args.nn_k,
        random_per_class=args.random_per_class,
        max_per_class=args.max_per_class,
        min_sim=args.min_sim,
        pca_dim=args.pca,
        verbose=True
    )

    print("Saved residuals to:", args.out_dir)
    print("Residuals shape:", residuals_out.shape)
    print("Anchors shape:", anchors_out.shape)
    print("Labels shape:", labels_out.shape)
    print("Done.")

if __name__ == "__main__":
    main()
