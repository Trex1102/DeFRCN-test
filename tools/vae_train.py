# vae_train.py
import torch, time, numpy as np
from torch.utils.data import DataLoader, TensorDataset
from vae_model import ResidualVAE, Adversary
import torch.optim as optim
import torch.nn.functional as F

# load residuals
deltas = np.load("data/residuals.npy")  # shape (M, D)
labels = np.load("data/residuals_labels.npy")
# optionally use PCA deltas if you applied PCA earlier
# deltas = np.load("data/residuals_pca.npy")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create dataset: we also need anchor features to compute class-consistency
# If using NN pairs we saved pairs; for prototype deltas, anchor is the original f
# For simplicity assume we have anchor features aligned: anchors.npy same order as deltas
anchors = np.load("data/residual_anchors.npy")  # (M, D) anchors (f_a)
dataset = TensorDataset(torch.from_numpy(deltas).float(), torch.from_numpy(anchors).float(), torch.from_numpy(labels).long())
loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4, drop_last=True)

# model
inp_dim = deltas.shape[1]
vae = ResidualVAE(inp_dim, z_dim=64).to(device)
adv = Adversary(64, n_classes=NUM_BASE_CLASSES).to(device)

opt_vae = optim.Adam(vae.parameters(), lr=1e-3, weight_decay=1e-6)
opt_adv = optim.Adam(adv.parameters(), lr=1e-4)

# Load classifier head from DeFRCN to compute CE on f+delta_hat
# Option: load entire ROI classifier head weights and use it frozen
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
cfg = ... # same cfg used before
det_model = build_model(cfg)
DetectionCheckpointer(det_model).load(cfg.MODEL.WEIGHTS)
det_model.eval().to(device)
# find function that maps RoI feature vector -> class logits in your codebase:
# It is often det_model.roi_heads.box_predictor.cls_score(fc_feat) or via box_predictor

def compute_logits_from_feat(feat_tensor):
    # feat_tensor: (B, D)
    # adapt to your model's predictor
    logits, _ = det_model.roi_heads.box_predictor(feat_tensor)
    return logits

# KL helpers
def kl_gaussian(mu, logvar):
    # return sum over dim, then mean over batch
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

# training hyperparams
lambda_cls = 1.0
lambda_inv = 0.1
beta = 0.0  # start KL anneal at 0
kl_anneal_steps = 10000
step = 0
n_epochs = 30

for epoch in range(n_epochs):
    for batch in loader:
        step += 1
        delta, anchor, y = [t.to(device) for t in batch]  # delta: (B,D)
        # ---- adversary step ----
        with torch.no_grad():
            mu, logvar = vae.encode(delta)
            z = vae.reparam(mu, logvar)
        adv_logits = adv(z.detach())
        adv_loss = F.cross_entropy(adv_logits, y)
        opt_adv.zero_grad()
        adv_loss.backward()
        opt_adv.step()

        # ---- VAE step ----
        delta_hat, mu, logvar, z = vae(delta)
        rec_loss = F.mse_loss(delta_hat, delta, reduction='mean')  # L2
        kl = kl_gaussian(mu, logvar)
        # class-consistency
        f_anchor = anchor  # assume normalized same as saved
        f_aug = f_anchor + delta_hat
        logits = compute_logits_from_feat(f_aug)   # (B, C)
        cls_loss = F.cross_entropy(logits, y)
        # invariance: encoder tries to fool adv (maximize adv loss => minimize -CE)
        adv_logits_for_enc = adv(z)
        inv_loss = -F.cross_entropy(adv_logits_for_enc, y)
        # KL annealing schedule
        beta = min(1.0, step / kl_anneal_steps)
        loss = rec_loss + beta * kl + lambda_cls * cls_loss + lambda_inv * inv_loss
        opt_vae.zero_grad()
        loss.backward()
        opt_vae.step()

        if step % 100 == 0:
            print(f"step {step} rec {rec_loss.item():.4f} kl {kl.item():.4f} cls {cls_loss.item():.4f} adv {adv_loss.item():.4f}")
            
    # optionally save checkpoints per epoch
    torch.save(vae.state_dict(), f"checkpoints/vae_epoch{epoch}.pth")
    torch.save(adv.state_dict(), f"checkpoints/adv_epoch{epoch}.pth")
