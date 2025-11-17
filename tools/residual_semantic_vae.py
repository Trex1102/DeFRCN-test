#!/usr/bin/env python3
"""
train_conditional_residual_vae_full.py

Conditional Residual-VAE matching requested architecture:
- Encoder: 3 x FC(4096) + LeakyReLU
- Decoder: 2 x FC(4096) + ReLU
- Latent dim: 512
- Semantic vector dim: 512 (expects class_embeds.npy)

Inputs:
- residuals.npy            (N, D)  float32  -- normalized residuals
- residuals_anchors.npy    (N, D)  float32  -- anchor features (same normalization)
- residuals_labels.npy     (N,)    int32    -- class labels (0..C-1)
- class_embeds.npy         (C, 512) float32 -- CLIP/class embeddings, aligned with labels

Optional:
- Provide --use_detector and paths to DeFRCN config + weights to compute class-consistency loss
"""

import os
import argparse
import math
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --------- Dataset ----------
class ResidualCondDataset(Dataset):
    def __init__(self, residuals_path, anchors_path, labels_path):
        self.residuals = np.load(residuals_path).astype(np.float32)
        self.anchors = np.load(anchors_path).astype(np.float32)
        self.labels = np.load(labels_path).astype(np.int64)
        assert len(self.residuals) == len(self.anchors) == len(self.labels)
    def __len__(self):
        return len(self.residuals)
    def __getitem__(self, idx):
        return self.residuals[idx], self.anchors[idx], self.labels[idx]

# --------- Model (conditional VAE) ----------
class CondResidualVAE(nn.Module):
    def __init__(self, resid_dim=2048, sem_dim=512, latent_dim=512, hidden_h=4096):
        super().__init__()
        self.resid_dim = resid_dim
        self.sem_dim = sem_dim
        self.latent_dim = latent_dim

        # Encoder: input = [resid (D), sem (S)] -> 3 FC layers of 4096 -> mu / logvar
        enc_in = resid_dim + sem_dim
        h = hidden_h
        self.enc_fc1 = nn.Linear(enc_in, h)
        self.enc_fc2 = nn.Linear(h, h)
        self.enc_fc3 = nn.Linear(h, h)
        self.enc_mu   = nn.Linear(h, latent_dim)
        self.enc_logvar = nn.Linear(h, latent_dim)
        self.enc_act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Prior network: maps sem vector -> prior mu_p, logvar_p (both latent_dim)
        # small MLP
        self.prior_fc1 = nn.Linear(sem_dim, 1024)
        self.prior_fc2 = nn.Linear(1024, 1024)
        self.prior_mu = nn.Linear(1024, latent_dim)
        self.prior_logvar = nn.Linear(1024, latent_dim)
        self.prior_act = nn.ReLU(inplace=True)

        # Decoder: input = [z (latent_dim), sem (S)] -> 2 FC layers of 4096 -> out resid_dim
        dec_in = latent_dim + sem_dim
        self.dec_fc1 = nn.Linear(dec_in, h)
        self.dec_fc2 = nn.Linear(h, h)
        self.dec_out = nn.Linear(h, resid_dim)
        self.dec_act = nn.ReLU(inplace=True)

        # initialize weights (optional but helpful)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, resid, sem):
        # resid: (B, D), sem: (B, S)
        x = torch.cat([resid, sem], dim=1)
        x = self.enc_act(self.enc_fc1(x))
        x = self.enc_act(self.enc_fc2(x))
        x = self.enc_act(self.enc_fc3(x))
        mu = self.enc_mu(x)
        logvar = self.enc_logvar(x)
        return mu, logvar

    def prior(self, sem):
        # sem: (B, S)
        x = self.prior_act(self.prior_fc1(sem)) if hasattr(self, "prior_act") else self.prior_fc1(sem)
        x = self.prior_act(self.prior_fc2(x))
        mu_p = self.prior_mu(x)
        logvar_p = self.prior_logvar(x)
        return mu_p, logvar_p

    def reparam(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, sem):
        x = torch.cat([z, sem], dim=1)
        x = self.dec_act(self.dec_fc1(x))
        x = self.dec_act(self.dec_fc2(x))
        out = self.dec_out(x)
        return out

    def forward(self, resid, sem):
        # resid, sem: tensors
        mu_q, logvar_q = self.encode(resid, sem)
        z = self.reparam(mu_q, logvar_q)
        mu_p, logvar_p = self.prior(sem)
        recon = self.decode(z, sem)
        return recon, mu_q, logvar_q, mu_p, logvar_p, z

# --------- KL between two Gaussians (analytical) ----------
def kl_divergence_gaussians(mu_q, logvar_q, mu_p, logvar_p):
    # computes KL(q||p) per batch, returns mean over batch
    # q ~ N(mu_q, sigma_q^2), p ~ N(mu_p, sigma_p^2)
    var_q = logvar_q.exp()
    var_p = logvar_p.exp()
    # elementwise KL
    kl_elem = 0.5 * ( (var_q + (mu_q - mu_p).pow(2)) / var_p - 1 + (logvar_p - logvar_q) )
    return kl_elem.sum(dim=1).mean()

# --------- Helper to optionally compute classifier logits using DeFRCN
def build_detector_predictor(cfg_path, weights_path, device):
    """
    Tries to import Detectron2 and build model to compute logits from feature vectors.
    Returns a function classifier_from_feat(feat_tensor) -> logits (B, num_classes).
    If detectron2 not available or load fails, returns None.
    """
    try:
        import detectron2
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        cfg = get_cfg()
        cfg.merge_from_file(cfg_path)
        cfg.MODEL.WEIGHTS = weights_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
        predictor = DefaultPredictor(cfg)
        model = predictor.model
        model.eval()
        model.to(device)
        # We need to find how to map a RoI feature (vector) to logits using this model
        # Many implementations use model.roi_heads.box_predictor.cls_score or box_predictor
        def classifier_from_feat(feat_tensor):
            # feat_tensor: (B, D) torch tensor on device
            with torch.no_grad():
                # depending on implementation, box_predictor might expect flattened features
                try:
                    logits, _ = model.roi_heads.box_predictor(feat_tensor)
                    return logits
                except Exception:
                    # Try to call linear layer directly if name differs
                    # This may need adapting to your DeFRCN implementation
                    raise RuntimeError("Unable to call box_predictor on features. Adapt classifier_from_feat.")
        return classifier_from_feat, model
    except Exception as e:
        print("Warning: detectron2/DeFRCN predictor could not be loaded:", e)
        return None, None

# --------- Training function ----------
def train(
    residuals_path,
    anchors_path,
    labels_path,
    class_embeds_path,
    out_path,
    resid_dim=2048,
    sem_dim=512,
    latent_dim=512,
    hidden=4096,
    batch_size=256,
    lr=1e-4,
    beta=0.05,
    kl_anneal_steps=10000,
    epochs=1000,
    patience=100,
    use_detector=False,
    detector_cfg=None,
    detector_weights=None,
    device_name="cuda",
    clip_grad=5.0,
    scheduler_patience=20
):
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    # Load data
    residuals = np.load(residuals_path).astype(np.float32)
    anchors = np.load(anchors_path).astype(np.float32)
    labels = np.load(labels_path).astype(np.int64)
    class_embeds = np.load(class_embeds_path).astype(np.float32)  # (C, sem_dim)
    num_classes = int(class_embeds.shape[0])
    assert class_embeds.shape[1] == sem_dim

    ds = ResidualCondDataset(residuals_path, anchors_path, labels_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    # model
    model = CondResidualVAE(resid_dim, sem_dim, latent_dim, hidden_h=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=scheduler_patience, verbose=True)

    # optional classifier
    classifier_fn = None
    det_model = None
    if use_detector:
        classifier_fn, det_model = build_detector_predictor(detector_cfg, detector_weights, device)
        if classifier_fn is None:
            print("Detector classifier not available; continuing without class-consistency.")

    # convert class_embeds to tensor on device
    class_embeds_t = torch.from_numpy(class_embeds).float().to(device)

    best_loss = float('inf')
    epochs_no_improve = 0
    global_step = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_recon = 0.0
        running_kl = 0.0
        running_cls = 0.0
        batches = 0

        for resid_np, anchor_np, lbl_np in loader:
            global_step += 1
            resid = resid_np.to(device)          # normalized residual
            anchor = anchor_np.to(device)
            lbl = lbl_np.to(device)
            sem = class_embeds_t[lbl]            # (B, sem_dim)

            # forward
            recon, mu_q, logvar_q, mu_p, logvar_p, z = model(resid, sem)
            # reconstruction
            rec_loss = F.mse_loss(recon, resid, reduction="mean")
            # KL between q and conditional prior p(z|a)
            kl = kl_divergence_gaussians(mu_q, logvar_q, mu_p, logvar_p)

            # KL annealing (linear ramp up)
            beta_step = beta * min(1.0, global_step / max(1, kl_anneal_steps))

            loss = rec_loss + beta_step * kl

            # optional class-consistency: require classifier(anchor + recon) to predict label
            if classifier_fn is not None:
                try:
                    f_aug = anchor + recon  # both normalized in same space
                    logits = classifier_fn(f_aug)  # (B, C)
                    cls_loss = F.cross_entropy(logits, lbl)
                    lambda_cls = 1.0
                    loss = loss + lambda_cls * cls_loss
                    running_cls += cls_loss.item()
                except Exception as e:
                    # if mapping to detector fails, skip cls loss
                    # don't crash training due to API mismatch
                    pass

            optimizer.zero_grad()
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

            running_loss += loss.item()
            running_recon += rec_loss.item()
            running_kl += kl.item()
            batches += 1

        avg_loss = running_loss / max(1, batches)
        avg_recon = running_recon / max(1, batches)
        avg_kl = running_kl / max(1, batches)
        avg_cls = running_cls / max(1, batches) if running_cls > 0 else 0.0

        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.6f} Recon: {avg_recon:.6f} KL: {avg_kl:.6f} CLS: {avg_cls:.6f}")

        # scheduler step
        scheduler.step(avg_loss)

        # early stopping / checkpoint
        if avg_loss + 1e-8 < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            # save model + metadata
            save_dict = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss": best_loss,
                "resid_dim": resid_dim,
                "sem_dim": sem_dim,
                "latent_dim": latent_dim,
                "hidden": hidden
            }
            torch.save(save_dict, out_path)
            print(f"Saved best model to {out_path} (loss {best_loss:.6f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs (no improvement for {patience} epochs).")
                break

    print("Training finished. Best loss:", best_loss)
    return model

# --------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--residuals", default="data/residuals/residuals.npy", help="path to residuals.npy")
    p.add_argument("--anchors", default="data/residuals/residuals_anchors.npy", help="path to anchors")
    p.add_argument("--labels", default="data/residuals/residuals_labels.npy", help="path to labels")
    p.add_argument("--class_embeds", default="data/class_embeds.npy", help="path to class embeddings (C x 512)")
    p.add_argument("--out", default="cvae_best.pth", help="where to save checkpoint")
    p.add_argument("--resid_dim", type=int, default=2048)
    p.add_argument("--sem_dim", type=int, default=512)
    p.add_argument("--latent_dim", type=int, default=512)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--kl_anneal_steps", type=int, default=10000)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--use_detector", action="true", help="enable class-consistency via DeFRCN classifier")
    p.add_argument("--detector_cfg", default="", help="detectron2/defrcn cfg file (if using detector)")
    p.add_argument("--detector_weights", default="", help="weights (if using detector)")
    p.add_argument("--device", default="cuda")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.residuals):
        raise FileNotFoundError("residuals not found: " + args.residuals)
    if not os.path.exists(args.class_embeds):
        raise FileNotFoundError("class_embeds not found: " + args.class_embeds)
    model = train(
        residuals_path=args.residuals,
        anchors_path=args.anchors,
        labels_path=args.labels,
        class_embeds_path=args.class_embeds,
        out_path=args.out,
        resid_dim=args.resid_dim,
        sem_dim=args.sem_dim,
        latent_dim=args.latent_dim,
        hidden=args.hidden,
        batch_size=args.batch_size,
        lr=args.lr,
        beta=args.beta,
        kl_anneal_steps=args.kl_anneal_steps,
        epochs=args.epochs,
        patience=args.patience,
        use_detector=args.use_detector,
        detector_cfg=args.detector_cfg,
        detector_weights=args.detector_weights,
        device_name=args.device
    )
