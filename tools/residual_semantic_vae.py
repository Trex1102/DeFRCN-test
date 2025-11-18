import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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


class CondResidualVAE(nn.Module):
    def __init__(self, resid_dim=2048, sem_dim=512, latent_dim=512, hidden_h=4096, leaky_slope=0.2):
        super().__init__()


        self.enc_fcx = nn.Linear(resid_dim, hidden_h)
        self.enc_fcy = nn.Linear(sem_dim, hidden_h)
        self.enc_fc1 = nn.Linear(hidden_h*2, hidden_h)
        self.enc_fc3 = nn.Linear(hidden_h, hidden_h)
        self.enc_mu = nn.Linear(hidden_h, latent_dim)
        self.enc_logvar = nn.Linear(hidden_h, latent_dim)
        self.enc_act = nn.LeakyReLU(leaky_slope, inplace=True)

        # Prior: maps semantic vector -> mu_p, logvar_p (uses same hidden size)
        self.prior_fc1 = nn.Linear(sem_dim, hidden_h)
        self.prior_fc2 = nn.Linear(hidden_h, hidden_h)
        self.prior_mu = nn.Linear(hidden_h, latent_dim)
        self.prior_logvar = nn.Linear(hidden_h, latent_dim)
        self.prior_act = nn.LeakyReLU(leaky_slope, inplace=True)

        # Decoder: input = z concat sem
        self.dec_fc1 = nn.Linear(latent_dim + sem_dim, hidden_h)
        self.dec_out = nn.Linear(hidden_h, resid_dim)
        self.dec_hidden_act = nn.LeakyReLU(leaky_slope, inplace=True)
        self.dec_out_act = nn.Identity()  # keep linear output (residual scale preserved)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, resid):
        x = self.enc_act(self.enc_fc1(resid))
        x = self.enc_act(self.enc_fc2(x))
        x = self.enc_act(self.enc_fc3(x))
        x = self.enc_act(self.enc_fc4(x))
        mu = self.enc_mu(x)
        logvar = self.enc_logvar(x)
        return mu, logvar

    def prior(self, sem):
        x = self.prior_act(self.prior_fc1(sem))
        x = self.prior_act(self.prior_fc2(x))
        mu = self.prior_mu(x)
        logvar = self.prior_logvar(x)
        return mu, logvar

    def reparam(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, sem):
        x = torch.cat([z, sem], dim=1)
        x = self.dec_hidden_act(self.dec_fc1(x))
        x = self.dec_out(x)
        x = self.dec_out_act(x)
        return x

    def forward(self, resid, sem):
        mu_q, logvar_q = self.encode(resid)
        z = self.reparam(mu_q, logvar_q)
        mu_p, logvar_p = self.prior(sem)
        recon = self.decode(z, sem)
        return recon, mu_q, logvar_q, mu_p, logvar_p, z


def kl_divergence_gaussians(mu_q, logvar_q, mu_p, logvar_p):
    var_q = logvar_q.exp()
    var_p = logvar_p.exp()
    kl_elem = 0.5 * ((var_q + (mu_q - mu_p).pow(2)) / var_p - 1 + (logvar_p - logvar_q))
    return kl_elem.sum(dim=1).mean()


# ---------------------------
# CLIP embedding helper (on-the-fly)
# ---------------------------
def build_clip_class_embeddings(class_names, device="cpu", clip_model_name="ViT-B/32"):
    """
    Build CLIP text embeddings for class_names (list of strings).
    Requires the 'clip' package: pip install git+https://github.com/openai/CLIP.git
    Returns np.array shape (num_classes, clip_dim).
    """
    try:
        import clip
    except Exception as e:
        raise RuntimeError("Please install CLIP: pip install git+https://github.com/openai/CLIP.git") from e

    model, _ = clip.load(clip_model_name, device=device)
    model.eval()
    # simple prompt template (can be changed)
    texts = [f"a photo of a {c}" for c in class_names]
    with torch.no_grad():
        tokens = clip.tokenize(texts).to(device)
        text_emb = model.encode_text(tokens)  # (num_classes, dim)
        text_emb = text_emb / text_emb.norm(dim=1, keepdim=True)
        return text_emb.cpu().numpy()


# ---------------------------
# Training function
# ---------------------------
def train(
    residuals_path,
    anchors_path,
    labels_path,
    class_names_file,
    class_embeds_path,
    out_path,
    resid_dim=2048,
    sem_dim=512,
    latent_dim=128,
    hidden_h=4096,
    batch_size=256,
    epochs=1000,
    lr=1e-4,
    beta_start=0.0,
    beta_end=1.0,
    kl_anneal_steps=100000,
    kl_free_bits = 1,
    patience=50,
    device="cuda",
    use_pca=False
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    ds = ResidualCondDataset(residuals_path, anchors_path, labels_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    # load or build class embeddings
    if class_embeds_path and os.path.exists(class_embeds_path):
        class_embeds = np.load(class_embeds_path).astype(np.float32)
        if class_embeds.shape[1] != sem_dim:
            raise ValueError(f"class_embeds has dim {class_embeds.shape[1]} but sem_dim={sem_dim}")
    else:
        if class_names_file is None:
            raise ValueError("Either provide --class_embeds or --class_names file")
        with open(class_names_file, "r") as f:
            class_names = [ln.strip() for ln in f.readlines() if ln.strip()]
        print(f"[INFO] Building CLIP embeddings for {len(class_names)} classes (this may take a moment)...")
        class_embeds = build_clip_class_embeddings(class_names, device=device, clip_model_name="ViT-B/32")
        # save for future runs
        if class_embeds_path:
            np.save(class_embeds_path, class_embeds)

    class_embeds = torch.from_numpy(class_embeds).float().to(device)  # (num_classes, sem_dim)

    model = CondResidualVAE(resid_dim=resid_dim, sem_dim=sem_dim, latent_dim=latent_dim, hidden_h=hidden_h).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20, verbose=True)

    best_loss = float("inf")
    epochs_no_improve = 0
    global_step = 0
    total_steps_est = epochs * (len(loader) if len(loader)>0 else 1)

    # training loop
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_rec = 0.0
        epoch_kl = 0.0
        for resid_np, anchor_np, labels_np in loader:
            global_step += 1
            # resid = torch.from_numpy(resid_np).float().to(device)   # normalized residuals
            # labels = torch.from_numpy(labels_np).long().to(device)


            # convert to tensor only if not already tensor
            if isinstance(resid_np, np.ndarray):
                resid = torch.from_numpy(resid_np).float()
            else:
                resid = resid_np.float()

            if isinstance(labels_np, np.ndarray):
                labels = torch.from_numpy(labels_np).long()
            else:
                labels = labels_np.long()

            resid = resid.to(device)
            labels = labels.to(device)


            # gather semantic embeddings for this batch
            sem = class_embeds[labels]   # (B, sem_dim)

            # forward
            recon, mu_q, logvar_q, mu_p, logvar_p, z = model(resid, sem)

            # losses
            rec_loss = F.mse_loss(recon, resid, reduction="mean")
            kl_loss = kl_divergence_gaussians(mu_q, logvar_q, mu_p, logvar_p)
            
            # Apply Free Bits constraint: only penalize KL if it exceeds the budget
            if kl_free_bits > 0.0:
                kl_penalty = torch.max(kl_loss, torch.tensor(kl_free_bits).to(kl_loss.device))
            else:
                kl_penalty = kl_loss # Standard KL when free_bits is 0

            # KL anneal schedule (linear anneal from beta_start -> beta_end over kl_anneal_steps)
            if kl_anneal_steps > 0:
                t = min(float(global_step) / float(kl_anneal_steps), 1.0)
                beta = beta_start + t * (beta_end - beta_start)
            else:
                beta = beta_end

            loss = rec_loss + beta * kl_penalty

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_rec += rec_loss.item()
            epoch_kl += kl_loss.item()

        avg_loss = epoch_loss / len(loader)
        avg_rec = epoch_rec / len(loader)
        avg_kl = epoch_kl / len(loader)

        print(f"[Epoch {epoch}/{epochs}] Loss: {avg_loss:.6f} | Recon: {avg_rec:.6f} | KL: {avg_kl:.6f} | beta: {beta:.4f}")

        # scheduler step
        scheduler.step(avg_loss)

        # checkpointing & early stopping
        if avg_loss + 1e-8 < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "class_embeds": class_embeds.detach().cpu().numpy(),
                "args": {
                    "resid_dim": resid_dim,
                    "sem_dim": sem_dim,
                    "latent_dim": latent_dim,
                    "hidden_h": hidden_h
                }
            }, out_path)
            print(f"[INFO] Saved best model to {out_path} (loss {best_loss:.6f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[INFO] Early stopping after {epoch} epochs (no improvement for {patience} epochs).")
                break

    print("[TRAINING DONE] best loss:", best_loss)


# ---------------------------
# CLI main
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--residuals", type=str,  default="data/residuals/residuals.npy")
    p.add_argument("--anchors", type=str, default="data/residuals/residuals_anchors.npy")
    p.add_argument("--labels", type=str, default="data/residuals/residuals_labels.npy")
    p.add_argument("--class_names", type=str, default="data/class_names.txt")
    p.add_argument("--class_embeds", type=str, default="data/class_embeds.npy")
    p.add_argument("--out", type=str, default="cvae_best.pth")
    p.add_argument("--resid_dim", type=int, default=2048)
    p.add_argument("--sem_dim", type=int, default=512)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--hidden_h", type=int, default=4096)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta_start", type=float, default=0.0)
    p.add_argument("--beta_end", type=float, default=1.0)
    p.add_argument("--kl_anneal_steps", type=int, default=100000)
    p.add_argument("--kl_free_bits", type=float, default=1)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(
        residuals_path=args.residuals,
        anchors_path=args.anchors,
        labels_path=args.labels,
        class_names_file=args.class_names,
        class_embeds_path=args.class_embeds,
        out_path=args.out,
        resid_dim=args.resid_dim,
        sem_dim=args.sem_dim,
        latent_dim=args.latent_dim,
        hidden_h=args.hidden_h,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        kl_anneal_steps=args.kl_anneal_steps,
        kl_free_bits=args.kl_free_bits,
        patience=args.patience,
        device=args.device
    )
