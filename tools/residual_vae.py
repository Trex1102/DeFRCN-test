import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
#  Dataset: loads residuals saved as .npz
# ============================================================

class ResidualDataset(Dataset):
    def __init__(self, residuals_path):
        self.residuals = np.load(residuals_path).astype(np.float32)

    def __len__(self):
        return len(self.residuals)

    def __getitem__(self, idx):
        return self.residuals[idx]

# ============================================================
#  VAE Architecture
#  — NO normalization layers
#  — Fully-connected MLP
#  — Residual dim = 2048
#  — Latent dim = 256 (can be adjusted)
# ============================================================

class ResidualVAE(nn.Module):
    def __init__(self, dim=2048, latent_dim=256):
        super().__init__()
        self.dim = dim
        self.latent_dim = latent_dim

        # -----------------------
        # Encoder
        # -----------------------
        self.encoder = nn.Sequential(
            nn.Linear(dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(512, latent_dim)
        self.logvar_layer = nn.Linear(512, latent_dim)

        # -----------------------
        # Decoder
        # -----------------------
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# ============================================================
#  VAE Loss (ELBO)
# ============================================================

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="mean")

    # KL term
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl, recon_loss, kl

# ============================================================
#  Train Function
# ============================================================

def train_vae(
    residuals_path="data/residuals/residuals.npy",,
    output_path="vae_residuals.pth",
    batch_size=128,
    latent_dim=256,
    epochs=40,
    lr=1e-4,
):

    dataset = ResidualDataset(residuals_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = ResidualVAE(latent_dim=latent_dim).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0

        for x in loader:
            x = x.cuda()
            recon, mu, logvar = model(x)

            loss, rl, kl = vae_loss(recon, x, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += rl.item()
            total_kl += kl.item()

        print(f"[Epoch {epoch+1}/{epochs}] "
              f"Loss: {total_loss/len(loader):.4f} | "
              f"Recon: {total_recon/len(loader):.4f} | "
              f"KL: {total_kl/len(loader):.4f}")

        # Save model checkpoint
        torch.save({
            "model": model.state_dict(),
            "latent_dim": latent_dim
        }, output_path)

    print(f"VAE saved to {output_path}")
    return model
