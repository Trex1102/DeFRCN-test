import math
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# 1) VAE
# ----------------------------
class VAE(nn.Module):
    def __init__(self, feat_dim=2048, latent_dim=512, hid=4096):
        """
        Encoder: 3 FC layers -> mu, logvar (latent_dim)
        Decoder: 2 FC layers -> reconstruct feat_dim
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim

        # encoder
        self.enc_fc1 = nn.Linear(feat_dim, hid)
        self.enc_fc2 = nn.Linear(hid, hid)
        self.enc_fc3_mu = nn.Linear(hid, latent_dim)
        self.enc_fc3_logvar = nn.Linear(hid, latent_dim)

        # decoder
        self.dec_fc1 = nn.Linear(latent_dim, hid)
        self.dec_fc2 = nn.Linear(hid, feat_dim)

        self.act_lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.act_relu = nn.ReLU(inplace=True)

    def encode(self, x):
        """
        x: [B, feat_dim]
        returns: mu [B, z], logvar [B, z]
        """
        h = self.act_lrelu(self.enc_fc1(x))
        h = self.act_lrelu(self.enc_fc2(h))
        mu = self.enc_fc3_mu(h)
        logvar = self.enc_fc3_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        z: [B, latent_dim]
        returns: recon_feat [B, feat_dim]
        """
        h = self.act_relu(self.dec_fc1(z))
        recon = self.dec_fc2(h)
        return recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z