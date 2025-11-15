# vae_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, nl=nn.ReLU):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nl())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class ResidualVAE(nn.Module):
    def __init__(self, inp_dim, z_dim=64, hidden=[512,256]):
        super().__init__()
        self.inp_dim = inp_dim
        self.z_dim = z_dim
        # encoder -> μ, logvar
        self.enc_net = MLP(inp_dim, hidden, hidden[-1])
        self.fc_mu = nn.Linear(hidden[-1], z_dim)
        self.fc_logvar = nn.Linear(hidden[-1], z_dim)
        # decoder -> reconstruct Δ
        self.dec_net = MLP(z_dim, hidden[::-1], hidden[0])
        self.dec_out = nn.Linear(hidden[0], inp_dim)

    def encode(self, delta):
        h = self.enc_net(delta)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparam(self, mu, logvar):
        std = (0.5*logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = self.dec_net(z)
        out = self.dec_out(h)
        return out

    def forward(self, delta):
        mu, logvar = self.encode(delta)
        z = self.reparam(mu, logvar)
        delta_hat = self.decode(z)
        return delta_hat, mu, logvar, z

# Adversary for class prediction from z
class Adversary(nn.Module):
    def __init__(self, z_dim, n_classes, hidden=[128,128]):
        super().__init__()
        self.net = MLP(z_dim, hidden, n_classes)
    def forward(self, z):
        return self.net(z)
