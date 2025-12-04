import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class CondResidualVAE(nn.Module):
    def __init__(self, resid_dim=2048, sem_dim=512, latent_dim=512, hidden_h=4096, leaky_slope=0.2):
        super().__init__()

        # Encoder
        self.enc_fcx = nn.Linear(resid_dim, hidden_h)
        self.enc_fcy = nn.Linear(sem_dim, hidden_h)
        self.enc_fc1 = nn.Linear(hidden_h*2, hidden_h)
        self.enc_fc2 = nn.Linear(hidden_h, hidden_h)
        self.enc_fc3 = nn.Linear(hidden_h, hidden_h)
        self.enc_mu = nn.Linear(hidden_h, latent_dim)
        self.enc_logvar = nn.Linear(hidden_h, latent_dim)
        self.enc_act = nn.LeakyReLU(leaky_slope, inplace=True)

        # Prior
        self.prior_fc1 = nn.Linear(sem_dim, hidden_h)
        self.prior_fc2 = nn.Linear(hidden_h, hidden_h)
        self.prior_mu = nn.Linear(hidden_h, latent_dim)
        self.prior_logvar = nn.Linear(hidden_h, latent_dim)
        self.prior_act = nn.LeakyReLU(leaky_slope, inplace=True)

        # Decoder
        self.dec_fcx = nn.Linear(latent_dim, hidden_h)
        self.dec_fcy = nn.Linear(sem_dim, hidden_h)
        self.dec_fc1 = nn.Linear(hidden_h*2, hidden_h)
        self.dec_out = nn.Linear(hidden_h, resid_dim)
        self.dec_hidden_act = nn.LeakyReLU(leaky_slope, inplace=True)
        self.dec_out_act = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, resid, sem):
        x = self.enc_act(self.enc_fcx(resid))
        y = self.enc_act(self.enc_fcy(sem))
        x = torch.cat([x,y], dim=1)
        x = self.enc_act(self.enc_fc1(x))
        x = self.enc_act(self.enc_fc2(x))
        x = self.enc_act(self.enc_fc3(x))
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
        x = self.dec_hidden_act(self.dec_fcx(z))   
        y = self.dec_hidden_act(self.dec_fcy(sem))
        x = torch.cat([x,y], dim=1)
        x = self.dec_hidden_act(self.dec_fc1(x))
        x = self.dec_out(x)
        x = self.dec_out_act(x)
        return x

    def forward(self, resid, sem):
        mu_q, logvar_q = self.encode(resid, sem)
        z = self.reparam(mu_q, logvar_q)
        mu_p, logvar_p = self.prior(sem)
        recon = self.decode(z, sem)
        return recon, mu_q, logvar_q, mu_p, logvar_p, z
