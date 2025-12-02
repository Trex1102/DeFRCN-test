
import torch
import logging
import numpy as np
import torch.nn.functional as F
from torch import nn
from detectron2.structures import ImageList
from defrcn.modeling.meta_arch.build import META_ARCH_REGISTRY
from defrcn.modeling.meta_arch.gdl import AffineLayer

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

    def decode(self, z, sem):
        x = self.dec_hidden_act(self.dec_fcx(z))   
        y = self.dec_hidden_act(self.dec_fcy(sem))
        x = torch.cat([x,y], dim=1)
        x = self.dec_hidden_act(self.dec_fc1(x))
        x = self.dec_out(x)
        x = self.dec_out_act(x)
        return x