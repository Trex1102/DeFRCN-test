import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureHallucinationModule(nn.Module):
    def __init__(self, channels: int, hidden_ratio: int = 4):
        super().__init__()
        mid = max(8, channels // hidden_ratio)
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(mid, mid, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1),
        )
        # initialize last conv to zero so residual starts small
        nn.init.constant_(self.decoder[-1].weight, 0.0)
        nn.init.constant_(self.decoder[-1].bias, 0.0)

    def forward(self, x: torch.Tensor):
        # returns reconstructed feature (residual added)
        return x + self.decoder(self.encoder(x))