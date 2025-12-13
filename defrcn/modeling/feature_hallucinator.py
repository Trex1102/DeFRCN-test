import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class FeatureHallucinationModule(nn.Module):
    """
    Residual conv auto-encoder for RoI feature hallucination.

    Signature:
        FeatureHallucinationModule(channels: int,
                                   hidden_channels: Optional[int] = None,
                                   hidden_ratio: int = 4)

    - If `hidden_channels` is provided it is used directly.
    - Otherwise `hidden_channels = max(8, channels // hidden_ratio)`.
    """
    def __init__(self, channels: int, hidden_channels: Optional[int] = None, hidden_ratio: int = 4):
        super().__init__()
        if hidden_channels is None:
            mid = max(8, channels // hidden_ratio)
        else:
            mid = int(hidden_channels)

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )

        # bottleneck/resblocks (simple two conv residual block)
        self.resblocks = nn.Sequential(
            nn.Conv2d(mid, mid, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(mid, mid, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1),
        )

        # initialize last conv to zero so residuals start near zero
        nn.init.constant_(self.decoder[-1].weight, 0.0)
        nn.init.constant_(self.decoder[-1].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (R, C, H, W)
        returns: reconstructed feature (x + residual) same shape as x
        """
        h = self.encoder(x)
        h = self.resblocks(h)
        residual = self.decoder(h)
        return x + residual
