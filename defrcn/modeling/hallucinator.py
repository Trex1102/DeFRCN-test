import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureHallucinator(nn.Module):
    def __init__(self, channels=2048):
        super().__init__()
        # Bottleneck design to force the model to learn semantic structure
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(channels // 4, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True) 
        )

    def forward(self, x):
        residual = x
        x = self.encoder(x)
        out = self.decoder(x)
        # Residual connection is CRITICAL here. 
        # We want to keep valid features and only fill in the missing ones.
        return residual + out