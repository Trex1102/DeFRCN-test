import torch
import torch.nn as nn
import torch.nn.functional as F



class AttentiveGlobalPooling(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        mid = max(1, in_channels // reduction)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor):
        # x: (N, C, H, W)
        mask_logits = self.net(x)                    # (N,1,H,W)
        mask = torch.sigmoid(mask_logits)
        # mask_sum = mask.sum(dim=[2, 3], keepdim=True).clamp(min=1e-6)
        # pooled = (x * mask).sum(dim=[2, 3]) / mask_sum.view(mask_sum.size(0), 1)
        pooled = (mask * x).sum(dim=[2, 3])
        sparsity = mask.mean()
        return pooled, mask, sparsity