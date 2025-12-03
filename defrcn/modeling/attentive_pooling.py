import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentiveGlobalPooling(nn.Module):
    def __init__(self, in_channels, pool_size=7):
        """
        Args:
            in_channels (int): Depth of input feature map (e.g., 2048 for ResNet101).
            pool_size (int): Spatial dimension of RoI (usually 7).
        """
        super(AttentiveGlobalPooling, self).__init__()
        
        # 1. Spatial Attention Head
        # Simple 1x1 conv to squash channels to 1 attention map
        # You can use 3x3 if you want to consider local context
        self.attn_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, kernel_size=1),
            nn.Sigmoid() # Output score [0, 1] for relevance
        )

    def forward(self, x):
        """
        Args:
            x: RoI Features of shape (N, C, H, W) -> e.g., (Batch, 2048, 7, 7)
        Returns:
            pooled_feat: (N, C)
        """
        # Calculate Attention Map (N, 1, H, W)
        attn_map = self.attn_conv(x)
        
        # Apply Attention to Features
        # Element-wise multiplication: (N, C, H, W) * (N, 1, H, W)
        weighted_features = x * attn_map
        
        # Global Sum Pooling over Spatial Dimensions (H, W)
        # We sum instead of average because the attention weights handle the normalization
        pooled_feat = torch.sum(weighted_features, dim=(2, 3))
        
        # Optional: Add the original Global Avg Pool as a residual connection
        # to prevent feature collapse early in training
        # pooled_feat = pooled_feat + F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        
        return pooled_feat