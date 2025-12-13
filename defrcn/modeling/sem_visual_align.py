import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SemanticVisualAlignment(nn.Module):
    def __init__(self, in_dim: int, clip_dim: int = 512, hidden: Optional[int] = None, hidden_dim: int = 1024, tau: float = 0.07):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, clip_dim),
        )
        # store temperature as a tensor for device-awareness
        self.register_buffer("tau", torch.tensor(tau, dtype=torch.float32))

        # text embeddings buffer (num_classes, clip_dim) -- to be set via register_buffer
        self.register_buffer("text_embeddings", None)

    def forward(self, x: torch.Tensor):
        z = self.adapter(x)
        z = F.normalize(z, dim=1)
        return z

    def contrastive_loss(self, zvis: torch.Tensor, gt_class_ids: torch.Tensor):
        """
        zvis: (N, D) normalized
        gt_class_ids: (N,) long
        """
        assert self.text_embeddings is not None, "SVA: text embeddings not set"
        # make sure on same device
        if self.text_embeddings.device != zvis.device:
            self.text_embeddings = self.text_embeddings.to(zvis.device)
        logits = (zvis @ self.text_embeddings.t()) / self.tau
        return F.cross_entropy(logits, gt_class_ids, reduction="mean")