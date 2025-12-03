from .meta_arch import META_ARCH_REGISTRY, GeneralizedRCNN, build_model
from .roi_heads import (
    ROI_BOX_HEAD_REGISTRY, ROI_HEADS_REGISTRY, ROIHeads, StandardROIHeads, build_box_head,
    build_roi_heads)
import torch

from .utils import (
    concat_all_gathered,select_all_gather, cat
)
from .contrastive_loss import (
    ContrastiveHead,
    SupConLoss,
)

_EXCLUDE = {"torch", "ShapeSpec"}
__all__ = [k for k in globals().keys() if k not in _EXCLUDE and not k.startswith("_")]

assert (
    torch.Tensor([1]) == torch.Tensor([2])
).dtype == torch.bool, "Your Pytorch is too old. Please update to contain https://github.com/pytorch/pytorch/pull/21113"