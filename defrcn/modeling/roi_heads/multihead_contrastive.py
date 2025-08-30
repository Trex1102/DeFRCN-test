import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

EPS = 1e-8

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadContrastive(nn.Module):
    """
    Multi-head contrastive module for DeFRCN / Faster R-CNN RoI features.

    - fg_bg_head: projects RoI features to an embedding used for foreground-vs-background contrast.
    - class_head: projects RoI features to an embedding used for class-level supervised contrast.

    Forward:
        roi_feats: Tensor [N, C]           - RoI features (after box head / before classifier)
        labels: Tensor [N]                 - integer labels: -1 (ignore), 0 (background), 1..K (classes)
        ious: Optional[Tensor [N]]         - IoU between proposal and its matched GT (0 for background)
                                             If None, IoU reweighting is disabled.
    Returns:
        dict with keys 'loss_fg_bg' and 'loss_supcon' (scalars).
    """
    def __init__(
        self,
        feat_dim: int,
        proj_hidden: int = 256,
        fg_bg_dim: int = 64,
        class_dim: int = 128,
        tau: float = 0.2,
        iou_threshold: float = 0.5,
        use_iou_reweight: bool = True,
        bg_as_negatives_only: bool = True,
        loss_weights: Tuple[float, float] = (1.0, 1.0),  # (fg_bg_weight, class_weight)
    ):
        super().__init__()
        self.fg_bg_proj = ProjectionHead(feat_dim, proj_hidden, fg_bg_dim)
        self.class_proj = ProjectionHead(feat_dim, proj_hidden, class_dim)
        self.tau = tau
        self.iou_threshold = iou_threshold
        self.use_iou_reweight = use_iou_reweight
        self.bg_as_negatives_only = bg_as_negatives_only
        self.fg_bg_weight, self.class_weight = loss_weights

    def _normalize(self, z):
        # l2 normalize with small eps
        return F.normalize(z, p=2, dim=1, eps=EPS)

    def forward(self, roi_feats: torch.Tensor, labels: torch.Tensor, ious: Optional[torch.Tensor] = None):
        """
        roi_feats: [N, C]
        labels:   [N]    ints (-1 ignore, 0 bg, >0 class)
        ious:     [N]    floats in [0,1] or None
        """
        device = roi_feats.device
        N = roi_feats.shape[0]

        # project + normalize
        z_fg = self._normalize(self.fg_bg_proj(roi_feats))    # [N, D1]
        z_cls = self._normalize(self.class_proj(roi_feats))   # [N, D2]

        labels = labels.clone().to(device)
        if ious is None:
            ious = torch.zeros(N, device=device)
        else:
            ious = ious.clone().to(device)

        # masks
        ignore_mask = labels == -1
        bg_mask = (labels == 0) & (~ignore_mask)
        fg_mask = (labels > 0) & (~ignore_mask)

        # IoU weight function f(u): configurable. Default: linear reweight above threshold
        if self.use_iou_reweight:
            iou_weight = torch.where(ious > self.iou_threshold, ious, torch.zeros_like(ious))
        else:
            iou_weight = torch.ones_like(ious)

        # FG/BG contrastive loss
        loss_fg_bg = self._fg_bg_contrastive(z_fg, fg_mask, bg_mask, ignore_mask, iou_weight)

        # class-level supervised contrastive loss (only on foreground proposals)
        loss_supcon = self._class_supervised_contrastive(z_cls, labels, fg_mask, ignore_mask, iou_weight)

        return {
            "loss_fg_bg": loss_fg_bg * self.fg_bg_weight,
            "loss_supcon": loss_supcon * self.class_weight
        }

    def _compute_supcon_loss(self, z_norm: torch.Tensor, anchor_mask: torch.Tensor, pos_mask: torch.Tensor):
        """
        Generic supervised contrastive mechanics.
        - z_norm: [N, D] normalized embeddings
        - anchor_mask: [N] bool mask indicating which anchors to compute loss for
        - pos_mask: [N, N] bool, pos_mask[i,j] True if j is positive for anchor i (excluding self)
        Returns scalar loss (averaged over valid anchors)
        """
        device = z_norm.device
        N = z_norm.shape[0]
        sim = torch.matmul(z_norm, z_norm.T)  # cosine since normalized
        sim = sim / self.tau

        # mask self
        diag = torch.eye(N, dtype=torch.bool, device=device)
        sim_exp = torch.exp(sim) * (~diag).float()  # zero out self-sim before exp was applied but keep matrix shape
        # denominator for each anchor: sum over k != i of exp(sim_ik)
        denom = sim_exp.sum(dim=1)  # [N]

        losses = []
        valid_anchor_count = 0
        for i in torch.nonzero(anchor_mask, as_tuple=False).view(-1):
            i = int(i.item())
            positives = pos_mask[i]  # boolean tensor length N
            positives[i] = False  # ensure exclude self
            n_pos = positives.sum().item()
            if n_pos == 0:
                # no positives for this anchor -> skip
                continue
            # numerator: sum over positive j of exp(sim_ij)
            numer = (torch.exp(sim[i]) * positives.float()).sum()
            # denom: sum over all k != i
            den = denom[i] + EPS
            loss_i = - torch.log((numer + EPS) / (den + EPS))
            losses.append(loss_i)
            valid_anchor_count += 1

        if valid_anchor_count == 0:
            return torch.tensor(0., device=device, requires_grad=True)
        return torch.stack(losses).mean()

    def _fg_bg_contrastive(self, z_fg: torch.Tensor, fg_mask: torch.Tensor, bg_mask: torch.Tensor,
                           ignore_mask: torch.Tensor, iou_weight: torch.Tensor):
        """
        Foreground-background contrastive loss.
        Options:
         - bg_as_negatives_only=True -> compute loss only for foreground anchors; background acts only as negatives.
         - if bg_as_negatives_only=False -> compute loss for both fg and bg anchors (pos = same fg/bg label).
        IoU weighting scales the anchor loss by iou_weight[i].
        """
        device = z_fg.device
        N = z_fg.shape[0]

        if self.bg_as_negatives_only:
            anchor_mask = fg_mask.clone()
            # positives for a fg anchor: other fg proposals
            pos_mask = (fg_mask.unsqueeze(0).expand(N, N))
            # exclude self later in generic function
        else:
            # anchors are both fg and bg (but not ignored)
            anchor_mask = (~ignore_mask)
            # positives are same fg/bg label pairs
            fgbg_labels = (fg_mask.float())  # 1 for fg, 0 for bg
            pos_mask = (fgbg_labels.unsqueeze(0) == fgbg_labels.unsqueeze(1))

        # now compute generic supcon-like for this binary grouping, but apply IoU weights to anchor losses
        # We'll compute per-anchor losses and then weight & average
        # Build pos_mask excluding ignored proposals
        pos_mask = pos_mask & (~ignore_mask.unsqueeze(0).expand(N, N))
        anchor_mask = anchor_mask & (~ignore_mask)

        # compute sim matrix & exp sims (done inside helper but we need per-anchor numerator)
        sim = torch.matmul(z_fg, z_fg.T) / self.tau
        diag = torch.eye(N, dtype=torch.bool, device=device)
        sim_exp = torch.exp(sim) * (~diag).float()
        denom = sim_exp.sum(dim=1)  # [N]

        per_anchor_losses = []
        per_anchor_weights = []
        for i in torch.nonzero(anchor_mask, as_tuple=False).view(-1):
            i = int(i.item())
            positives = pos_mask[i].clone()
            positives[i] = False
            n_pos = positives.sum().item()
            if n_pos == 0:
                continue
            numer = (torch.exp(sim[i]) * positives.float()).sum()
            den = denom[i] + EPS
            loss_i = -torch.log((numer + EPS) / (den + EPS))
            # weight by IoU (if enabled). For background-as-negatives-only we only computed anchors for fg, so iou corresponds to gt IoU.
            w = iou_weight[i] if self.use_iou_reweight else 1.0
            per_anchor_losses.append(loss_i * w)
            per_anchor_weights.append(w)

        if len(per_anchor_losses) == 0:
            return torch.tensor(0., device=device, requires_grad=True)
        per_anchor_losses = torch.stack(per_anchor_losses)
        per_anchor_weights = torch.tensor(per_anchor_weights, device=device)
        # normalize by sum of weights (so that absolute scale is stable)
        return per_anchor_losses.sum() / (per_anchor_weights.sum() + EPS)

    # def _class_supervised_contrastive(self, z_cls: torch.Tensor, labels: torch.Tensor, fg_mask: torch.Tensor,
    #                                   ignore_mask: torch.Tensor, iou_weight: torch.Tensor):
    #     """
    #     Supervised contrastive on class labels. Only anchors that are foreground (labels>0) are considered.
    #     Positives for anchor i are other proposals with same class label.
    #     IoU weight applied per anchor (as in FSCE).
    #     """
    #     device = z_cls.device
    #     N = z_cls.shape[0]
    #     # anchor candidates: only fg (exclude bg and ignore)
    #     anchor_mask = fg_mask.clone() & (~ignore_mask)

    #     # pos_mask[i,j] = True if labels[i] == labels[j] AND both not ignored
    #     labels_expand = labels.unsqueeze(0).expand(N, N)
    #     pos_mask = (labels_expand == labels_expand.T) & (~ignore_mask.unsqueeze(0).expand(N, N))
    #     # exclude background from positives (bg label==0)
    #     pos_mask = pos_mask & (labels_expand != 0)

    #     # compute per-anchor numerator/denom like supcon
    #     sim = torch.matmul(z_cls, z_cls.T) / self.tau
    #     diag = torch.eye(N, dtype=torch.bool, device=device)
    #     sim_exp = torch.exp(sim) * (~diag).float()
    #     denom = sim_exp.sum(dim=1)  # [N]

    #     per_anchor_losses = []
    #     per_anchor_weights = []
    #     for i in torch.nonzero(anchor_mask, as_tuple=False).view(-1):
    #         i = int(i.item())
    #         positives = pos_mask[i].clone()
    #         positives[i] = False
    #         n_pos = positives.sum().item()
    #         if n_pos == 0:
    #             continue
    #         numer = (torch.exp(sim[i]) * positives.float()).sum()
    #         den = denom[i] + EPS
    #         loss_i = -torch.log((numer + EPS) / (den + EPS))
    #         w = iou_weight[i] if self.use_iou_reweight else 1.0
    #         per_anchor_losses.append(loss_i * w)
    #         per_anchor_weights.append(w)

    #     if len(per_anchor_losses) == 0:
    #         return torch.tensor(0., device=device, requires_grad=True)
    #     per_anchor_losses = torch.stack(per_anchor_losses)
    #     per_anchor_weights = torch.tensor(per_anchor_weights, device=device)
    #     return per_anchor_losses.sum() / (per_anchor_weights.sum() + EPS)

    def _class_supervised_contrastive(self, z_cls: torch.Tensor, labels: torch.Tensor, fg_mask: torch.Tensor,
                                  ignore_mask: torch.Tensor, iou_weight: torch.Tensor):
        """
        Supervised contrastive on class labels (implements the equation from your image).
        Per-anchor L_{z_i} = -(1 / (N_yi - 1)) * sum_{j in Pos(i)} log p_ij
        where p_ij = exp(s_ij) / sum_{k != i} exp(s_ik).

        Vectorized implementation; returns weighted average over valid anchors.
        """
        device = z_cls.device
        N = z_cls.shape[0]

        # anchor candidates: only fg (exclude bg and ignore)
        anchor_mask = fg_mask.clone() & (~ignore_mask)

        # pos_mask[i,j] = True if labels[i] == labels[j] AND both not ignored, exclude background (label==0)
        labels_expand = labels.unsqueeze(0).expand(N, N)
        pos_mask = (labels_expand == labels_expand.T) & (~ignore_mask.unsqueeze(0).expand(N, N))
        pos_mask = pos_mask & (labels_expand != 0)

        # similarity matrix
        sim = torch.matmul(z_cls, z_cls.T) / self.tau  # [N, N]

        # exclude self by setting diagonal to -inf for log-sum-exp stability
        diag = torch.eye(N, dtype=torch.bool, device=device)
        sim_masked = sim.masked_fill(diag, float("-inf"))  # ensures k != i

        # log denominator per anchor: log sum_{k != i} exp(s_ik)
        log_denom = torch.logsumexp(sim_masked, dim=1)  # [N], may be -inf if all -inf

        # log p_ij = s_ij - log_denom[i]  (for k!=i). Using sim_masked ensures self is -inf.
        # Expand log_denom to subtract from sim_masked; where log_denom is -inf we'll handle later.
        log_denom_exp = log_denom.unsqueeze(1)  # [N,1]
        log_p = sim_masked - log_denom_exp  # [N,N], may contain -inf or nan for invalid rows

        # ensure numerical sanity: replace -inf/nan in log_p with large negative (so sums are safe)
        # but we'll only use rows/anchors that are valid below
        log_p = torch.where(torch.isfinite(log_p), log_p, torch.tensor(float("-1e9"), device=device))

        # count positives per anchor (excluding i)
        n_pos = pos_mask.sum(dim=1).float()  # [N]

        # sum of log_p over positive indices per anchor
        sum_log_p_pos = (log_p * pos_mask.float()).sum(dim=1)  # [N]

        # valid anchors: anchor_mask True, at least one positive, and denom finite
        valid_anchors = anchor_mask & (n_pos > 0) & torch.isfinite(log_denom)

        if valid_anchors.sum() == 0:
            return torch.tensor(0., device=device, requires_grad=True)

        # compute per-anchor L_{z_i} = - (1 / n_pos) * sum_log_p_pos for valid anchors
        Lzi = torch.zeros(N, device=device)
        Lzi[valid_anchors] = - sum_log_p_pos[valid_anchors] / (n_pos[valid_anchors] + EPS)

        # weights: IoU weighting if enabled, otherwise ones
        if self.use_iou_reweight:
            weights = iou_weight.to(device) * valid_anchors.float()
        else:
            weights = valid_anchors.float()

        # final weighted average (normalize by sum of weights to keep scale stable)
        loss = (weights * Lzi).sum() / (weights.sum() + EPS)
        return loss
