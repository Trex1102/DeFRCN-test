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

        # gamma = 2.0

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

            # # ======= CHANGES START =======
            # # OLD (removed) line:
            # # loss_i = -torch.log((numer + EPS) / (den + EPS))
            # # REPLACED BY focal-style probability + loss:
            # p = (numer + EPS) / (den + EPS)             # <-- ADDED: probability of sampling a positive
            # p = torch.clamp(p, min=EPS, max=1.0 - EPS)  # <-- ADDED: numerical safety clamp
            # loss_i = -((1.0 - p) ** gamma) * torch.log(p + EPS)  # <-- CHANGED: focal-style loss
            # # ======= CHANGES END =======

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

    #     anchor_mask = fg_mask.clone() & (~ignore_mask)

    #     labels_expand = labels.unsqueeze(0).expand(N, N)
    #     pos_mask = (labels_expand == labels_expand.T) & (~ignore_mask.unsqueeze(0).expand(N, N))
        

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



    # def _class_supervised_contrastive(self, z_cls: torch.Tensor, labels: torch.Tensor, fg_mask: torch.Tensor,
    #                               ignore_mask: torch.Tensor, iou_weight: torch.Tensor):
    #     """
    #     Implements:
    #     L = (1 / sum_w) * sum_i w_i * L_{z_i}
    #     where
    #     L_{z_i} = - (1 / n_pos) * sum_{j in P(i)} log( exp(sim_ij) / sum_{k != i, not ignored} exp(sim_ik) )
    #     and P(i) are positives (same class, label != 0, not ignored).
    #     """
    #     device = z_cls.device
    #     N = z_cls.shape[0]
    #     EPS = 1e-8

    #     anchor_mask = fg_mask.clone() & (~ignore_mask)           # which anchors to compute loss for

    #     # pairwise same-label positives (exclude ignored proposals)
    #     labels_expand = labels.unsqueeze(0).expand(N, N)         # [N, N]
    #     pos_mask = (labels_expand == labels_expand.T) & (~ignore_mask.unsqueeze(0).expand(N, N))
    #     pos_mask = pos_mask & (labels_expand != 0)              # remove label 0 as positive

    #     # pairwise similarity
    #     sim = torch.matmul(z_cls, z_cls.T) / self.tau            # [N, N]
    #     diag = torch.eye(N, dtype=torch.bool, device=device)

    #     # build valid mask for denominator: exclude self and ignored items
    #     valid_for_den = (~diag) & (~ignore_mask.unsqueeze(0).expand(N, N))

    #     # compute exponentials only where valid (to exclude ignored in denom)
    #     sim_exp = torch.exp(sim) * valid_for_den.float()        # [N, N]
    #     denom = sim_exp.sum(dim=1) + EPS                        # [N]

    #     per_anchor_losses = []
    #     per_anchor_weights = []
    #     for i_idx in torch.nonzero(anchor_mask, as_tuple=False).view(-1):
    #         i = int(i_idx.item())
    #         positives = pos_mask[i].clone()
    #         positives[i] = False
    #         n_pos = int(positives.sum().item())
    #         if n_pos == 0:
    #             continue

    #         # numerator exponentials for positives (only those not ignored and same class)
    #         numer_exp = (torch.exp(sim[i]) * positives.float())   # [N]

    #         # probabilities for each positive: exp(sim_ij) / denom_i
    #         probs = numer_exp / (denom[i] + EPS)                 # [N]

    #         # per-anchor loss = average negative log probability over positives
    #         # add EPS for numerical safety inside log
    #         loss_i = - (torch.log(probs + EPS).sum()) / float(n_pos)

    #         # IoU reweight if requested (f(u_i))
    #         w = iou_weight[i] if getattr(self, "use_iou_reweight", False) else 1.0

    #         per_anchor_losses.append(loss_i * w)
    #         per_anchor_weights.append(w)

    #     if len(per_anchor_losses) == 0:
    #         return torch.tensor(0., device=device, requires_grad=True)

    #     per_anchor_losses = torch.stack(per_anchor_losses)               # [M]
    #     per_anchor_weights = torch.tensor(per_anchor_weights, device=device)  # [M]

    #     # weighted average over anchors (matches 1/N sum f(u_i) * L_z_i up to normalization by sum_w)
    #     loss = per_anchor_losses.sum() / (per_anchor_weights.sum() + EPS)
    #     return loss


    def _class_supervised_contrastive_stable(self, z_cls: torch.Tensor, labels: torch.Tensor,
                                         fg_mask: torch.Tensor, ignore_mask: torch.Tensor,
                                         iou_weight: torch.Tensor):
        device = z_cls.device
        N = z_cls.shape[0]
        EPS = 1e-12

        anchor_mask = fg_mask.clone() & (~ignore_mask)  # anchors to compute loss for

        # pairwise same-label positives (exclude ignored proposals)
        labels_expand = labels.unsqueeze(0).expand(N, N)         # [N, N]
        pos_mask = (labels_expand == labels_expand.T) & (~ignore_mask.unsqueeze(0).expand(N, N))
        pos_mask = pos_mask & (labels_expand != 0)              # remove label 0 as positive

        # pairwise similarity logits
        sim = torch.matmul(z_cls, z_cls.T) / self.tau           # [N, N]
        diag = torch.eye(N, dtype=torch.bool, device=device)

        # valid entries in denominator: exclude self and ignored items
        valid_for_den = (~diag) & (~ignore_mask.unsqueeze(0).expand(N, N))

        per_anchor_losses = []
        per_anchor_weights = []

        for i_idx in torch.nonzero(anchor_mask, as_tuple=False).view(-1):
            i = int(i_idx.item())

            # mask for valid logits (exclude self and ignored)
            valid_mask = valid_for_den[i]  # bool mask [N]
            if not valid_mask.any():
                continue

            # positives for this anchor (exclude self)
            positives = pos_mask[i].clone()
            positives[i] = False
            positives = positives & valid_mask  # ensure positives are valid (not ignored)
            n_pos = int(positives.sum().item())
            if n_pos == 0:
                continue

            # Extract logits for valid entries
            logits_valid = sim[i][valid_mask]   # shape [K]
            # compute log-sum-exp for denom in a stable way
            denom_log = torch.logsumexp(logits_valid, dim=0)  # scalar

            # we need the logits of positives (they are a subset of valid_mask)
            # map the positive indices into the valid_logits positions:
            # find indices of valid positions
            valid_indices = torch.nonzero(valid_mask, as_tuple=False).view(-1)  # global indices
            # create boolean selection among valid entries
            pos_in_valid = torch.isin(valid_indices, torch.nonzero(positives, as_tuple=False).view(-1))

            logits_pos = logits_valid[pos_in_valid]  # [n_pos]

            # sum of log-probs over positives: sum_j (logits_pos_j - denom_log)
            sum_log_probs = (logits_pos - denom_log).sum()

            # per-anchor loss = - (1/n_pos) * sum_log_probs
            loss_i = - sum_log_probs / float(n_pos)

            w = iou_weight[i] if getattr(self, "use_iou_reweight", False) else 1.0
            per_anchor_losses.append(loss_i * w)
            per_anchor_weights.append(w)

        if len(per_anchor_losses) == 0:
            return torch.tensor(0., device=device, requires_grad=True)

        per_anchor_losses = torch.stack(per_anchor_losses)
        per_anchor_weights = torch.tensor(per_anchor_weights, device=device, device=device)

        loss = per_anchor_losses.sum() / (per_anchor_weights.sum() + EPS)
        return loss