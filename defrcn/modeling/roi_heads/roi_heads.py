import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from torch import nn
from typing import Dict
from detectron2.layers import ShapeSpec
from detectron2.utils.registry import Registry
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.utils.events import get_event_storage
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.backbone import build_backbone
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.modeling.backbone.resnet import BottleneckBlock, make_stage
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from .box_head import build_box_head
from .fast_rcnn import (
    FastRCNNOutputLayers,
    FastRCNNOutputs,
    FastRCNNContrastOutputs,
    FastRCNNMoCoOutputs,
    ContrastWithPrototypeOutputs,
    ContrastOutputsWithStorage,
    ROI_HEADS_OUTPUT_REGISTRY,
)

from ..attentive_pooling import AttentiveGlobalPooling
from ..feature_hallucinator import FeatureHallucinationModule
from ..sem_visual_align import SemanticVisualAlignment
from ..utils import concat_all_gathered, select_all_gather, cat, apply_random_block_mask , sample_indices

from ..contrastive_loss import (
    SupConLoss,
    SupConLossV2,
    ContrastiveHead,
    SupConLossWithPrototype,
    SupConLossWithStorage
)

import fvcore.nn.weight_init as weight_init



ROI_HEADS_REGISTRY = Registry("ROI_HEADS")
ROI_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""

logger = logging.getLogger(__name__)


def build_roi_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)


def select_foreground_proposals(proposals, bg_label):
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.

    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


class ROIHeads(torch.nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It contains logic of cropping the regions, extract per-region features,
    and make per-region predictions.

    It can have many variants, implemented as subclasses of this class.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super(ROIHeads, self).__init__()

        # fmt: off
        self.batch_size_per_image     = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        self.test_score_thresh        = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.test_nms_thresh          = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.test_detections_per_img  = cfg.TEST.DETECTIONS_PER_IMAGE
        self.in_features              = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes              = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.proposal_append_gt       = cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT
        self.feature_strides          = {k: v.stride for k, v in input_shape.items()}
        self.feature_channels         = {k: v.channels for k, v in input_shape.items()}
        self.cls_agnostic_bbox_reg    = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        self.smooth_l1_beta           = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA
        # fmt: on

        # Matcher to assign box proposals to gt boxes
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )

        # Box2BoxTransform for bounding box regression
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)

    def _sample_proposals(self, matched_idxs, matched_labels, gt_classes):
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]  # post_nms_top_k proposals have no matche will be drop here
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_sample_fraction, self.num_classes
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets):
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns `self.batch_size_per_image` random samples from proposals and groundtruth boxes,
        with a fraction of positives that is no larger than `self.positive_sample_fraction.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:
                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                   then the ground-truth box is random)
                Other fields such as "gt_classes" that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            # use ground truth bboxes as super-high quality proposals for training
            # with logits = math.log((1.0 - 1e-10) / (1 - (1.0 - 1e-10)))
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            # matched_idxs in [0, M)
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            iou, _ = match_quality_matrix.max(dim=0)
            # random sample batche_size_per_image proposals with positive fraction
            # NOTE: only matched proposals will be returned
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes
            proposals_per_image.iou = iou[sampled_idxs]

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # will filter the proposals again (by foreground/background,
                # etc), so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes
            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt
        # proposals_with_gt, List[Instances], fields = ['gt_boxes', 'gt_classes', ‘proposal_boxes’, 'objectness_logits']

    def forward(self, images, features, proposals, targets=None):
        """
        Args:
            images (ImageList):
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`s. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:
                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].

        Returns:
            results (list[Instances]): length `N` list of `Instances`s containing the
                detected instances. Returned during inference only; may be []
                during training.
            losses (dict[str: Tensor]): mapping from a named loss to a tensor
                storing the loss. Used during training only.
        """
        raise NotImplementedError()


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeads(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where the heads share the
    cropping and the per-region feature computation by a Res5 block.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        assert len(self.in_features) == 1

        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / self.feature_strides[self.in_features[0]], )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.res5, out_channels = self._build_res5_block(cfg)
        output_layer = cfg.MODEL.ROI_HEADS.OUTPUT_LAYER
        self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg, out_channels, self.num_classes, self.cls_agnostic_bbox_reg
        )

    def _build_res5_block(self, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        blocks = make_stage(
            BottleneckBlock,
            3,
            first_stride=2,
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks), out_channels

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        return self.res5(x)  # RoI Align 之后的 feature 进入 res5

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images

        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        pred_class_logits, pred_proposal_deltas = self.box_predictor(feature_pooled)
        del feature_pooled

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )

        if self.training:
            del features
            losses = outputs.losses()
            return [], losses
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            return pred_instances, {}


@ROI_HEADS_REGISTRY.register()
class StandardROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(StandardROIHeads, self).__init__(cfg, input_shape)
        self._init_box_head(cfg)

    def _init_box_head(self, cfg):
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        self.output_layer_name = cfg.MODEL.ROI_HEADS.OUTPUT_LAYER
        self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(self.output_layer_name)(
            cfg, self.box_head.output_size, self.num_classes, self.cls_agnostic_bbox_reg
        )

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
            proposals (List[Instance]): fields=[proposal_boxes, objectness_logits]
                post_nms_top_k proposals for each image， len = N

            targets (List[Instance]):   fields=[gt_boxes, gt_classes]
                gt_instances for each image, len = N
        """
        del images
        if self.training:
            # label and sample 256 from post_nms_top_k each images
            # has field [proposal_boxes, objectness_logits ,gt_classes, gt_boxes]
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        features_list = [features[f] for f in self.in_features]

        if self.training:
            # FastRCNNOutputs.losses()
            # {'loss_cls':, 'loss_box_reg':}
            losses = self._forward_box(features_list, proposals)  # get losses from fast_rcnn.py::FastRCNNOutputs
            return proposals, losses  # return to rcnn.py line 201
        else:
            pred_instances = self._forward_box(features_list, proposals)
            return pred_instances, {}

    def _forward_box(self, features, proposals):
        """
        Forward logic of the box prediction branch.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])  # [None, 256, POOLER_RESOLU, POOLER_RESOLU]
        box_features = self.box_head(box_features)  # [None, FC_DIM]
        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)
        del box_features

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )
        if self.training:
            return outputs.losses()
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            return pred_instances

@ROI_HEADS_REGISTRY.register()
class Res5ROIHeadsNSHEfficient(Res5ROIHeads):
    """
    Extends the official Res5ROIHeads by adding:
      - AttentiveGlobalPooling (AGP)
      - FeatureHallucinationModule (FHM)
      - SemanticVisualAlignment (SVA)
    Efficiency knobs in cfg.MODEL.NSH:
      - ENABLED (bool)
      - FHM_SAMPLE_RATIO (float)
      - FHM_MAX_SAMPLES (int)
      - FHM_SPATIAL_RED (int)
      - FHM_HIDDEN (int)
      - LAMBDA_REC, LAMBDA_SVA, LAMBDA_SP
      - FHM_TRAIN (bool) (True for base training, False for novel fine-tuning)
      - CLASS_NAMES, CLIP_MODEL, CLIP_TAU (optional for automatic CLIP text embedding load)
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        # NSH config
        ns_cfg = getattr(cfg.MODEL, "NSH", {}) or {}
        self.nsh_enabled = bool(ns_cfg.get("ENABLED", True))

        if not self.nsh_enabled:
            return

        # Efficiency knobs
        self.fhm_sample_ratio = float(ns_cfg.get("FHM_SAMPLE_RATIO", 0.25))
        self.fhm_max_samples = int(ns_cfg.get("FHM_MAX_SAMPLES", 128))
        self.fhm_spatial_red = int(ns_cfg.get("FHM_SPATIAL_RED", 1))
        self.fhm_train = bool(ns_cfg.get("FHM_TRAIN", True))  # True for base training, False for novel

        # loss weights
        self.lambda_rec = float(ns_cfg.get("LAMBDA_REC", 1.0))
        self.lambda_sva = float(ns_cfg.get("LAMBDA_SVA", 0.1))
        self.lambda_sp = float(ns_cfg.get("LAMBDA_SP", 0.01))

        # reuse out_channels set by parent (Res5ROIHeads)
        out_ch = getattr(self, "out_channels", None)
        if out_ch is None:
            # fallback (shouldn't happen if parent built res5)
            out_ch = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * (2 ** 3)
            self.out_channels = out_ch

        # NSH modules
        hidden = int(ns_cfg.get("FHM_HIDDEN", max(64, out_ch // 4)))
        self.agp = AttentiveGlobalPooling(out_ch)
        self.fhm = FeatureHallucinationModule(out_ch, hidden_channels=hidden)
        clip_dim = int(ns_cfg.get("CLIP_DIM", 512))
        self.sva = SemanticVisualAlignment(in_dim=out_ch, clip_dim=clip_dim, hidden=1024, tau=float(ns_cfg.get("CLIP_TAU", 0.07)))

        # lazy text embeddings buffer: try to auto-load text embeddings if class names provided
        class_names = ns_cfg.get("CLASS_NAMES", None)
        if class_names:
            # try to load CLIP locally (optional); if fails, user should set sva.text_embeddings externally
            try:
                import clip
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                clip_model_name = ns_cfg.get("CLIP_MODEL", "ViT-B/32")
                clip_model, _ = clip.load(clip_model_name, device=device, jit=False)
                clip_model.eval()
                with torch.no_grad():
                    prompts = [f"a photo of a {c}" for c in class_names]
                    tokens = clip.tokenize(prompts).to(device)
                    t_emb = clip_model.encode_text(tokens).float()
                    t_emb = F.normalize(t_emb, dim=1)
                    # store on CPU initially to avoid GPU memory until training
                    self.sva.text_embeddings = t_emb.cpu()
            except Exception as e:
                # do not raise; user can set text embeddings later via `self.sva.text_embeddings = tensor`
                print("Warning: automatic CLIP load for SVA failed. Set SVA text embeddings manually.", e)

    def forward(self, images, features, proposals, targets=None):
        # reuse parent's proposal labeling & sampling
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        # _shared_roi_transform is inherited from Res5ROIHeads and already does pooler+res5
        box_features = self._shared_roi_transform([features[f] for f in self.in_features], proposal_boxes)
        # box_features: (R, C, H, W)

        # default pooled features and aux losses
        pooled = None
        aux_losses = {}

        if self.nsh_enabled and self.training and self.fhm_train:
            # base training: train hallucinator and compute aux losses
            R, C, H, W = box_features.shape
            device = box_features.device

            # sample subset for hallucination (efficiency)
            inds = sample_indices(R, self.fhm_sample_ratio, max_samples=self.fhm_max_samples, device=device)

            # build masked features for the sampled indices (others remain intact)
            masked_all = box_features.clone()
            masked_subset, mask = apply_random_block_mask(box_features[inds], block_h=3, block_w=3)
            masked_all[inds] = masked_subset

            # optional spatial reduction before FHM (cheap)
            if self.fhm_spatial_red > 1:
                new_h = max(1, H // self.fhm_spatial_red)
                new_w = max(1, W // self.fhm_spatial_red)
                small = F.adaptive_avg_pool2d(masked_all, (new_h, new_w))
                recon_small = self.fhm(small)
                recon = F.interpolate(recon_small, size=(H, W), mode="bilinear", align_corners=False)
            else:
                recon = self.fhm(masked_all)

            # reconstruction loss computed on sampled indices only
            aux_losses["loss_rec"] = self.lambda_rec * F.mse_loss(recon[inds], box_features[inds], reduction="mean")

            # AGP & sparsity on reconstructed features
            pooled_rec, mask_rec, sparsity = self.agp(recon)
            aux_losses["loss_sp"] = self.lambda_sp * sparsity

            pooled = pooled_rec

            # SVA: compute only on foreground proposals if gt_classes available
            if proposals and proposals[0].has("gt_classes") and self.sva.text_embeddings is not None:
                gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0).to(device)
                fg_inds = torch.nonzero((gt_classes >= 0) & (gt_classes < self.num_classes)).squeeze(1)
                if fg_inds.numel() > 0:
                    # ensure text embeddings on same device
                    if self.sva.text_embeddings.device != pooled.device:
                        self.sva.text_embeddings = self.sva.text_embeddings.to(pooled.device)
                    zvis = self.sva(pooled[fg_inds])
                    loss_sva = self.sva.contrastive_loss(zvis, gt_classes[fg_inds].long())
                    aux_losses["loss_sva"] = self.lambda_sva * loss_sva

        elif self.nsh_enabled and self.training and (not self.fhm_train):
            # novel training: freeze FHM (no aux losses). Use frozen FHM to project.
            with torch.no_grad():
                was_train = self.fhm.training
                self.fhm.eval()
                recon = self.fhm(box_features)
                if was_train:
                    self.fhm.train(was_train)
            pooled, _, _ = self.agp(recon)

        elif self.nsh_enabled and (not self.training):
            # inference: use FHM (no gradients)
            recon = self.fhm(box_features)
            pooled, _, _ = self.agp(recon)

        # fallback pooling if none computed
        if pooled is None:
            pooled = box_features.mean(dim=[2, 3])  # (R, C)

        # standard Fast R-CNN predictions
        pred_class_logits, pred_proposal_deltas = self.box_predictor(pooled)
        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )

        if self.training:
            losses = outputs.losses()
            # merge auxiliary losses
            losses.update(aux_losses)
            return [], losses
        else:
            pred_instances, _ = outputs.inference(self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img)
            return pred_instances, {}




@ROI_HEADS_REGISTRY.register()
class Res5ROIHeadsNSHEfficient2(Res5ROIHeads):
    """
    Extends the official Res5ROIHeads by adding:
      - AttentiveGlobalPooling (AGP)
      - FeatureHallucinationModule (FHM)

    Efficiency knobs in cfg.MODEL.NSH:
      - ENABLED (bool)
      - FHM_SAMPLE_RATIO (float)
      - FHM_MAX_SAMPLES (int)
      - FHM_SPATIAL_RED (int)
      - FHM_HIDDEN (int)
      - LAMBDA_REC, LAMBDA_SP
      - FHM_TRAIN (bool) (True for base training, False for novel fine-tuning)
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        # NSH config
        ns_cfg = getattr(cfg.MODEL, "NSH", {}) or {}
        self.nsh_enabled = bool(ns_cfg.get("ENABLED", True))

        if not self.nsh_enabled:
            return

        # Efficiency knobs
        self.fhm_sample_ratio = float(ns_cfg.get("FHM_SAMPLE_RATIO", 0.25))
        self.fhm_max_samples = int(ns_cfg.get("FHM_MAX_SAMPLES", 128))
        self.fhm_spatial_red = int(ns_cfg.get("FHM_SPATIAL_RED", 1))
        self.fhm_train = bool(ns_cfg.get("FHM_TRAIN", True))  # True for base training, False for novel

        # loss weights (SVA removed)
        self.lambda_rec = float(ns_cfg.get("LAMBDA_REC", 1.0))
        self.lambda_sp = float(ns_cfg.get("LAMBDA_SP", 0.01))

        # reuse out_channels set by parent (Res5ROIHeads)
        out_ch = getattr(self, "out_channels", None)
        if out_ch is None:
            # fallback (shouldn't happen if parent built res5)
            out_ch = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * (2 ** 3)
            self.out_channels = out_ch

        # NSH modules
        hidden = int(ns_cfg.get("FHM_HIDDEN", max(64, out_ch // 4)))
        self.agp = AttentiveGlobalPooling(out_ch)
        self.fhm = FeatureHallucinationModule(out_ch, hidden_channels=hidden)

        # NOTE: SVA/CLIP-related code removed

    def forward(self, images, features, proposals, targets=None):
        # reuse parent's proposal labeling & sampling
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        # _shared_roi_transform is inherited from Res5ROIHeads and already does pooler+res5
        box_features = self._shared_roi_transform([features[f] for f in self.in_features], proposal_boxes)
        # box_features: (R, C, H, W)

        # default pooled features and aux losses
        pooled = None
        aux_losses = {}

        if self.nsh_enabled and self.training and self.fhm_train:
            # base training: train hallucinator and compute aux losses
            R, C, H, W = box_features.shape
            device = box_features.device

            # sample subset for hallucination (efficiency)
            inds = sample_indices(R, self.fhm_sample_ratio, max_samples=self.fhm_max_samples, device=device)

            # build masked features for the sampled indices (others remain intact)
            masked_all = box_features.clone()
            masked_subset, mask = apply_random_block_mask(box_features[inds], block_h=3, block_w=3)
            masked_all[inds] = masked_subset

            # optional spatial reduction before FHM (cheap)
            if self.fhm_spatial_red > 1:
                new_h = max(1, H // self.fhm_spatial_red)
                new_w = max(1, W // self.fhm_spatial_red)
                small = F.adaptive_avg_pool2d(masked_all, (new_h, new_w))
                recon_small = self.fhm(small)
                recon = F.interpolate(recon_small, size=(H, W), mode="bilinear", align_corners=False)
            else:
                recon = self.fhm(masked_all)

            # reconstruction loss computed on sampled indices only
            aux_losses["loss_rec"] = self.lambda_rec * F.mse_loss(recon[inds], box_features[inds], reduction="mean")

            # AGP & sparsity on reconstructed features
            pooled_rec, mask_rec, sparsity = self.agp(recon)
            aux_losses["loss_sp"] = self.lambda_sp * sparsity

            pooled = pooled_rec

            # SVA removed — no semantic visual alignment or CLIP losses

        elif self.nsh_enabled and self.training and (not self.fhm_train):
            # novel training: freeze FHM (no aux losses). Use frozen FHM to project.
            with torch.no_grad():
                was_train = self.fhm.training
                self.fhm.eval()
                recon = self.fhm(box_features)
                if was_train:
                    self.fhm.train(was_train)
            pooled, _, _ = self.agp(recon)

        elif self.nsh_enabled and (not self.training):
            # inference: use FHM (no gradients)
            recon = self.fhm(box_features)
            pooled, _, _ = self.agp(recon)

        # fallback pooling if none computed
        if pooled is None:
            pooled = box_features.mean(dim=[2, 3])  # (R, C)

        # standard Fast R-CNN predictions
        pred_class_logits, pred_proposal_deltas = self.box_predictor(pooled)
        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )

        if self.training:
            losses = outputs.losses()
            # merge auxiliary losses
            losses.update(aux_losses)
            return [], losses
        else:
            pred_instances, _ = outputs.inference(self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img)
            return pred_instances, {}


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeadsNSHEfficient3(Res5ROIHeads):
    """
    Res5 ROI head with NSH (AGP + FHM) improvements.
    Defensive sanitization added to avoid Inf/NaN propagation.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        ns_cfg = getattr(cfg.MODEL, "NSH", {}) or {}
        self.nsh_enabled = bool(ns_cfg.get("ENABLED", True))
        if not self.nsh_enabled:
            return

        # Efficiency knobs / hyperparams
        self.fhm_sample_ratio = float(ns_cfg.get("FHM_SAMPLE_RATIO", 0.25))
        self.fhm_max_samples = int(ns_cfg.get("FHM_MAX_SAMPLES", 128))
        self.fhm_spatial_red = int(ns_cfg.get("FHM_SPATIAL_RED", 1))
        self.fhm_train = bool(ns_cfg.get("FHM_TRAIN", True))

        # loss weights
        self.lambda_rec = float(ns_cfg.get("LAMBDA_REC", 1.0))
        self.lambda_sp = float(ns_cfg.get("LAMBDA_SP", 0.01))

        # whether to use reconstructed features for prediction (False = aux-only)
        self.use_recon_for_pred = bool(ns_cfg.get("USE_RECON_FOR_PRED", False))

        # channels: out_ch is Res5 output channels (e.g. 2048)
        out_ch = getattr(self, "out_channels", None)
        if out_ch is None:
            out_ch = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * (2 ** 3)
            self.out_channels = out_ch

        # Determine pooler (pre-Res5) channel count safely from input_shape
        try:
            in_feat_name = self.in_features[0]
            pooler_in_ch = input_shape[in_feat_name].channels
        except Exception:
            pooler_in_ch = out_ch // 2

        # modules
        hidden = int(ns_cfg.get("FHM_HIDDEN", max(64, pooler_in_ch // 4)))
        self.agp = AttentiveGlobalPooling(out_ch)
        self.fhm = FeatureHallucinationModule(pooler_in_ch, hidden_channels=hidden)

        # a small residual scaling factor to stabilize optimization (start near masked input)
        self.register_buffer(
            "fhm_res_scale",
            torch.tensor(float(ns_cfg.get("FHM_RES_SCALE", 0.01)), dtype=torch.float32),
        )

        # single-run flag for logging
        self._nans_warned = False

    def forward(self, images, features, proposals, targets=None):
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]

        # --- use pooler outputs (pre-Res5) as reconstruction targets ---
        roi_pooled = self.pooler([features[f] for f in self.in_features], proposal_boxes)

        # original res5 outputs baseline
        box_features_orig = self.res5(roi_pooled)
        pooled_orig, mask_orig, sparsity_orig = self.agp(box_features_orig)

        pooled = None
        aux_losses = {}
        nan_detected = False  # per-forward flag

        # --- helper sanitization function (available in all branches) ---
        def _sanitize(tensor, name, clip_val=1e4):
            nonlocal nan_detected
            if not torch.isfinite(tensor).all():
                if not self._nans_warned:
                    logging.warning("NSH ROIHead: non-finite values detected in %s; sanitizing.", name)
                    self._nans_warned = True
                nan_detected = True
                tensor = torch.nan_to_num(tensor, nan=0.0, posinf=clip_val, neginf=-clip_val)
            # clamp to a reasonable range to avoid extreme activations
            tensor = torch.clamp(tensor, min=-clip_val, max=clip_val)
            return tensor

        if self.nsh_enabled and self.training and self.fhm_train:
            R, C, H, W = roi_pooled.shape
            device = roi_pooled.device

            # sample subset indices for hallucination
            inds = sample_indices(R, self.fhm_sample_ratio, max_samples=self.fhm_max_samples, device=device)

            # masked subset only (avoid identity learning)
            masked_subset, _ = apply_random_block_mask(roi_pooled[inds], block_h=2, block_w=2)

            # optional spatial reduction before FHM for efficiency
            if self.fhm_spatial_red > 1:
                new_h = max(1, H // self.fhm_spatial_red)
                new_w = max(1, W // self.fhm_spatial_red)
                small_masked = F.adaptive_avg_pool2d(masked_subset, (new_h, new_w))
                recon_small = self.fhm(small_masked)
                recon_subset = F.interpolate(recon_small, size=(H, W), mode="bilinear", align_corners=False)
            else:
                recon_subset = self.fhm(masked_subset)

            # small residual scaling (keep recon close to masked input initially)
            res_scale = float(self.fhm_res_scale.item())
            recon_subset = masked_subset + res_scale * (recon_subset - masked_subset)

            # build full reconstructed tensor (replace subset positions)
            recon_all = roi_pooled.clone()
            recon_all[inds] = recon_subset

            # sanitize recon_all before res5
            recon_all = _sanitize(recon_all, "recon_all_pre_res5")

            # Pass reconstructed pre-Res5 features through res5
            box_features_recon = self.res5(recon_all)
            box_features_recon = _sanitize(box_features_recon, "box_features_recon")

            # Reconstruction loss computed only on the sampled indices (LayerNorm normalized)
            target = F.layer_norm(roi_pooled[inds], roi_pooled[inds].shape[1:])
            pred = F.layer_norm(recon_subset, recon_subset.shape[1:])
            aux_losses["loss_rec"] = self.lambda_rec * F.mse_loss(pred, target, reduction="mean")

            # AGP & sparsity on reconstructed Res5 outputs
            pooled_rec, mask_rec, sparsity_rec = self.agp(box_features_recon)
            aux_losses["loss_sp"] = self.lambda_sp * sparsity_rec

            # choose pooled features for prediction according to config
            pooled = pooled_rec if self.use_recon_for_pred else pooled_orig

        elif self.nsh_enabled and self.training and (not self.fhm_train):
            # novel training: frozen FHM projection (no aux losses)
            with torch.no_grad():
                was_train = self.fhm.training
                self.fhm.eval()
                if self.fhm_spatial_red > 1:
                    new_h = max(1, roi_pooled.shape[2] // self.fhm_spatial_red)
                    new_w = max(1, roi_pooled.shape[3] // self.fhm_spatial_red)
                    small = F.adaptive_avg_pool2d(roi_pooled, (new_h, new_w))
                    recon_small = self.fhm(small)
                    recon_all = F.interpolate(recon_small, size=(roi_pooled.shape[2], roi_pooled.shape[3]), mode="bilinear", align_corners=False)
                else:
                    recon_all = self.fhm(roi_pooled)
                if was_train:
                    self.fhm.train(was_train)
            recon_all = _sanitize(recon_all, "recon_all_novel")
            box_features_recon = self.res5(recon_all)
            box_features_recon = _sanitize(box_features_recon, "box_features_recon_novel")
            pooled, _, _ = self.agp(box_features_recon)

        elif self.nsh_enabled and (not self.training):
            # inference: full projection through FHM
            if self.fhm_spatial_red > 1:
                new_h = max(1, roi_pooled.shape[2] // self.fhm_spatial_red)
                new_w = max(1, roi_pooled.shape[3] // self.fhm_spatial_red)
                small = F.adaptive_avg_pool2d(roi_pooled, (new_h, new_w))
                recon_small = self.fhm(small)
                recon_all = F.interpolate(recon_small, size=(roi_pooled.shape[2], roi_pooled.shape[3]), mode="bilinear", align_corners=False)
            else:
                recon_all = self.fhm(roi_pooled)
            recon_all = _sanitize(recon_all, "recon_all_infer")
            box_features_recon = self.res5(recon_all)
            box_features_recon = _sanitize(box_features_recon, "box_features_recon_infer")
            pooled, _, _ = self.agp(box_features_recon)

        # fallback pooling if none computed
        if pooled is None:
            pooled = pooled_orig

        # sanitize pooled before predictor (and normalize to stabilize scale)
        pooled = _sanitize(pooled, "pooled_before_predictor")
        try:
            # apply LayerNorm to pooled features to reduce chance of extreme logits/deltas
            pooled_normed = F.layer_norm(pooled, pooled.shape[1:])
        except Exception:
            pooled_normed = pooled

        # Also sanitize proposals' proposal_boxes (replace non-finite values)
        for p in proposals:
            boxes_tensor = p.proposal_boxes.tensor
            if not torch.isfinite(boxes_tensor).all():
                if not self._nans_warned:
                    logging.warning("NSH ROIHead: non-finite values in proposal boxes; sanitizing.")
                    self._nans_warned = True
                nan_detected = True
                p.proposal_boxes.tensor = torch.nan_to_num(boxes_tensor, nan=0.0, posinf=1e4, neginf=-1e4)
                p.proposal_boxes.tensor = torch.clamp(p.proposal_boxes.tensor, min=-1e4, max=1e4)

        # standard Fast R-CNN predictions (wrapped to catch/policize NaNs)
        try:
            pred_class_logits, pred_proposal_deltas = self.box_predictor(pooled_normed)
        except Exception as e:
            # If the predictor itself crashes, sanitize inputs and retry once
            if not self._nans_warned:
                logging.warning("NSH ROIHead: box_predictor raised %s; sanitizing inputs and retrying.", e)
                self._nans_warned = True
            pooled_normed = _sanitize(pooled_normed, "pooled_normed_retry")
            pooled_normed = torch.clamp(pooled_normed, min=-1e4, max=1e4)
            pred_class_logits, pred_proposal_deltas = self.box_predictor(pooled_normed)

        # sanitize predictions
        if not torch.isfinite(pred_class_logits).all() or not torch.isfinite(pred_proposal_deltas).all():
            if not self._nans_warned:
                logging.warning("NSH ROIHead: non-finite values in predictions; sanitizing.")
                self._nans_warned = True
            nan_detected = True
            pred_class_logits = torch.nan_to_num(pred_class_logits, nan=0.0, posinf=1e4, neginf=-1e4)
            pred_proposal_deltas = torch.nan_to_num(pred_proposal_deltas, nan=0.0, posinf=1e4, neginf=-1e4)
            pred_class_logits = torch.clamp(pred_class_logits, min=-1e4, max=1e4)
            pred_proposal_deltas = torch.clamp(pred_proposal_deltas, min=-1e4, max=1e4)

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )

        if self.training:
            losses = outputs.losses()
            # attach diagnostic flag if sanitization occurred
            if nan_detected:
                # small diagnostic scalar, kept on same device as pred_class_logits
                losses["loss_nan_detected"] = torch.tensor(1.0, device=pred_class_logits.device)
            losses.update(aux_losses)
            return [], losses
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            return pred_instances, {}



@ROI_HEADS_REGISTRY.register()
class Res5ROIHeadsNSHEfficient4(Res5ROIHeads):
    """
    Res5 ROI head with NSH (AGP + FHM) improvements + SVA integration.
    Defensive sanitization kept from previous version.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        ns_cfg = getattr(cfg.MODEL, "NSH", {}) or {}
        self.nsh_enabled = bool(ns_cfg.get("ENABLED", True))
        if not self.nsh_enabled:
            return

        # Efficiency knobs / hyperparams
        self.fhm_sample_ratio = float(ns_cfg.get("FHM_SAMPLE_RATIO", 0.25))
        self.fhm_max_samples = int(ns_cfg.get("FHM_MAX_SAMPLES", 128))
        self.fhm_spatial_red = int(ns_cfg.get("FHM_SPATIAL_RED", 1))
        self.fhm_train = bool(ns_cfg.get("FHM_TRAIN", True))

        # loss weights
        self.lambda_rec = float(ns_cfg.get("LAMBDA_REC", 1.0))
        self.lambda_sva = float(ns_cfg.get("LAMBDA_SVA", 0.1))
        self.lambda_sp = float(ns_cfg.get("LAMBDA_SP", 0.01))

        # whether to use reconstructed features for prediction (False = aux-only)
        self.use_recon_for_pred = bool(ns_cfg.get("USE_RECON_FOR_PRED", False))

        # channels: out_ch is Res5 output channels (e.g. 2048)
        out_ch = getattr(self, "out_channels", None)
        if out_ch is None:
            out_ch = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * (2 ** 3)
            self.out_channels = out_ch

        # Determine pooler (pre-Res5) channel count safely from input_shape
        try:
            in_feat_name = self.in_features[0]
            pooler_in_ch = input_shape[in_feat_name].channels
        except Exception:
            pooler_in_ch = out_ch // 2

        # modules
        hidden = int(ns_cfg.get("FHM_HIDDEN", max(64, pooler_in_ch // 4)))
        self.agp = AttentiveGlobalPooling(out_ch)
        self.fhm = FeatureHallucinationModule(pooler_in_ch, hidden_channels=hidden)

        # SVA module (adapter operates on Res5 pooled dim = out_ch)
        clip_dim = int(ns_cfg.get("CLIP_DIM", 512))
        # use keyword hidden_dim for clarity/backwards-compat
        self.sva = SemanticVisualAlignment(in_dim=out_ch, clip_dim=clip_dim, hidden_dim=int(ns_cfg.get("SVA_HIDDEN", 1024)), tau=float(ns_cfg.get("CLIP_TAU", 0.07)))

        # lazy text embeddings auto-load (optional)
        class_names = ns_cfg.get("CLASS_NAMES", None)
        if class_names:
            try:
                import clip
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                clip_model_name = ns_cfg.get("CLIP_MODEL", "ViT-B/32")
                clip_model, _ = clip.load(clip_model_name, device=device, jit=False)
                clip_model.eval()
                with torch.no_grad():
                    prompts = [f"a photo of a {c}" for c in class_names]
                    tokens = clip.tokenize(prompts).to(device)
                    t_emb = clip_model.encode_text(tokens).float()
                    t_emb = F.normalize(t_emb, dim=1)
                    # store CPU to avoid reserving GPU memory; moved lazily when needed
                    self.sva.text_embeddings = t_emb.cpu()
            except Exception as e:
                logging.warning("Automatic CLIP load for SVA failed; set SVA text_embeddings manually. %s", e)

        # a small residual scaling factor to stabilize optimization (start near masked input)
        self.register_buffer(
            "fhm_res_scale",
            torch.tensor(float(ns_cfg.get("FHM_RES_SCALE", 0.01)), dtype=torch.float32),
        )

        # single-run flag for logging
        self._nans_warned = False
        self._text_emb_on_device = False  # track lazy move of text embeddings

    def _ensure_text_emb_on_device(self, device):
        """
        Move text embeddings to `device` once (lazy). Keep as buffer on module.
        """
        te = getattr(self.sva, "text_embeddings", None)
        if te is None:
            return
        if not self._text_emb_on_device or te.device != device:
            # move (and keep flag)
            self.sva.text_embeddings = te.to(device)
            self._text_emb_on_device = True

    def forward(self, images, features, proposals, targets=None):
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]

        # --- use pooler outputs (pre-Res5) as reconstruction targets ---
        roi_pooled = self.pooler([features[f] for f in self.in_features], proposal_boxes)

        # original res5 outputs baseline
        box_features_orig = self.res5(roi_pooled)
        pooled_orig, mask_orig, sparsity_orig = self.agp(box_features_orig)

        pooled = None
        aux_losses = {}
        nan_detected = False  # per-forward flag

        # sanitizer helper
        def _sanitize(tensor, name, clip_val=1e4):
            nonlocal nan_detected
            if not torch.isfinite(tensor).all():
                if not self._nans_warned:
                    logging.warning("NSH ROIHead: non-finite values detected in %s; sanitizing.", name)
                    self._nans_warned = True
                nan_detected = True
                tensor = torch.nan_to_num(tensor, nan=0.0, posinf=clip_val, neginf=-clip_val)
            tensor = torch.clamp(tensor, min=-clip_val, max=clip_val)
            return tensor

        if self.nsh_enabled and self.training and self.fhm_train:
            R, C, H, W = roi_pooled.shape
            device = roi_pooled.device

            # sample subset indices for hallucination
            inds = sample_indices(R, self.fhm_sample_ratio, max_samples=self.fhm_max_samples, device=device)

            # masked subset only (avoid identity learning)
            masked_subset, _ = apply_random_block_mask(roi_pooled[inds], block_h=2, block_w=2)

            # optional spatial reduction before FHM for efficiency
            if self.fhm_spatial_red > 1:
                new_h = max(1, H // self.fhm_spatial_red)
                new_w = max(1, W // self.fhm_spatial_red)
                small_masked = F.adaptive_avg_pool2d(masked_subset, (new_h, new_w))
                recon_small = self.fhm(small_masked)
                recon_subset = F.interpolate(recon_small, size=(H, W), mode="bilinear", align_corners=False)
            else:
                recon_subset = self.fhm(masked_subset)

            # small residual scaling (keep recon close to masked input initially)
            res_scale = float(self.fhm_res_scale.item())
            recon_subset = masked_subset + res_scale * (recon_subset - masked_subset)

            # build full reconstructed tensor (replace subset positions)
            recon_all = roi_pooled.clone()
            recon_all[inds] = recon_subset
            recon_all = _sanitize(recon_all, "recon_all_pre_res5")

            # Pass reconstructed pre-Res5 features through res5
            box_features_recon = self.res5(recon_all)
            box_features_recon = _sanitize(box_features_recon, "box_features_recon")

            # Reconstruction loss computed only on the sampled indices (LayerNorm normalized)
            target = F.layer_norm(roi_pooled[inds], roi_pooled[inds].shape[1:])
            pred = F.layer_norm(recon_subset, recon_subset.shape[1:])
            aux_losses["loss_rec"] = self.lambda_rec * F.mse_loss(pred, target, reduction="mean")

            # AGP & sparsity on reconstructed Res5 outputs
            pooled_rec, mask_rec, sparsity_rec = self.agp(box_features_recon)
            aux_losses["loss_sp"] = self.lambda_sp * sparsity_rec

            # choose pooled features for prediction according to config
            pooled = pooled_rec if self.use_recon_for_pred else pooled_orig

            # ------------------ SVA (contrastive) ------------------
            # compute only if text embeddings are provided
            if proposals and proposals[0].has("gt_classes") and getattr(self.sva, "text_embeddings", None) is not None:
                # prepare gt class ids and foreground mask
                gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0).to(device)
                fg_inds = torch.nonzero((gt_classes >= 0) & (gt_classes < self.num_classes)).squeeze(1)
                if fg_inds.numel() > 0:
                    # ensure text embeddings on same device
                    self._ensure_text_emb_on_device(device)
                    # defensive: ensure dtype float32 and normalized
                    te = self.sva.text_embeddings
                    if te is not None:
                        if te.dtype != torch.float32:
                            te = te.float()
                        te = F.normalize(te, dim=1)
                        self.sva.text_embeddings = te
                        # compute visual embeddings from pooled (use pooled, not pooled_rec necessarily)
                        vis_feats = self.sva(pooled[fg_inds])
                        loss_sva = self.sva.contrastive_loss(vis_feats, gt_classes[fg_inds].long())
                        aux_losses["loss_sva"] = self.lambda_sva * loss_sva

        elif self.nsh_enabled and self.training and (not self.fhm_train):
            # novel training: frozen FHM projection (no aux losses)
            with torch.no_grad():
                was_train = self.fhm.training
                self.fhm.eval()
                if self.fhm_spatial_red > 1:
                    new_h = max(1, roi_pooled.shape[2] // self.fhm_spatial_red)
                    new_w = max(1, roi_pooled.shape[3] // self.fhm_spatial_red)
                    small = F.adaptive_avg_pool2d(roi_pooled, (new_h, new_w))
                    recon_small = self.fhm(small)
                    recon_all = F.interpolate(recon_small, size=(roi_pooled.shape[2], roi_pooled.shape[3]), mode="bilinear", align_corners=False)
                else:
                    recon_all = self.fhm(roi_pooled)
                if was_train:
                    self.fhm.train(was_train)
            recon_all = _sanitize(recon_all, "recon_all_novel")
            box_features_recon = self.res5(recon_all)
            box_features_recon = _sanitize(box_features_recon, "box_features_recon_novel")
            pooled, _, _ = self.agp(box_features_recon)

        elif self.nsh_enabled and (not self.training):
            # inference: full projection through FHM
            if self.fhm_spatial_red > 1:
                new_h = max(1, roi_pooled.shape[2] // self.fhm_spatial_red)
                new_w = max(1, roi_pooled.shape[3] // self.fhm_spatial_red)
                small = F.adaptive_avg_pool2d(roi_pooled, (new_h, new_w))
                recon_small = self.fhm(small)
                recon_all = F.interpolate(recon_small, size=(roi_pooled.shape[2], roi_pooled.shape[3]), mode="bilinear", align_corners=False)
            else:
                recon_all = self.fhm(roi_pooled)
            recon_all = _sanitize(recon_all, "recon_all_infer")
            box_features_recon = self.res5(recon_all)
            box_features_recon = _sanitize(box_features_recon, "box_features_recon_infer")
            pooled, _, _ = self.agp(box_features_recon)

        # fallback pooling if none computed
        if pooled is None:
            pooled = pooled_orig

        # sanitize pooled before predictor (and normalize to stabilize scale)
        pooled = _sanitize(pooled, "pooled_before_predictor")
        try:
            pooled_normed = F.layer_norm(pooled, pooled.shape[1:])
        except Exception:
            pooled_normed = pooled

        # Also sanitize proposals' proposal_boxes (replace non-finite values)
        for p in proposals:
            boxes_tensor = p.proposal_boxes.tensor
            if not torch.isfinite(boxes_tensor).all():
                if not self._nans_warned:
                    logging.warning("NSH ROIHead: non-finite values in proposal boxes; sanitizing.")
                    self._nans_warned = True
                nan_detected = True
                p.proposal_boxes.tensor = torch.nan_to_num(boxes_tensor, nan=0.0, posinf=1e4, neginf=-1e4)
                p.proposal_boxes.tensor = torch.clamp(p.proposal_boxes.tensor, min=-1e4, max=1e4)

        # standard Fast R-CNN predictions (wrapped to catch/policize NaNs)
        try:
            pred_class_logits, pred_proposal_deltas = self.box_predictor(pooled_normed)
        except Exception as e:
            if not self._nans_warned:
                logging.warning("NSH ROIHead: box_predictor raised %s; sanitizing inputs and retrying.", e)
                self._nans_warned = True
            pooled_normed = _sanitize(pooled_normed, "pooled_normed_retry")
            pooled_normed = torch.clamp(pooled_normed, min=-1e4, max=1e4)
            pred_class_logits, pred_proposal_deltas = self.box_predictor(pooled_normed)

        # sanitize predictions
        if not torch.isfinite(pred_class_logits).all() or not torch.isfinite(pred_proposal_deltas).all():
            if not self._nans_warned:
                logging.warning("NSH ROIHead: non-finite values in predictions; sanitizing.")
                self._nans_warned = True
            nan_detected = True
            pred_class_logits = torch.nan_to_num(pred_class_logits, nan=0.0, posinf=1e4, neginf=-1e4)
            pred_proposal_deltas = torch.nan_to_num(pred_proposal_deltas, nan=0.0, posinf=1e4, neginf=-1e4)
            pred_class_logits = torch.clamp(pred_class_logits, min=-1e4, max=1e4)
            pred_proposal_deltas = torch.clamp(pred_proposal_deltas, min=-1e4, max=1e4)

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )

        if self.training:
            losses = outputs.losses()
            # attach diagnostic flag if sanitization occurred
            if nan_detected:
                losses["loss_nan_detected"] = torch.tensor(1.0, device=pred_class_logits.device)
            losses.update(aux_losses)
            return [], losses
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            return pred_instances, {}
