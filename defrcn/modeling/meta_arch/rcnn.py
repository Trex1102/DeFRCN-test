# import torch
# import logging
# from torch import nn
# from detectron2.structures import ImageList
# from detectron2.utils.logger import log_first_n
# from detectron2.modeling.backbone import build_backbone
# from detectron2.modeling.postprocessing import detector_postprocess
# from detectron2.modeling.proposal_generator import build_proposal_generator
# from .build import META_ARCH_REGISTRY
# from .gdl import decouple_layer, AffineLayer
# from defrcn.modeling.roi_heads import build_roi_heads

# __all__ = ["GeneralizedRCNN"]


# @META_ARCH_REGISTRY.register()
# class GeneralizedRCNN(nn.Module):

#     def __init__(self, cfg):
#         super().__init__()

#         self.cfg = cfg
#         self.device = torch.device(cfg.MODEL.DEVICE)
#         self.backbone = build_backbone(cfg)
#         self._SHAPE_ = self.backbone.output_shape()
#         self.proposal_generator = build_proposal_generator(cfg, self._SHAPE_)
#         self.roi_heads = build_roi_heads(cfg, self._SHAPE_)
#         self.normalizer = self.normalize_fn()
#         self.affine_rpn = AffineLayer(num_channels=self._SHAPE_['res4'].channels, bias=True)
#         self.affine_rcnn = AffineLayer(num_channels=self._SHAPE_['res4'].channels, bias=True)
#         self.to(self.device)

#         if cfg.MODEL.BACKBONE.FREEZE:
#             for p in self.backbone.parameters():
#                 p.requires_grad = False
#             print("froze backbone parameters")

#         if cfg.MODEL.RPN.FREEZE:
#             for p in self.proposal_generator.parameters():
#                 p.requires_grad = False
#             print("froze proposal generator parameters")

#         if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
#             for p in self.roi_heads.res5.parameters():
#                 p.requires_grad = False
#             print("froze roi_box_head parameters")

#     def forward(self, batched_inputs):
#         if not self.training:
#             return self.inference(batched_inputs)
#         assert "instances" in batched_inputs[0]
#         gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
#         proposal_losses, detector_losses, _, _ = self._forward_once_(batched_inputs, gt_instances)
#         losses = {}
#         losses.update(detector_losses)
#         losses.update(proposal_losses)
#         return losses

#     def inference(self, batched_inputs):
#         assert not self.training
#         _, _, results, image_sizes = self._forward_once_(batched_inputs, None)
#         processed_results = []
#         for r, input, image_size in zip(results, batched_inputs, image_sizes):
#             height = input.get("height", image_size[0])
#             width = input.get("width", image_size[1])
#             r = detector_postprocess(r, height, width)
#             processed_results.append({"instances": r})
#         return processed_results

#     def _forward_once_(self, batched_inputs, gt_instances=None):

#         images = self.preprocess_image(batched_inputs)
#         features = self.backbone(images.tensor)

#         features_de_rpn = features
#         if self.cfg.MODEL.RPN.ENABLE_DECOUPLE:
#             scale = self.cfg.MODEL.RPN.BACKWARD_SCALE
#             features_de_rpn = {k: self.affine_rpn(decouple_layer(features[k], scale)) for k in features}
#         proposals, proposal_losses = self.proposal_generator(images, features_de_rpn, gt_instances)

#         features_de_rcnn = features
#         if self.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
#             scale = self.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE
#             features_de_rcnn = {k: self.affine_rcnn(decouple_layer(features[k], scale)) for k in features}
#         results, detector_losses = self.roi_heads(images, features_de_rcnn, proposals, gt_instances)

#         return proposal_losses, detector_losses, results, images.image_sizes

#     def preprocess_image(self, batched_inputs):
#         images = [x["image"].to(self.device) for x in batched_inputs]
#         images = [self.normalizer(x) for x in images]
#         images = ImageList.from_tensors(images, self.backbone.size_divisibility)
#         return images

#     def normalize_fn(self):
#         assert len(self.cfg.MODEL.PIXEL_MEAN) == len(self.cfg.MODEL.PIXEL_STD)
#         num_channels = len(self.cfg.MODEL.PIXEL_MEAN)
#         pixel_mean = (torch.Tensor(
#             self.cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1))
#         pixel_std = (torch.Tensor(
#             self.cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1))
#         return lambda x: (x - pixel_mean) / pixel_std


import torch
import logging
import numpy as np
import torch.nn.functional as F
from torch import nn
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from .build import META_ARCH_REGISTRY
from .gdl import decouple_layer, AffineLayer
from defrcn.modeling.roi_heads import build_roi_heads
from .cond_vae import CondResidualVAE

__all__ = ["GeneralizedRCNN"]

@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self._SHAPE_ = self.backbone.output_shape()
        self.proposal_generator = build_proposal_generator(cfg, self._SHAPE_)
        self.roi_heads = build_roi_heads(cfg, self._SHAPE_)
        self.normalizer = self.normalize_fn()
        self.affine_rpn = AffineLayer(num_channels=self._SHAPE_['res4'].channels, bias=True)
        self.affine_rcnn = AffineLayer(num_channels=self._SHAPE_['res4'].channels, bias=True)
        self.to(self.device)

        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("froze backbone parameters")

        if cfg.MODEL.RPN.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            print("froze proposal generator parameters")

        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for p in self.roi_heads.res5.parameters():
                p.requires_grad = False
            print("froze roi_box_head parameters")

        # =======================================================================
        # 2. INITIALIZE VAE FOR FEATURE AUGMENTATION
        # =======================================================================
        # Hardcoded paths - In production, add these to your cfg
        VAE_PATH = "checkpoints/defrcn_vae_model.pth" 
        CLIP_PATH = "data/roifeats_base/clip_prototypes_finetune.npy"
        
        # Load VAE Model
        self.vae_enabled = False
        try:
            checkpoint = torch.load(VAE_PATH, map_location=self.device)
            self.vae = CondResidualVAE().to(self.device)
            self.vae.load_state_dict(checkpoint['model_state_dict'])
            self.vae.eval() # Freeze VAE
            for p in self.vae.parameters():
                p.requires_grad = False
                
            # Load Stats (Mean/Std) for un-normalization
            self.feat_mean = checkpoint['feat_mean'].to(self.device)
            self.feat_std = checkpoint['feat_std'].to(self.device)
            
            # Load CLIP Prototypes
            self.clip_prototypes = torch.from_numpy(np.load(CLIP_PATH)).float().to(self.device)
            
            self.vae_enabled = True
            print(f"--> VAE Generator loaded successfully from {VAE_PATH}")
        except Exception as e:
            print(f"--> WARNING: Could not load VAE. Feature augmentation disabled. Error: {e}")

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)
        assert "instances" in batched_inputs[0]
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        
        proposal_losses, detector_losses, _, _ = self._forward_once_(batched_inputs, gt_instances)
        
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        
        if self.vae_enabled and self.training:
            # Generate features and get classification loss
            vae_cls_loss = self.get_vae_loss(gt_instances)
            # Add with a weight (usually 1.0 or 0.5, tunable)
            losses["loss_vae_cls"] = vae_cls_loss * 0.1 
            
        return losses

    def get_vae_loss(self, gt_instances):
        """
        Generates synthetic features for the classes present in the current batch
        and computes classification loss using the detector's box_predictor.
        """
        # 1. Identify unique classes in this batch
        all_gt_classes = torch.cat([x.gt_classes for x in gt_instances])
        unique_classes = torch.unique(all_gt_classes)
        
        # Filter out background or ignore classes if necessary
        valid_mask = (unique_classes >= 0) & (unique_classes < len(self.clip_prototypes))
        target_classes = unique_classes[valid_mask]
        
        if len(target_classes) == 0:
            return torch.tensor(0.0, device=self.device)

        # 2. Prepare Generation Inputs
        # Number of synthetic samples per class to generate
        num_samples = 30 
        
        # Repeat classes to form a batch
        # shape: [num_classes * num_samples]
        batch_classes = target_classes.repeat_interleave(num_samples)
        
        # Get Semantic Vectors (CLIP)
        # shape: [N, 512]
        semantic_vectors = self.clip_prototypes[batch_classes]
        
        # 3. Sample Z and Normalize (Equation 4 logic)
        # "Given a noise vector z, we generate... G( z/||z|| * beta, a^y )"
        with torch.no_grad():
            latent_dim = 512
            z = torch.randn(len(batch_classes), latent_dim, device=self.device)
            
            # Normalize z to unit sphere
            z_norm = z / (z.norm(dim=1, keepdim=True) + 1e-6)
            
            # Sample Beta (Control diversity)
            # Paper says "vary beta to obtain variations". We sample uniformly around 1.0
            # Range e.g., [0.8, 1.2]
            beta = (torch.rand(len(batch_classes), 1, device=self.device) * 0.4) + 0.8
            
            # Scale z
            z_scaled = z_norm * beta
            
            # Decode
            generated_feats = self.vae.decode(z_scaled, semantic_vectors)
            
            # Un-normalize features (Transform back to ResNet space)
            generated_feats = (generated_feats * self.feat_std) + self.feat_mean

        # 4. Compute Loss using the Detector's Classifier
        # We assume self.roi_heads has a 'box_predictor' (FastRCNNOutputLayers)
        # The predictor typically returns (scores, deltas)
        
        if hasattr(self.roi_heads, "box_predictor"):
            # Standard/Res5 ROI Heads
            scores, _ = self.roi_heads.box_predictor(generated_feats)
        elif hasattr(self.roi_heads, "box_head"):
            # Some implementations wrap it differently
            # Pass through box_head if needed, but usually VAE produces final feature
            scores, _ = self.roi_heads.box_predictor(generated_feats)
        else:
            # Fallback if structure is unknown (safeguard)
            return torch.tensor(0.0, device=self.device)
        
        # Compute Cross Entropy Loss
        # We only care about classification for these synthetic features
        loss_cls = F.cross_entropy(scores, batch_classes, reduction="mean")
        
        return loss_cls

    def inference(self, batched_inputs):
        assert not self.training
        _, _, results, image_sizes = self._forward_once_(batched_inputs, None)
        processed_results = []
        for r, input, image_size in zip(results, batched_inputs, image_sizes):
            height = input.get("height", image_size[0])
            width = input.get("width", image_size[1])
            r = detector_postprocess(r, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def _forward_once_(self, batched_inputs, gt_instances=None):

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        features_de_rpn = features
        if self.cfg.MODEL.RPN.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.RPN.BACKWARD_SCALE
            features_de_rpn = {k: self.affine_rpn(decouple_layer(features[k], scale)) for k in features}
        proposals, proposal_losses = self.proposal_generator(images, features_de_rpn, gt_instances)

        features_de_rcnn = features
        if self.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE
            features_de_rcnn = {k: self.affine_rcnn(decouple_layer(features[k], scale)) for k in features}
        results, detector_losses = self.roi_heads(images, features_de_rcnn, proposals, gt_instances)

        return proposal_losses, detector_losses, results, images.image_sizes

    def preprocess_image(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def normalize_fn(self):
        assert len(self.cfg.MODEL.PIXEL_MEAN) == len(self.cfg.MODEL.PIXEL_STD)
        num_channels = len(self.cfg.MODEL.PIXEL_MEAN)
        pixel_mean = (torch.Tensor(
            self.cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1))
        pixel_std = (torch.Tensor(
            self.cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1))
        return lambda x: (x - pixel_mean) / pixel_std


