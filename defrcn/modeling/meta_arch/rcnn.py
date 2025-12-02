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


# import torch
# import logging
# import numpy as np
# import torch.nn.functional as F
# from torch import nn
# from detectron2.structures import ImageList
# from detectron2.utils.logger import log_first_n
# from detectron2.modeling.backbone import build_backbone
# from detectron2.modeling.postprocessing import detector_postprocess
# from detectron2.modeling.proposal_generator import build_proposal_generator
# from .build import META_ARCH_REGISTRY
# from .gdl import decouple_layer, AffineLayer
# from defrcn.modeling.roi_heads import build_roi_heads
# from .cond_vae import CondResidualVAE

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

#         # =======================================================================
#         # 2. INITIALIZE VAE FOR FEATURE AUGMENTATION
#         # =======================================================================
#         # Hardcoded paths - In production, add these to your cfg
#         VAE_PATH = "checkpoints/defrcn_vae_model.pth" 
#         CLIP_PATH = "data/roifeats_base/clip_prototypes_finetune.npy"
        
#         # Load VAE Model
#         self.vae_enabled = False
#         try:
#             checkpoint = torch.load(VAE_PATH, map_location=self.device)
#             self.vae = CondResidualVAE().to(self.device)
#             self.vae.load_state_dict(checkpoint['model_state_dict'])
#             self.vae.eval() # Freeze VAE
#             for p in self.vae.parameters():
#                 p.requires_grad = False
                
#             # Load Stats (Mean/Std) for un-normalization
#             self.feat_mean = checkpoint['feat_mean'].to(self.device)
#             self.feat_std = checkpoint['feat_std'].to(self.device)
            
#             # Load CLIP Prototypes
#             self.clip_prototypes = torch.from_numpy(np.load(CLIP_PATH)).float().to(self.device)
            
#             self.vae_enabled = True
#             print(f"--> VAE Generator loaded successfully from {VAE_PATH}")
#         except Exception as e:
#             print(f"--> WARNING: Could not load VAE. Feature augmentation disabled. Error: {e}")

#     def forward(self, batched_inputs):
#         if not self.training:
#             return self.inference(batched_inputs)
#         assert "instances" in batched_inputs[0]
#         gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        
#         proposal_losses, detector_losses, _, _ = self._forward_once_(batched_inputs, gt_instances)
        
#         losses = {}
#         losses.update(detector_losses)
#         losses.update(proposal_losses)
        
#         if self.vae_enabled and self.training:
#             # Generate features and get classification loss
#             vae_cls_loss = self.get_vae_loss(gt_instances)
#             # Add with a weight (usually 1.0 or 0.5, tunable)
#             losses["loss_vae_cls"] = vae_cls_loss * 0.1 
            
#         return losses

#     def get_vae_loss(self, gt_instances):
#         """
#         Generates synthetic features for the classes present in the current batch
#         and computes classification loss using the detector's box_predictor.
#         """
#         # 1. Identify unique classes in this batch
#         all_gt_classes = torch.cat([x.gt_classes for x in gt_instances])
#         unique_classes = torch.unique(all_gt_classes)
        
#         # Filter out background or ignore classes if necessary
#         valid_mask = (unique_classes >= 0) & (unique_classes < len(self.clip_prototypes))
#         target_classes = unique_classes[valid_mask]
        
#         if len(target_classes) == 0:
#             return torch.tensor(0.0, device=self.device)

#         # 2. Prepare Generation Inputs
#         # Number of synthetic samples per class to generate
#         num_samples = 30 
        
#         # Repeat classes to form a batch
#         # shape: [num_classes * num_samples]
#         batch_classes = target_classes.repeat_interleave(num_samples)
        
#         # Get Semantic Vectors (CLIP)
#         # shape: [N, 512]
#         semantic_vectors = self.clip_prototypes[batch_classes]
        
#         # 3. Sample Z and Normalize (Equation 4 logic)
#         # "Given a noise vector z, we generate... G( z/||z|| * beta, a^y )"
#         with torch.no_grad():
#             latent_dim = 512
#             z = torch.randn(len(batch_classes), latent_dim, device=self.device)
            
#             # Normalize z to unit sphere
#             z_norm = z / (z.norm(dim=1, keepdim=True) + 1e-6)
            
#             # Sample Beta (Control diversity)
#             # Paper says "vary beta to obtain variations". We sample uniformly around 1.0
#             # Range e.g., [0.8, 1.2]
#             beta = (torch.rand(len(batch_classes), 1, device=self.device) * 0.4) + 0.8
            
#             # Scale z
#             z_scaled = z_norm * beta
            
#             # Decode
#             generated_feats = self.vae.decode(z_scaled, semantic_vectors)
            
#             # Un-normalize features (Transform back to ResNet space)
#             generated_feats = (generated_feats * self.feat_std) + self.feat_mean

#         # 4. Compute Loss using the Detector's Classifier
#         # We assume self.roi_heads has a 'box_predictor' (FastRCNNOutputLayers)
#         # The predictor typically returns (scores, deltas)
        
#         if hasattr(self.roi_heads, "box_predictor"):
#             # Standard/Res5 ROI Heads
#             scores, _ = self.roi_heads.box_predictor(generated_feats)
#         elif hasattr(self.roi_heads, "box_head"):
#             # Some implementations wrap it differently
#             # Pass through box_head if needed, but usually VAE produces final feature
#             scores, _ = self.roi_heads.box_predictor(generated_feats)
#         else:
#             # Fallback if structure is unknown (safeguard)
#             return torch.tensor(0.0, device=self.device)
        
#         # Compute Cross Entropy Loss
#         # We only care about classification for these synthetic features
#         loss_cls = F.cross_entropy(scores, batch_classes, reduction="mean")
        
#         return loss_cls

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
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn.functional as F
from torch import nn

from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.utils.events import get_event_storage
from .build import META_ARCH_REGISTRY
from .gdl import decouple_layer, AffineLayer
from defrcn.modeling.roi_heads import build_roi_heads


__all__ = ["GeneralizedRCNN"]


class CondResidualVAE(nn.Module):
    def __init__(self, resid_dim=2048, sem_dim=512, latent_dim=512, hidden_h=4096, leaky_slope=0.2):
        super().__init__()
        # Encoder
        self.enc_fcx = nn.Linear(resid_dim, hidden_h)
        self.enc_fcy = nn.Linear(sem_dim, hidden_h)
        self.enc_fc1 = nn.Linear(hidden_h*2, hidden_h)
        self.enc_fc2 = nn.Linear(hidden_h, hidden_h)
        self.enc_fc3 = nn.Linear(hidden_h, hidden_h)
        self.enc_mu = nn.Linear(hidden_h, latent_dim)
        self.enc_logvar = nn.Linear(hidden_h, latent_dim)
        self.enc_act = nn.LeakyReLU(leaky_slope, inplace=True)
        # Prior
        self.prior_fc1 = nn.Linear(sem_dim, hidden_h)
        self.prior_fc2 = nn.Linear(hidden_h, hidden_h)
        self.prior_mu = nn.Linear(hidden_h, latent_dim)
        self.prior_logvar = nn.Linear(hidden_h, latent_dim)
        self.prior_act = nn.LeakyReLU(leaky_slope, inplace=True)
        # Decoder
        self.dec_fcx = nn.Linear(latent_dim, hidden_h)
        self.dec_fcy = nn.Linear(sem_dim, hidden_h)
        self.dec_fc1 = nn.Linear(hidden_h*2, hidden_h)
        self.dec_out = nn.Linear(hidden_h, resid_dim)
        self.dec_hidden_act = nn.LeakyReLU(leaky_slope, inplace=True)
        self.dec_out_act = nn.Identity()

    def decode(self, z, sem):
        x = self.dec_hidden_act(self.dec_fcx(z))   
        y = self.dec_hidden_act(self.dec_fcy(sem))
        x = torch.cat([x,y], dim=1)
        x = self.dec_hidden_act(self.dec_fc1(x))
        x = self.dec_out(x)
        x = self.dec_out_act(x)
        return x

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

        # Freezing Logic
        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters(): p.requires_grad = False
        if cfg.MODEL.RPN.FREEZE:
            for p in self.proposal_generator.parameters(): p.requires_grad = False
        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for p in self.roi_heads.res5.parameters(): p.requires_grad = False

        # =======================================================================
        # 2. VAE & DIAGNOSTICS SETUP
        # =======================================================================
        VAE_PATH = "checkpoints/defrcn_vae_model.pth" 
        CLIP_PATH = "data/roifeats_base/clip_prototypes_finetune.npy"
        
        # Diagnostic Config
        self.vis_period = 500  # Run t-SNE every 500 steps
        self.vis_output_dir = cfg.OUTPUT_DIR # Save plots here
        
        # Load VAE
        self.vae_enabled = False
        try:
            checkpoint = torch.load(VAE_PATH, map_location=self.device)
            self.vae = CondResidualVAE().to(self.device)
            self.vae.load_state_dict(checkpoint['model_state_dict'])
            self.vae.eval() 
            for p in self.vae.parameters(): p.requires_grad = False
            
            self.feat_mean = checkpoint['feat_mean'].to(self.device)
            self.feat_std = checkpoint['feat_std'].to(self.device)
            
            # Load CLIP Prototypes
            self.clip_prototypes = torch.from_numpy(np.load(CLIP_PATH)).float().to(self.device)
            
            # TODO: CRITICAL FIX - ID MAPPING
            # If your dataset maps "Class 0" to "Person", but CLIP array index 0 is "Airplane",
            # you need a mapping array here.
            # Example: self.id_map = [5, 12, 40] (Dataset ID 0 -> CLIP ID 5)
            # For now, we assume Identity mapping, but YOU MUST CHECK THIS.
            self.id_map = None 
            
            self.vae_enabled = True
            print(f"--> VAE Generator loaded. Stats - Mean: {self.feat_mean.mean():.4f}, Std: {self.feat_std.mean():.4f}")
        except Exception as e:
            print(f"--> WARNING: Could not load VAE. Error: {e}")

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)
        
        assert "instances" in batched_inputs[0]
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        
        # 1. Standard Forward Pass
        proposal_losses, detector_losses, _, features_dict = self._forward_once_(batched_inputs, gt_instances)
        
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        
        # 2. VAE Loss Addition
        if self.vae_enabled and self.training:
            # Generate features and get classification loss
            vae_cls_loss = self.get_vae_loss(gt_instances)
            losses["loss_vae_cls"] = vae_cls_loss * 0.1 
            
            # 3. DIAGNOSTIC: Run Visualization periodically
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                 self.visualize_features(features_dict, gt_instances, step=storage.iter)
            
        return losses

    def get_vae_loss(self, gt_instances):
        """
        Generates synthetic features for the classes present in the current batch.
        """
        # Identify unique classes in this batch
        all_gt_classes = torch.cat([x.gt_classes for x in gt_instances])
        unique_classes = torch.unique(all_gt_classes)
        
        # Filter background classes (standard Detectron2 convention is usually last index)
        # Assuming CLIP prototypes cover only foreground classes
        if self.id_map is not None:
            # Map Dataset IDs to CLIP IDs
            # This handles the mismatch if Dataset 0 != CLIP 0
            # Ensure unique_classes are within map range
            valid_mask = unique_classes < len(self.id_map)
            target_classes_dataset = unique_classes[valid_mask]
            target_classes_clip = torch.tensor([self.id_map[c] for c in target_classes_dataset], device=self.device)
        else:
            # Identity mapping fallback (Direct index)
            valid_mask = (unique_classes >= 0) & (unique_classes < len(self.clip_prototypes))
            target_classes_dataset = unique_classes[valid_mask]
            target_classes_clip = target_classes_dataset

        if len(target_classes_dataset) == 0:
            return torch.tensor(0.0, device=self.device)

        # Prepare Generation Inputs
        num_samples = 30 
        
        # We need two lists:
        # 1. The CLIP IDs (to generate features)
        # 2. The Dataset IDs (to calculate Loss against the classifier)
        
        batch_clip_ids = target_classes_clip.repeat_interleave(num_samples)
        batch_dataset_ids = target_classes_dataset.repeat_interleave(num_samples)
        
        # Get Semantic Vectors (CLIP)
        semantic_vectors = self.clip_prototypes[batch_clip_ids]
        
        # Sample Z and Generate
        with torch.no_grad():
            latent_dim = 512
            z = torch.randn(len(batch_clip_ids), latent_dim, device=self.device)
            z_norm = z / (z.norm(dim=1, keepdim=True) + 1e-6)
            beta = (torch.rand(len(batch_clip_ids), 1, device=self.device) * 0.4) + 0.8
            z_scaled = z_norm * beta
            
            generated_feats = self.vae.decode(z_scaled, semantic_vectors)
            
            # Un-normalize features
            generated_feats = (generated_feats * self.feat_std) + self.feat_mean

        # Compute Loss using the Detector's Classifier
        if hasattr(self.roi_heads, "box_predictor"):
            scores, _ = self.roi_heads.box_predictor(generated_feats)
            
            # Calculate Cross Entropy
            # Important: We compare scores against batch_dataset_ids (the IDs the classifier knows)
            loss_cls = F.cross_entropy(scores, batch_dataset_ids, reduction="mean")
            return loss_cls
        
        return torch.tensor(0.0, device=self.device)

    def visualize_features(self, features_dict, gt_instances, step=0):
        """
        Extracts REAL features from the current batch and plots them against SYNTHETIC features.
        Handles both StandardROIHeads (FPN) and Res5ROIHeads (C4).
        """
        print(f"--> Generating Feature t-SNE for step {step}...")
        try:
            with torch.no_grad():
                # --- 1. Extract Real Features ---
                gt_boxes = [x.gt_boxes for x in gt_instances]
                gt_classes = torch.cat([x.gt_classes for x in gt_instances]).cpu().numpy()
                
                # Determine inputs based on Head Type
                if hasattr(self.roi_heads, "box_pooler"):
                    # Case A: StandardROIHeads (FPN)
                    # 1. Pool
                    box_features = self.roi_heads.box_pooler(
                        [features_dict[f] for f in self.roi_heads.in_features], 
                        gt_boxes
                    )
                    # 2. Head (MLP) -> Output is usually a vector [N, 1024]
                    real_feats = self.roi_heads.box_head(box_features)
                
                elif hasattr(self.roi_heads, "pooler"):
                    # Case B: Res5ROIHeads (C4 Backbone - Likely your case)
                    # 1. Pool
                    box_features = self.roi_heads.pooler(
                        [features_dict[f] for f in self.roi_heads.in_features], 
                        gt_boxes
                    )
                    # 2. Head (Res5 Block) -> Output is [N, 2048, 7, 7]
                    real_feats = self.roi_heads.res5(box_features)
                    
                    # 3. Global Average Pooling (CRITICAL for VAE comparison)
                    # Flatten spatial dims to get [N, 2048] vector
                    if len(real_feats.shape) == 4:
                        real_feats = real_feats.mean(dim=[2, 3])
                else:
                    print("--> Error: Unknown ROIHeads structure.")
                    return

                real_feats_np = real_feats.cpu().numpy()
                
                # --- 2. Generate Synthetic Features ---
                unique_cls = np.unique(gt_classes)
                # Filter background (assuming background is the highest index)
                # Adjust this check if your background logic differs
                if hasattr(self.roi_heads, "num_classes"):
                    unique_cls = unique_cls[unique_cls < self.roi_heads.num_classes]
                
                fake_feats_list = []
                fake_labels_list = []
                
                if len(unique_cls) == 0:
                    print("--> No foreground classes in batch to visualize.")
                    return

                for cls_id in unique_cls:
                    # Handle Mapping
                    clip_id = cls_id if self.id_map is None else self.id_map[cls_id]
                    
                    # Safety check for CLIP index
                    if clip_id >= len(self.clip_prototypes): continue

                    sem_vec = self.clip_prototypes[clip_id].unsqueeze(0).repeat(30, 1)
                    
                    # Generate
                    z = torch.randn(30, 512, device=self.device)
                    z = z / (z.norm(dim=1, keepdim=True) + 1e-6)
                    # Add mild beta variation
                    beta = (torch.rand(30, 1, device=self.device) * 0.4) + 0.8
                    gen = self.vae.decode(z * beta, sem_vec)
                    
                    # Un-normalize
                    gen = (gen * self.feat_std) + self.feat_mean
                    
                    fake_feats_list.append(gen.cpu().numpy())
                    fake_labels_list.append(np.full(30, cls_id))
                    
                if len(fake_feats_list) == 0: 
                    print("--> Could not generate valid fake features.")
                    return

                fake_feats_np = np.concatenate(fake_feats_list, axis=0)
                fake_labels_np = np.concatenate(fake_labels_list, axis=0)
                
                # --- 3. t-SNE & Plot ---
                all_feats = np.concatenate([real_feats_np, fake_feats_np], axis=0)
                
                # Use PCA init for stability
                tsne = TSNE(n_components=2, perplexity=min(30, len(all_feats)-1), init='pca', learning_rate='auto')
                emb = tsne.fit_transform(all_feats)
                
                real_emb = emb[:len(real_feats_np)]
                fake_emb = emb[len(real_feats_np):]
                
                plt.figure(figsize=(10, 8))
                cmap = plt.get_cmap('tab10')
                
                for i, cls in enumerate(unique_cls):
                    # Plot Real
                    idx_real = gt_classes == cls
                    if idx_real.sum() > 0:
                        plt.scatter(real_emb[idx_real, 0], real_emb[idx_real, 1], 
                                    color=cmap(i % 10), marker='o', alpha=0.6, label=f'Real {cls}')
                    
                    # Plot Fake
                    idx_fake = fake_labels_np == cls
                    if idx_fake.sum() > 0:
                        plt.scatter(fake_emb[idx_fake, 0], fake_emb[idx_fake, 1], 
                                    color=cmap(i % 10), marker='x', s=100, label=f'Fake {cls}')
                                
                plt.legend()
                plt.title(f"Feature Distribution Step {step}")
                save_path = os.path.join(self.vis_output_dir, f"tsne_step_{step}.png")
                plt.savefig(save_path)
                plt.close()
                print(f"--> Saved t-SNE visualization to {save_path}")
                
        except Exception as e:
            # Print full trace for debugging
            import traceback
            traceback.print_exc()
            print(f"--> Error in visualization: {e}")

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
        
        # Return features_de_rcnn so we can use them in visualization/loss without re-computing
        return proposal_losses, detector_losses, results, features_de_rcnn

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


