# import os
# import cv2
# import json
# import torch
# import logging
# import detectron2
# import numpy as np
# from detectron2.structures import ImageList
# from detectron2.modeling.poolers import ROIPooler
# from sklearn.metrics.pairwise import cosine_similarity
# from defrcn.dataloader import build_detection_test_loader
# from defrcn.evaluation.archs import resnet101

# logger = logging.getLogger(__name__)


# class PrototypicalCalibrationBlock:

#     def __init__(self, cfg):
#         super().__init__()
#         self.cfg = cfg
#         self.device = torch.device(cfg.MODEL.DEVICE)
#         self.alpha = self.cfg.TEST.PCB_ALPHA

#         self.imagenet_model = self.build_model()
#         self.dataloader = build_detection_test_loader(self.cfg, self.cfg.DATASETS.TRAIN[0])
#         self.roi_pooler = ROIPooler(output_size=(1, 1), scales=(1 / 32,), sampling_ratio=(0), pooler_type="ROIAlignV2")
#         self.prototypes = self.build_prototypes()

#         self.exclude_cls = self.clsid_filter()

#     def build_model(self):
#         logger.info("Loading ImageNet Pre-train Model from {}".format(self.cfg.TEST.PCB_MODELPATH))
#         if self.cfg.TEST.PCB_MODELTYPE == 'resnet':
#             imagenet_model = resnet101()
#         else:
#             raise NotImplementedError
#         state_dict = torch.load(self.cfg.TEST.PCB_MODELPATH)
#         imagenet_model.load_state_dict(state_dict)
#         imagenet_model = imagenet_model.to(self.device)
#         imagenet_model.eval()
#         return imagenet_model

#     def build_prototypes(self):

#         all_features, all_labels = [], []
#         for index in range(len(self.dataloader.dataset)):
#             inputs = [self.dataloader.dataset[index]]
#             assert len(inputs) == 1
#             # load support images and gt-boxes
#             img = cv2.imread(inputs[0]['file_name'])  # BGR
#             img_h, img_w = img.shape[0], img.shape[1]
#             ratio = img_h / inputs[0]['instances'].image_size[0]
#             inputs[0]['instances'].gt_boxes.tensor = inputs[0]['instances'].gt_boxes.tensor * ratio
#             boxes = [x["instances"].gt_boxes.to(self.device) for x in inputs]

#             # extract roi features
#             features = self.extract_roi_features(img, boxes)
#             all_features.append(features.cpu().data)

#             gt_classes = [x['instances'].gt_classes for x in inputs]
#             all_labels.append(gt_classes[0].cpu().data)

#         # concat
#         all_features = torch.cat(all_features, dim=0)
#         all_labels = torch.cat(all_labels, dim=0)
#         assert all_features.shape[0] == all_labels.shape[0]

#         # calculate prototype
#         features_dict = {}
#         for i, label in enumerate(all_labels):
#             label = int(label)
#             if label not in features_dict:
#                 features_dict[label] = []
#             features_dict[label].append(all_features[i].unsqueeze(0))

#         prototypes_dict = {}
#         for label in features_dict:
#             features = torch.cat(features_dict[label], dim=0)
#             prototypes_dict[label] = torch.mean(features, dim=0, keepdim=True)

#         return prototypes_dict

#     def extract_roi_features(self, img, boxes):
#         """
#         :param img:
#         :param boxes:
#         :return:
#         """

#         mean = torch.tensor([0.406, 0.456, 0.485]).reshape((3, 1, 1)).to(self.device)
#         std = torch.tensor([[0.225, 0.224, 0.229]]).reshape((3, 1, 1)).to(self.device)

#         img = img.transpose((2, 0, 1))
#         img = torch.from_numpy(img).to(self.device)
#         images = [(img / 255. - mean) / std]
#         images = ImageList.from_tensors(images, 0)
#         conv_feature = self.imagenet_model(images.tensor[:, [2, 1, 0]])[1]  # size: BxCxHxW

#         box_features = self.roi_pooler([conv_feature], boxes).squeeze(2).squeeze(2)

#         activation_vectors = self.imagenet_model.fc(box_features)

#         return activation_vectors

#     def execute_calibration(self, inputs, dts):

#         img = cv2.imread(inputs[0]['file_name'])

#         ileft = (dts[0]['instances'].scores > self.cfg.TEST.PCB_UPPER).sum()
#         iright = (dts[0]['instances'].scores > self.cfg.TEST.PCB_LOWER).sum()
#         assert ileft <= iright
#         boxes = [dts[0]['instances'].pred_boxes[ileft:iright]]

#         features = self.extract_roi_features(img, boxes)

#         for i in range(ileft, iright):
#             tmp_class = int(dts[0]['instances'].pred_classes[i])
#             if tmp_class in self.exclude_cls:
#                 continue
#             tmp_cos = cosine_similarity(features[i - ileft].cpu().data.numpy().reshape((1, -1)),
#                                         self.prototypes[tmp_class].cpu().data.numpy())[0][0]
#             dts[0]['instances'].scores[i] = dts[0]['instances'].scores[i] * self.alpha + tmp_cos * (1 - self.alpha)
#         return dts

#     def clsid_filter(self):
#         dsname = self.cfg.DATASETS.TEST[0]
#         exclude_ids = []
#         if 'test_all' in dsname:
#             if 'coco' in dsname:
#                 exclude_ids = [7, 9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
#                                30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45,
#                                46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 59, 61, 63, 64, 65,
#                                66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
#             elif 'voc' in dsname:
#                 exclude_ids = list(range(0, 15))
#             else:
#                 raise NotImplementedError
#         return exclude_ids


# @torch.no_grad()
# def concat_all_gather(tensor):
#     """
#     Performs all_gather operation on the provided tensors.
#     *** Warning ***: torch.distributed.all_gather has no gradient.
#     """
#     tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
#     torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
#     output = torch.cat(tensors_gather, dim=0)
#     return output


# modified_prototypical_calibration_block.py
import os
import cv2
import json
import torch
import logging
import detectron2
import numpy as np
import matplotlib.pyplot as plt

from detectron2.structures import ImageList
from detectron2.modeling.poolers import ROIPooler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from defrcn.dataloader import build_detection_test_loader
from defrcn.evaluation.archs import resnet101

logger = logging.getLogger(__name__)


class PrototypicalCalibrationBlock:

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.alpha = float(self.cfg.TEST.PCB_ALPHA)

        self.imagenet_model = self.build_model()
        self.dataloader = build_detection_test_loader(self.cfg, self.cfg.DATASETS.TRAIN[0])
        # keep ROIAlignV2 configuration same as original
        self.roi_pooler = ROIPooler(output_size=(1, 1),
                                    scales=(1 / 32,),
                                    sampling_ratio=(0),
                                    pooler_type="ROIAlignV2")

        # placeholders filled in build_prototypes()
        self.prototypes = {}         # dict: class_id -> tensor(1, C)
        self._feature_bank = None    # tensor: N x C (all instance features)
        self._label_bank = None      # tensor: N (corresponding labels)

        # build prototypes and feature bank
        self.build_prototypes()

        self.exclude_cls = self.clsid_filter()

    def build_model(self):
        logger.info("Loading ImageNet Pre-train Model from {}".format(self.cfg.TEST.PCB_MODELPATH))
        if self.cfg.TEST.PCB_MODELTYPE == 'resnet':
            imagenet_model = resnet101()
        else:
            raise NotImplementedError
        state_dict = torch.load(self.cfg.TEST.PCB_MODELPATH, map_location="cpu")
        imagenet_model.load_state_dict(state_dict)
        imagenet_model = imagenet_model.to(self.device)
        imagenet_model.eval()
        return imagenet_model

    def build_prototypes(self):
        """
        Builds:
         - self._feature_bank : FloatTensor [N, D] all roi features from support set
         - self._label_bank   : LongTensor  [N]   corresponding labels
         - self.prototypes    : dict {class_id: tensor([1, D])} mean feature per class
        """
        all_features, all_labels = [], []
        logger.info("Building prototypes from dataloader with {} samples".format(len(self.dataloader.dataset)))

        for index in range(len(self.dataloader.dataset)):
            inputs = [self.dataloader.dataset[index]]
            assert len(inputs) == 1
            # load support images and gt-boxes
            img = cv2.imread(inputs[0]['file_name'])  # BGR
            if img is None:
                logger.warning(f"Could not read image: {inputs[0]['file_name']}")
                continue
            img_h, img_w = img.shape[0], img.shape[1]
            # scale boxes to current loaded image size
            ratio = img_h / inputs[0]['instances'].image_size[0]
            inputs[0]['instances'].gt_boxes.tensor = inputs[0]['instances'].gt_boxes.tensor * ratio
            boxes = [x["instances"].gt_boxes.to(self.device) for x in inputs]

            # extract roi features
            features = self.extract_roi_features(img, boxes)   # shape: num_rois x D
            all_features.append(features.cpu().detach())

            gt_classes = [x['instances'].gt_classes for x in inputs]
            all_labels.append(gt_classes[0].cpu().detach())

        if len(all_features) == 0:
            raise RuntimeError("No features collected from dataloader (empty). Check dataset paths.")

        # concat
        all_features = torch.cat(all_features, dim=0)   # [N, D]
        all_labels = torch.cat(all_labels, dim=0)       # [N]
        assert all_features.shape[0] == all_labels.shape[0]

        # save banks
        self._feature_bank = all_features
        self._label_bank = all_labels

        # calculate prototypes (mean per class)
        features_dict = {}
        for i, label in enumerate(all_labels):
            label = int(label)
            if label not in features_dict:
                features_dict[label] = []
            features_dict[label].append(all_features[i].unsqueeze(0))

        prototypes_dict = {}
        for label in features_dict:
            features = torch.cat(features_dict[label], dim=0)  # [M, D]
            prototypes_dict[label] = torch.mean(features, dim=0, keepdim=True)  # [1, D]

        self.prototypes = prototypes_dict
        logger.info("Built {} prototypes".format(len(self.prototypes)))
        return prototypes_dict

    def extract_roi_features(self, img, boxes):
        """
        :param img: numpy BGR HxWxC
        :param boxes: list of Boxes tensors on device, length batch=1
        :return: activation_vectors: torch.Tensor [num_rois, D]
        """
        mean = torch.tensor([0.406, 0.456, 0.485]).reshape((3, 1, 1)).to(self.device)
        std = torch.tensor([[0.225, 0.224, 0.229]]).reshape((3, 1, 1)).to(self.device)

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).to(self.device)
        images = [(img / 255. - mean) / std]
        images = ImageList.from_tensors(images, 0)
        # imagenet_model returns a tuple, the conv features at index 1 in original code
        conv_feature = self.imagenet_model(images.tensor[:, [2, 1, 0]])[1]  # size: BxCxHxW

        # roi_pooler expects list[feature_map], list[Boxes]
        box_features = self.roi_pooler([conv_feature], boxes).squeeze(2).squeeze(2)  # [num_rois, C]
        activation_vectors = self.imagenet_model.fc(box_features)  # forward fc to get embeddings

        return activation_vectors

    def execute_calibration(self, inputs, dts):
        img = cv2.imread(inputs[0]['file_name'])

        ileft = (dts[0]['instances'].scores > self.cfg.TEST.PCB_UPPER).sum()
        iright = (dts[0]['instances'].scores > self.cfg.TEST.PCB_LOWER).sum()
        assert ileft <= iright
        boxes = [dts[0]['instances'].pred_boxes[ileft:iright]]

        features = self.extract_roi_features(img, boxes)

        for i in range(ileft, iright):
            tmp_class = int(dts[0]['instances'].pred_classes[i])
            if tmp_class in self.exclude_cls:
                continue
            tmp_cos = cosine_similarity(features[i - ileft].cpu().data.numpy().reshape((1, -1)),
                                        self.prototypes[tmp_class].cpu().data.numpy())[0][0]
            dts[0]['instances'].scores[i] = dts[0]['instances'].scores[i] * self.alpha + tmp_cos * (1 - self.alpha)
        return dts

    def clsid_filter(self):
        dsname = self.cfg.DATASETS.TEST[0]
        exclude_ids = []
        if 'test_all' in dsname:
            if 'coco' in dsname:
                exclude_ids = [7, 9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                               30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45,
                               46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 59, 61, 63, 64, 65,
                               66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
            elif 'voc' in dsname:
                exclude_ids = list(range(0, 15))
            else:
                raise NotImplementedError
        return exclude_ids

    # ----------------- New utilities for visualization -----------------
    def _prepare_embeddings(self, vectors, n_components_pca=0):
        """
        Optionally reduce dimensionality with PCA before t-SNE to speed up.
        vectors: np.ndarray [N, D]
        returns: np.ndarray [N, D']
        This version safely clamps n_components_pca to valid range.
        """
        n_samples, n_features = vectors.shape
        if not n_components_pca or n_samples < 2:
            # nothing to do: can't PCA with <2 samples
            return vectors

        # max allowed components is min(n_samples, n_features)
        max_allowed = min(n_samples, n_features)
        n_components = int(min(n_components_pca, max_allowed))
        if n_components <= 0:
            return vectors

        # if n_components equals original features, no reduction needed
        if n_components == n_features:
            return vectors

        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        return pca.fit_transform(vectors)

    def visualize_prototypes_tsne(self, out_path="prototypes_tsne.png",
                                  perplexity=30, n_iter=1000, pca_components=50, random_state=42):
        """
        Safe t-SNE for prototypes. Handles small sample counts by using PCA-2D.
        """
        if not self.prototypes or len(self.prototypes) == 0:
            raise RuntimeError("No prototypes available. Ensure build_prototypes() was called.")

        class_ids = sorted(self.prototypes.keys())
        prot_list = [self.prototypes[c].cpu().numpy().reshape(-1) for c in class_ids]
        X = np.stack(prot_list, axis=0)  # [num_classes, D]
        n_samples = X.shape[0]

        # If too few samples for t-SNE, do PCA->2D instead
        if n_samples < 3:
            # fallback: PCA to 2D
            if X.shape[1] >= 2:
                from sklearn.decomposition import PCA
                emb = PCA(n_components=2).fit_transform(X)
            else:
                # 1D features: expand dims
                emb = np.zeros((n_samples, 2))
                emb[:, 0] = X.reshape(n_samples)
        else:
            # safe pca pre-reduction
            X_prep = self._prepare_embeddings(X, n_components_pca=pca_components)

            # ensure perplexity < n_samples
            safe_perplex = int(min(perplexity, max(1, n_samples - 1)))
            safe_perplex = max(1, safe_perplex)

            tsne = TSNE(n_components=2,
                        perplexity=safe_perplex,
                        n_iter=n_iter,
                        random_state=random_state,
                        init="pca")
            emb = tsne.fit_transform(X_prep)

        # plotting
        plt.figure(figsize=(10, 8))
        cmap = plt.get_cmap("tab20")
        for i, cls in enumerate(class_ids):
            plt.scatter(emb[i, 0], emb[i, 1], marker='X', s=120, label=str(cls), alpha=0.9, color=cmap(i % 20))
            plt.text(emb[i, 0] + 1e-3, emb[i, 1] + 1e-3, str(cls), fontsize=9)

        plt.title("Prototypes ({} classes)".format(len(class_ids)))
        plt.xlabel("dim-1")
        plt.ylabel("dim-2")
        plt.legend(title="class id", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        plt.savefig(out_path, dpi=200)
        plt.close()
        logger.info("Saved prototype t-SNE/PCA to {}".format(out_path))
        return out_path

    def visualize_featurebank_tsne(self, out_path="featurebank_tsne.png", perplexity=50, n_iter=1000,
                                  pca_components=50, max_points=2000, random_state=42):
        """
        Safe visualization of the feature bank (with prototype overlay).
        Handles PCA clamping and safe perplexity. If combined sample count is too small for t-SNE,
        fall back to PCA-2D.
        """
        if self._feature_bank is None or self._label_bank is None:
            raise RuntimeError("Feature bank empty. Ensure build_prototypes() was called.")

        X = self._feature_bank.numpy()  # [N, D]
        y = self._label_bank.numpy()    # [N]

        # subsample balanced (per class) if too many
        unique = np.unique(y)
        if X.shape[0] > max_points:
            per_class = max(1, max_points // len(unique))
            idxs = []
            for cls in unique:
                cls_idx = np.where(y == cls)[0]
                if len(cls_idx) == 0:
                    continue
                if len(cls_idx) > per_class:
                    chosen = np.random.RandomState(seed=int(cls) + 123).choice(cls_idx, per_class, replace=False)
                else:
                    chosen = cls_idx
                idxs.append(chosen)
            idxs = np.concatenate(idxs, axis=0)
            X = X[idxs]
            y = y[idxs]

        # prepare combined (instances + prototypes)
        prot_ids = sorted(self.prototypes.keys())
        prot_list = [self.prototypes[c].cpu().numpy().reshape(-1) for c in prot_ids]
        prot_arr = np.stack(prot_list, axis=0)
        combined = np.concatenate([X, prot_arr], axis=0)

        n_combined = combined.shape[0]

        # If too few samples for t-SNE, do PCA to 2D on combined
        if n_combined < 3:
            if combined.shape[1] >= 2:
                from sklearn.decomposition import PCA
                combined_emb = PCA(n_components=2).fit_transform(combined)
            else:
                combined_emb = np.zeros((n_combined, 2))
                combined_emb[:, 0] = combined.reshape(n_combined)
            X_emb = combined_emb[:X.shape[0]]
            prot_emb = combined_emb[X.shape[0]:]
        else:
            # safe PCA pre-reduction
            combined_prep = self._prepare_embeddings(combined, n_components_pca=pca_components)

            # ensure perplexity < n_samples
            safe_perplex = int(min(perplexity, max(1, combined_prep.shape[0] - 1)))
            safe_perplex = max(1, safe_perplex)

            tsne2 = TSNE(n_components=2,
                         perplexity=safe_perplex,
                         n_iter=n_iter,
                         random_state=random_state,
                         init="pca")
            combined_emb = tsne2.fit_transform(combined_prep)

            X_emb = combined_emb[:X.shape[0]]
            prot_emb = combined_emb[X.shape[0]:]

        # plotting
        plt.figure(figsize=(12, 10))
        cmap = plt.get_cmap("tab20")
        unique_sorted = np.unique(y)
        for i, cls in enumerate(unique_sorted):
            cls_mask = (y == cls)
            plt.scatter(X_emb[cls_mask, 0], X_emb[cls_mask, 1], s=10, alpha=0.6, label=str(cls), color=cmap(i % 20))

        for i, cls in enumerate(prot_ids):
            if cls in unique_sorted:
                color = cmap(np.where(unique_sorted == cls)[0][0] % 20)
            else:
                color = cmap(i % 20)
            plt.scatter(prot_emb[i, 0], prot_emb[i, 1], marker='X', s=200, edgecolors='k', linewidths=1.2, label=f"proto_{cls}", color=color)
            plt.text(prot_emb[i, 0] + 1e-3, prot_emb[i, 1] + 1e-3, str(cls), fontsize=9, fontweight='bold')

        plt.title("Feature Bank (points) + Prototypes (X)")
        plt.xlabel("dim-1")
        plt.ylabel("dim-2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
        plt.tight_layout()
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        plt.savefig(out_path, dpi=200)
        plt.close()
        logger.info("Saved featurebank t-SNE/PCA to {}".format(out_path))
        return out_path