from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN

_CC = _C

# ----------- Backbone ----------- #
_CC.MODEL.BACKBONE.FREEZE = False
_CC.MODEL.BACKBONE.FREEZE_AT = 3

# ------------- RPN -------------- #
_CC.MODEL.RPN.FREEZE = False
_CC.MODEL.RPN.ENABLE_DECOUPLE = False
_CC.MODEL.RPN.BACKWARD_SCALE = 1.0

# ------------- ROI -------------- #
_CC.MODEL.ROI_HEADS.NAME = "Res5ROIHeads"
_CC.MODEL.ROI_HEADS.FREEZE_FEAT = False
_CC.MODEL.ROI_HEADS.ENABLE_DECOUPLE = False
_CC.MODEL.ROI_HEADS.BACKWARD_SCALE = 1.0
_CC.MODEL.ROI_HEADS.OUTPUT_LAYER = "FastRCNNOutputLayers"
_CC.MODEL.ROI_HEADS.CLS_DROPOUT = False
_CC.MODEL.ROI_HEADS.DROPOUT_RATIO = 0.8
_CC.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7  # for faster
_CC.MODEL.ROI_HEADS.NUM_CLASSES = 20


# ------------- TEST ------------- #
_CC.TEST.PCB_ENABLE = False
_CC.TEST.PCB_MODELTYPE = 'resnet'             # res-like
_CC.TEST.PCB_MODELPATH = ""
_CC.TEST.PCB_ALPHA = 0.50
_CC.TEST.PCB_UPPER = 1.0
_CC.TEST.PCB_LOWER = 0.05
_CC.TEST.PCB_REPULSION = False

# ------------ Other ------------- #
_CC.SOLVER.WEIGHT_DECAY = 5e-5
_CC.MUTE_HEADER = True

_CC.MODEL.ROI_BOX_HEAD = CN()

_CC.MODEL.ROI_BOX_HEAD.NAME = ""
# Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets
# These are empirically chosen to approximately lead to unit variance targets
_CC.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
# The transition point from L1 to L2 loss. Set to 0.0 to make the loss simply L1.
_CC.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA = 0.0
_CC.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 14
# set sampling ratio to 0 to sample densely
_CC.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 0
# Type of pooling operation applied to the incoming feature map for each RoI
_CC.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignV2"

_CC.MODEL.ROI_BOX_HEAD.NUM_FC = 0
# Hidden layer dimension for FC layers in the RoI box head
_CC.MODEL.ROI_BOX_HEAD.FC_DIM = 1024
_CC.MODEL.ROI_BOX_HEAD.NUM_CONV = 0
# Channel dimension for Conv layers in the RoI box head
_CC.MODEL.ROI_BOX_HEAD.CONV_DIM = 256
# Normalization method for the convolution layers.
# Options: "" (no norm), "GN", "SyncBN".
_CC.MODEL.ROI_BOX_HEAD.NORM = ""
# Whether to use class agnostic for bbox regression
_CC.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = False

_CC.MODEL.ROI_BOX_HEAD.BOX_REG_WEIGHT = 1.0
_CC.MODEL.ROI_BOX_HEAD.BOX_CLS_WEIGHT = 1.0


_CC.MODEL.ROI_BOX_HEAD.SUB_FC_DIM = 1024


# perform Supervised Contrastive Loss within batch
_CC.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH = CN({'ENABLED': False})  # to be used to enable contrastive loss for double head
_CC.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.MLP_FEATURE_DIM = 128
_CC.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.TEMPERATURE = 0.1
_CC.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.LOSS_WEIGHT = 1.0
_CC.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY = CN({'ENABLED': False})
_CC.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY.STEPS = [8000, 16000]
_CC.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY.RATE = 0.2
_CC.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.IOU_THRESHOLD = 0.5
_CC.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.LOSS_VERSION = 'V1'
_CC.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.REWEIGHT_FUNC = 'none'
