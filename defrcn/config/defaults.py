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


# ---------------------------------------------------------------------------- #
#  NSH Config
# ---------------------------------------------------------------------------- #
_CC.MODEL.NSH = CN()
_CC.MODEL.NSH.ENABLED = True       # Master switch

# ---- Base / Novel Training Control ---- #
_CC.MODEL.NSH.FHM_TRAIN = True     # True = Base training, False = Novel ft

# ---- FHM Efficiency Options ---- #
_CC.MODEL.NSH.FHM_SAMPLE_RATIO = 0.25     # Use 25% of RoIs for hallucination
_CC.MODEL.NSH.FHM_MAX_SAMPLES = 128       # Max number of hallucination RoIs
_CC.MODEL.NSH.FHM_SPATIAL_RED = 1         # Spatial downsample (1 = none)
_CC.MODEL.NSH.FHM_HIDDEN = 256            # Hidden channels in FHM

# ---- SVA / CLIP ---- #
_CC.MODEL.NSH.CLIP_MODEL = "ViT-B/32"
_CC.MODEL.NSH.CLIP_DIM = 512
_CC.MODEL.NSH.CLIP_TAU = 0.07
_CC.MODEL.NSH.CLASS_NAMES = ["aeroplane", "bicycle", "boat", "bottle", "car",
        "cat", "chair", "diningtable", "dog", "horse",
        "person", "pottedplant", "sheep", "train", "tvmonitor",
    ]            # user should fill dataset class names

# ---- Loss Weights ---- #
_CC.MODEL.NSH.LAMBDA_REC = 1.0
_CC.MODEL.NSH.LAMBDA_SVA = 0.1
_CC.MODEL.NSH.LAMBDA_SP  = 0.01

# ---------------------------------------------------------------------------- #
#  Solver
# ---------------------------------------------------------------------------- #
_CC.SOLVER.WEIGHT_DECAY = 5e-5

_CC.MUTE_HEADER = True


