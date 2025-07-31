import random
from enum import Enum

SEED = 42
NUM_CLASSES = 4 #4 for Alzheimer | 8 for Skin_Lesions
NUM_WORKERS = 4
AUGMENT = True     #<--- change if data augmentation isn't needed
HEIGHT_WIDTH = (224, 224)
RESIZE_IMAGE = (250, 250)
BRIGHTNESS, CONTRAST, HUE, SATURATION, FLIP = (0.1, 0.1, 0.1, 0.1, 0.2)
NORMALIZE_MEAN, NORMALIZE_STD = (0.5, 0.5)

# Albumentations Augmentation
A_RANDOM_SCALE = 0.07
A_ROTATE_LIMIT = 50
A_FLIP_PROB = 0.2
A_SHEAR = 0.1
A_BRIGHTNESS_CONTRAST = dict(brightness_limit=0.15, contrast_limit=0.1)
A_COARSE_MAX_HOLES = 8   
A_COARSE_HEIGHT = 16     
A_COARSE_WIDTH = 16
A_COARSE_P = 0.25

# Weights & Bais
WANDB_PROJECT = "PrivacyBench"
WANDB_ENTITY = "MLC-FedML"


class ExperimentName(str, Enum):
    # Baseline
    CNN_BASE = "CNN Baseline"
    VIT_BASE = "ViT Baseline"
    # FL + CNN & ViT
    FL_CNN = "FL (CNN)"
    FL_VIT = "FL (ViT)"
    DP_CNN = "DP (CNN)"
    DP_VIT = "DP (ViT)"
    # FL + SMPC
    FL_SMPC_CNN = "FL + SMPC (CNN)"
    FL_SMPC_VIT = "FL + SMPC (ViT)"
    # FL + DP Variants (Central & Local DP)
    FL_CDP_SF_CNN = "FL + CDP-SF (CNN)"
    FL_CDP_SF_VIT = "FL + CDP-SF (ViT)"
    FL_CDP_SA_CNN = "FL + CDP-SA (CNN)"
    FL_CDP_SA_VIT = "FL + CDP-SA (ViT)"
    FL_CDP_CF_CNN = "FL + CDP-CF (CNN)"
    FL_CDP_CF_VIT = "FL + CDP-CF (ViT)"
    FL_CDP_CA_CNN = "FL + CDP-CA (CNN)"
    FL_CDP_CA_VIT = "FL + CDP-CA (ViT)"
    FL_LDP_MOD_CNN = "FL + LDP-Mod (CNN)"
    FL_LDP_MOD_VIT = "FL + LDP-Mod (ViT)"
    FL_LDP_PE_CNN = "FL + LDP-PE (CNN)"
    FL_LDP_PE_VIT = "FL + LDP-PE (ViT)"
    # FL + SMPC + DP
    