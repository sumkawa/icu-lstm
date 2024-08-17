import torch
import random
import numpy as np


class CFG:
    verbose = 1  # Verbosity
    seed = 35  # Random seed
    preset = "efficientnetv2_b2_imagenet"  # Name of pretrained classifier
    image_size = [400, 300]  # Input image size
    epochs = 13  # Training epochs
    batch_size = 64  # Batch size
    lr_mode = "cos"  # LR scheduler mode from one of "cos", "step", "exp"
    drop_remainder = True  # Drop incomplete batches
    num_classes = 6  # Number of classes in the dataset
    fold = 0  # Which fold to set as validation data
    class_names = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']
    label2name = dict(enumerate(class_names))
    name2label = {v: k for k, v in label2name.items()}


# Set seeds for reproducibility
torch.manual_seed(CFG.seed)
random.seed(CFG.seed)
np.random.seed(CFG.seed)

BASE_PATH = "../data"
SPEC_DIR = "./tmp/dataset/hms-hbac"
NPY_SAVE_DIR = "/data/npy/dataset/hms-hbac"
