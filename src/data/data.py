import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import torch
import cv2
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import joblib

import random

import matplotlib.pyplot as plt

class EEGDataset(Dataset):
    def __init__(self, df, data_dir, mode="train", transform=None, augment=None):
        self.df = df
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        self.augment = augment

        self.spec_paths = self.df['spec2_path'].values
        if mode != "test":
            self.labels = self.df['class_label'].values

    def __len__(self):
        return len(self.df)

    def preprocess(self, spectrogram):
        """
        Apply log transformation, normalization, and enforce a consistent size.
        """
        # Clip the spectrogram to [400, 300] or pad it accordingly
        target_width = 300
        if spectrogram.shape[1] > target_width:
            spectrogram = spectrogram[:, :target_width]  # Trim to the target width
        else:
            pad_width = target_width - spectrogram.shape[1]
            spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant')
    
        # Log transform to enhance smaller values and reduce large outliers
        spectrogram = np.clip(spectrogram, a_min=np.exp(-4.0), a_max=np.exp(8.0))  # avoid log(0)
        spectrogram = np.log(spectrogram)
        
        # Normalize to zero mean and unit variance
        spectrogram -= np.mean(spectrogram)
        spectrogram /= (np.std(spectrogram) + 1e-6)

        # Stack grayscale image into 3 channels for compatibility with ImageNet models
        spectrogram = np.stack([spectrogram] * 3, axis=-1)  # Shape will be (H, W, 3)
    
        return spectrogram

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        spec_path = self.spec_paths[idx]
        spectrogram = np.load(spec_path)
    
        # Apply preprocessing
        spectrogram = self.preprocess(spectrogram)
    
        # Apply any transformation if provided (e.g., resizing, augmentation)
        if self.transform:
            spectrogram = self.transform(spectrogram)
    
        # Convert to tensor if not already a tensor
        if not isinstance(spectrogram, torch.Tensor):
            spectrogram = torch.tensor(spectrogram, dtype=torch.float32).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    
        # Apply augmentation if provided
        if self.augment and self.mode == "train":
            spectrogram = self.augment(spectrogram)
        
        if self.mode != "test":
            # One-hot encode labels
            label = F.one_hot(torch.tensor(self.labels[idx]), num_classes=len(self.df['class_label'].unique()))
            label = label.float()  # Convert to float for compatibility with certain losses like KL Divergence
            return spectrogram, label
        else:
            return spectrogram
