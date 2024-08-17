import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datasets.ds_1 import EEGDataset
from configs.config_1 import CFG, SPEC_DIR, NPY_SAVE_DIR
import pandas as pd

# Define transformations and augmentations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((400, 300))  # Resizing to a fixed size (height, width)
])

augment = transforms.Compose([
    transforms.RandomErasing(
        p=0.5,
        scale=(0.02, 0.1),
        ratio=(8, 12),  # Frequency mask
        inplace=False),
    transforms.RandomErasing(
        p=0.5,
        scale=(0.02, 0.10),
        ratio=(0.5, 1.5),  # Time mask
        inplace=False)
])

# Load data
df = pd.read_csv(f'{CFG.BASE_PATH}/train.csv')
test_df = pd.read_csv(f'{CFG.BASE_PATH}/test.csv')

# Create datasets
train_dataset = EEGDataset(df=df,
                           data_dir=f"{NPY_SAVE_DIR}/train_spectrograms",
                           mode="train",
                           transform=transform,
                           augment=augment)
test_dataset = EEGDataset(df=test_df,
                          data_dir=f"{NPY_SAVE_DIR}/test_spectrograms",
                          mode="test",
                          transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset,
                          batch_size=CFG.batch_size,
                          shuffle=True)
test_loader = DataLoader(test_dataset,
                         batch_size=CFG.batch_size,
                         shuffle=False)

# Example to verify the dataset
spectrogram, label = train_dataset[0]
print(f"Spectrogram shape: {spectrogram.shape}")
print(f"Label shape: {label.shape}")

# Additional training code would go here
