import matplotlib.pyplot as plt
import torch
import random
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets.ds_1 import EEGDataset
from configs.config_1 import CFG
import pandas as pd

NPY_SAVE_DIR = CFG.NPY_SAVE_DIR

torch.manual_seed(21)
random.seed(55)


def visualize_batch(data_loader, num_samples=4):
    # Get the first batch from the DataLoader
    for spectrograms, labels in data_loader:
        fig, axes = plt.subplots(1, num_samples, figsize=(20, 5))

        for i in range(num_samples):
            ax = axes[i]
            # Extract the first channel (since all channels are identical after stacking)
            spec = spectrograms[i][0].numpy()
            # Normalize the image for better color scaling
            spec -= spec.min()
            spec /= spec.max() + 1e-4

            # Convert one-hot encoded label to the class index
            label_index = torch.argmax(labels[i]).item()

            # Display the image with a colormap, adjusting the origin
            cax = ax.imshow(spec,
                            aspect='auto',
                            origin='lower',
                            cmap='viridis')
            ax.set_title(f"Label: {CFG.label2name[label_index]}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Frequency")
            fig.colorbar(cax, ax=ax)

        plt.tight_layout()
        plt.show() 
        # Break after the first batch
        break


if __name__ == "__main__":
    # Load data
    df = pd.read_csv(f'{CFG.BASE_PATH}/train.csv')

    df['eeg_path'] = f'{CFG.BASE_PATH}/train_eegs/' + df['eeg_id'].astype(
        str) + '.parquet'
    df['spec_path'] = f'{CFG.BASE_PATH}/train_spectrograms/' + df[
        'spectrogram_id'].astype(str) + '.parquet'
    df['spec2_path'] = f'{CFG.SPEC_DIR}/train_spectrograms/' + df[
        'spectrogram_id'].astype(str) + '.npy'
    df['class_name'] = df.expert_consensus.copy()
    df['class_label'] = df.expert_consensus.map(CFG.name2label)

    # Define augmentations + transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(
            (400, 300),
            antialias=None)  # Resizing to a fixed size (height, width)
    ])
    augment = transforms.Compose([
        transforms.RandomErasing(
            p=0.5,
            scale=(0.02, 0.1),
            ratio=(8, 12),  # freq mask
            inplace=False),
        transforms.RandomErasing(
            p=0.5,
            scale=(0.02, 0.10),
            ratio=(0.5, 1.5),  # time mask
            inplace=False)
    ])
    # Create dataset + dataloader
    train_dataset = EEGDataset(df=df,
                               data_dir=f"{NPY_SAVE_DIR}/train_spectrograms",
                               mode="train",
                               transform=transform,
                               augment=augment)
    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=True)

    # Visualize a batch of spectrograms
    visualize_batch(train_loader, num_samples=4)
