import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from configs.config_1 import CFG


class EEGDataset(Dataset):

    def __init__(self,
                 df,
                 data_dir,
                 mode="train",
                 transform=None,
                 augment=None):
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
        target_width = 300
        if spectrogram.shape[1] > target_width:
            spectrogram = spectrogram[:, :target_width]  # Trim
        else:
            pad_width = target_width - spectrogram.shape[1]
            spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)),
                                 mode='constant')

        spectrogram = np.clip(spectrogram,
                              a_min=np.exp(-4.0),
                              a_max=np.exp(8.0))
        spectrogram = np.log(spectrogram)
        spectrogram -= np.mean(spectrogram)
        spectrogram /= (np.std(spectrogram) + 1e-6)

        # Stack grayscale image into 3 channels
        spectrogram = np.stack([spectrogram] * 3, axis=-1)

        return spectrogram

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        spec_path = self.spec_paths[idx]
        spectrogram = np.load(spec_path)
        spectrogram = self.preprocess(spectrogram)

        if self.transform:
            spectrogram = self.transform(spectrogram)

        if self.augment and self.mode == "train":
            spectrogram = self.augment(spectrogram)

        if not isinstance(spectrogram, torch.Tensor):
            spectrogram = torch.tensor(spectrogram,
                                       dtype=torch.float32).permute(2, 0, 1)

        if self.mode != "test":
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            label = F.one_hot(label,
                              num_classes=len(
                                  self.df['class_label'].unique())).float()
            return spectrogram, label
        else:
            return spectrogram
