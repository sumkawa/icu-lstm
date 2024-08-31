import os
import torch
import pandas as pd
import numpy as np
import random
from tqdm.notebook import tqdm

from scipy.signal import butter, lfilter
import timm

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torch.distributions import Beta


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p),
                        (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        super(GeM, self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return (self.__class__.__name__ +
                f"(p={self.p.data.tolist()[0]:.4f},eps={self.eps})")


class Mixup(nn.Module):
    """
  Implements mixup - make new input by taking linear combination
  of two inputs, and mix labels using same linear combination
  """

    def __init__(self, mix_beta, mixadd=False):
        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)
        self.mixadd = mixadd

    def forward(self, X, Y):
        bs = X.shape[0]
        perm = torch.randperm(bs).to(X.device)
        lambda_val = self.beta_distribution.sample((bs, )).to(X.device)

        if len(X.shape) == 2:
            X = lambda_val.view(-1,
                                1) * X + (1 - lambda_val.view(-1, 1)) * X[perm]
        elif len(X.shape) == 3:
            X = lambda_val.view(
                -1, 1, 1) * X + (1 - lambda_val.view(-1, 1, 1)) * X[perm]
        else:
            X = lambda_val.view(
                -1, 1, 1, 1) * X + (1 - lambda_val.view(-1, 1, 1, 1)) * X[perm]

        if self.mixadd:
            Y = (Y + Y[perm]).clip(0, 1)
        else:
            Y = lambda_val.view(-1,
                                1) * Y + (1 - lambda_val.view(-1, 1)) * Y[perm]

        return X, Y


class SpectrogramPreprocess(nn.Module):

    def __init__(self, cfg):
        super(SpectrogramPreprocess, self).__init__()
        self.cfg = cfg
        self.mel_spectrogram = MelSpectrogram(**cfg.spec_args).to(cfg.device)
        self.amplitude_to_db = AmplitudeToDB().to(cfg.device)

    def forward(self, x):
        batch_size, time, channels = x.shape
        x = x.permute(0, 2, 1)  # Now shape is [batch_size, channels, time]
        specs = []
        for i in range(channels):
            spec = self.mel_spectrogram(x[:, i, :])
            spec = self.amplitude_to_db(spec)
            specs.append(spec)
        spectrogram = torch.stack(specs,
                                  dim=1)  # Stack along the channels dimension
        return spectrogram


class Net(nn.Module):

    def __init__(self, cfg, device):
        super(Net, self).__init__()
        self.cfg = cfg
        self.device = device
        self.num_classes = len(self.cfg.targets)

        # Preprocess raw EEG into MelSpecs
        self.preprocessing = nn.Sequential(MelSpectrogram(**cfg.spec_args),
                                           AmplitudeToDB()).to(self.device)

        # Timm Backbone
        self.backbone = timm.create_model(cfg.backbone,
                                          pretrained=cfg.pretrained,
                                          num_classes=0,
                                          global_pool="",
                                          in_chans=self.cfg.in_channels,
                                          **cfg.model_args).to(self.device)
        out_ch = self.backbone.conv_stem.out_channels

        # Mixup
        self.mixup = Mixup(cfg.mixup_beta).to(self.device)
        self.mixup_signal = cfg.mixup_signal
        self.mixup_spectrogram = cfg.mixup_spectrogram

        # Pool
        if cfg.pool == "gem":
            self.global_pool = GeM(p_trainable=cfg.gem_p_trainable).to(
                self.device)
        elif cfg.pool == "identity":
            self.global_pool = torch.nn.Identity().to(self.device)
        elif cfg.pool == "avg":
            self.global_pool = nn.AdaptiveAvgPool2d(1).to(self.device)

        # Head
        self.head = nn.Linear(self.backbone.num_features,
                              self.num_classes).to(self.device)
        self.loss_fn = torch.nn.KLDivLoss(reduction='batchmean').to(
            self.device)
        print(f'Params: {count_parameters(self)}')

    def forward(self, batch):
        x = batch['input'].to(self.device)
        y = batch['target'].float().to(self.device)

        if self.training and self.mixup_signal:
            x, y = self.mixup(x, y)

        x = x.float()
        # Spectrogram processing (stack 16 channels into one image)
        bs, l, c = x.shape  # Correctly identify length and channels
        x = x.permute(0, 2, 1).reshape(
            bs * c, l
        )  # Move channels to batch dimension and prepare for MelSpectrogram
        x = self.preprocessing(x).to(self.device).float()

        bsc, h, w = x.shape
        x = x.reshape(bs, c, h, w)

        # Dropout
        if self.training:
            for tt in range(x.shape[0]):
                if self.cfg.aug_drop_spec_prob > np.random.random():
                    drop_ct = np.random.randint(1,
                                                1 + self.cfg.aug_drop_spec_max)
                    drop_idx = np.random.choice(np.arange(x.shape[1]), drop_ct)
                    x[tt, drop_idx] = 0
        x = x.reshape(bs, 1, c * h, w)

        # Mixup after spectrogram processing
        if self.training and self.mixup_spectrogram:
            x, y = self.mixup(x, y)

        # Backbone + Head
        x = self.backbone(x)
        x = self.global_pool(x).view(bs, -1)

        logits = self.head(x)

        logits_after_softmax = F.log_softmax(logits, dim=1)

        loss = self.loss_fn(F.log_softmax(logits, dim=1), y)
        outputs = {'loss': loss}
        if not self.training:
            outputs['logits'] = logits

        return outputs
