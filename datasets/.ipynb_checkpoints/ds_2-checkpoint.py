def batch_to_device(batch, device):
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict

def butterworth_filter(data, high_frequency=20, low_frequency=0.5, sample_rate=200, order=2):
  nyquist_frequency = sample_rate * 0.5
  high = high_frequency / nyquist_frequency
  low = low_frequency / nyquist_frequency
  b, a = butter(order, [low, high], btype='band', analog = False)
  filtered = lfilter(b, a, data, axis = 0)
  return filtered

class EEGDataset(Dataset):
  def __init__(self, df, CFG, augmentation = None, mode = "train"):
    self.df = df.copy()
    self.CFG = CFG
    print(f'mode: {mode}') # df shape: {df.shape}

    # Channel pairs for subtraction
    self.s0 = ['Fp1', 'Fp2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 
               'Fp1', 'Fp2','F3', 'F4', 'C3', 'C4', 'P3', 'P4']
    self.s1 = ['F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'O1', 'O2', 'F3', 
               'F4', 'C3','C4', 'P3', 'P4', 'O1', 'O2']
    self.mode = mode
    self.augmentation = augmentation
    self.data_folder = CFG.data_folder
    self.targets = CFG.targets
    self.eegs = self.df['eeg_id'].values
  
  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    eeg_id, eeg_label_offset_seconds = row[['eeg_id', 
      'eeg_label_offset_seconds']].astype(int)
    y = row[self.CFG.targets].values.astype(np.float32)
    eeg, center = self.load_one(eeg_id, eeg_label_offset_seconds)
    feature_dict = {
            "input": torch.from_numpy(eeg),
            "center":torch.tensor(center, dtype = torch.long),
            "target":torch.from_numpy(y)
        }
        
    return feature_dict
  
  def __len__(self):
    return len(self.eegs)

  def load_one(self, eeg_id, eeg_label_offset_seconds=0):
    eeg_combined = pd.read_parquet(f'{self.data_folder}{eeg_id}.parquet')
    print(eeg_combined.head())
    label_start = int(200* eeg_label_offset_seconds)
    # Setting the window is an idea from:
    # https://www.researchgate.net/publication/354358246_A_Novel_Two-Stage_Refine_Filtering_Method_for_EEG-Based_Motor_Imagery_Classification
    window = 10000

    # To improve generalization, randomly shift start window
    if self.mode == "train":
      # calculate random start position
      label_start_shift = int(np.random.uniform(label_start - window//3, 
                                                label_start + window//3))
      # ensure start doesn't go out of bounds
      label_start_shift =  np.clip(label_start_shift, 0, 
                                   eeg_combined.shape[0] - window)
    else:
      label_start_shift = label_start
    shift = label_start - label_start_shift

    eeg = eeg_combined.iloc[label_start_shift:label_start_shift + window]

    # Double Banana
    eeg_1 = eeg[self.s0].values
    eeg_2 = eeg[self.s1].values

    x = eeg_1 - eeg_2
    # filter null
    x[np.isnan(x)] = 0

    # Random Flip Augmentation
    if self.mode == 'train':
      if np.random.random() > 0.5:
        x = x[::-1].copy()
      if np.random.random() > 0.5:
        x[:, np.arange(x.shape[-1])[1::2]], x[:, np.arange(x.shape[-1])[0::2]] = \
          x[:, np.arange(x.shape[-1])[0::2]], x[:, np.arange(x.shape[-1])[1::2]]
    
    x = butterworth_filter(x, self.CFG.butter_high_freq, self.CFG.butter_low_freq, order=self.CFG.butter_order)

    if (self.mode == "train") and (self.cfg.aug_bandpass_prob > np.random.random() ):
            filt_idx = np.random.choice(np.arange(x.shape[-1]), 1 + np.random.randint(self.cfg.aug_bandpass_max))
            high_freq_aug = np.random.randint(10, self.cfg.butter_high_freq)
            low_freq_aug = np.random.uniform(self.cfg.butter_low_freq, 2)
            x[:,filt_idx] = butterworth_filter(x[:,filt_idx], high_freq=high_freq_aug, low_freq=low_freq_aug, order=max(self.cfg.butter_order, 1))
    x = x.clip(-1024, 1024)
    x /= 32
    
    center = shift + window//2
    return x, center