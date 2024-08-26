def batch_to_device(batch, device):
    return {key: batch[key].to(device) for key in batch}

def butter_bandpass_filter(data, high_freq=20, low_freq=0.5, sampling_rate=200, order=2):
    nyquist = 0.5 * sampling_rate
    high_cutoff = high_freq / nyquist
    low_cutoff = low_freq / nyquist
    b, a = butter(order, [low_cutoff, high_cutoff], btype='band', analog=False)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data

class EEGDataset(Dataset):
    def __init__(self, df, CFG, augmentation=None, mode="train"):
        self.df = df.copy()
        self.CFG = CFG

        # Channel pairs for subtraction
        self.s0 = ['Fp1', 'Fp2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 
                   'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4']
        self.s1 = ['F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'O1', 'O2', 'F3',
                   'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
        self.mode = mode
        self.augmentation = augmentation
        self.data_folder = CFG.data_folder
        self.targets = CFG.targets
        self.eegs = self.df['eeg_id'].values

        # Advanced: Filter & weight decay based on vote counts (confidence level)
        if mode == "train":
          self.apply_vote_count_filtering()
          self.apply_label_weighting()
        
    def apply_vote_count_filtering(self):
      """Filter data based on vote count ranges specific in CFG."""
      filter_low = min(i[0] for i in self.CFG.vote_ct_ranges)
      filter_high = max(i[1] for i in self.CFG.vote_ct_ranges)
      filterIdx = (self.df.filter(like="_vote").sum(1) >= filter_low) & \
                  (self.df.filter(like="_vote").sum(1) <= filter_high)
      self.df = self.df[filterIdx].copy()
      print(f'Filtered dataset based on vote counts, resulting shape: {self.df.shape}')

    def apply_label_weighting(self):
      """Assign label weights based on confidence and epoch"""
      self.df['vote_ct'] = self.df.filter(like="_vote").sum(1).values
      self.df['label_weight'] = 0.0

      # Calculate decay factor for the current epoch
      step_decay = 1 - ((self.CFG.curr_epoch / self.CFG.epochs) * self.CFG.vote_ct_weight_decay)
      vote_ct_weights = [
          max(w * step_decay, mw) if t + 1 != len(self.CFG.vote_ct_weights) else w
          for t, (w, mw) in enumerate(zip(self.CFG.vote_ct_weights, self.CFG.vote_ct_weights_min))
      ]
      print(f'Vote count weights after step decay: {vote_ct_weights}')

      for (filter_low, filter_hi), wt in zip(self.CFG.vote_ct_ranges, vote_ct_weights):
          filterIdx = (self.df.filter(like='_vote').sum(1) >= filter_low) & \
                    (self.df.filter(like='_vote').sum(1) <= filter_hi)
          self.df.loc[filterIdx, 'label_weight'] = wt

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        eeg_id, eeg_label_offset_seconds = row[['eeg_id', 'eeg_label_offset_seconds']].astype(int)
        y = row[self.CFG.targets].values.astype(np.float32)
        eeg, center = self.load_one(eeg_id, eeg_label_offset_seconds)

        feature_dict = {
            "input": torch.from_numpy(eeg),
            "center": torch.tensor(center, dtype=torch.long),
            "target": torch.from_numpy(y)
        }

        if self.mode == "train":
          feature_dict["label_weight"] = torch.tensor(row['label_weight'], dtype=torch.float32)
        return feature_dict

    def __len__(self):
        return len(self.eegs)

    def load_one(self, eeg_id, eeg_label_offset_seconds=0):
        eeg_combined = pd.read_parquet(f'{self.data_folder}{eeg_id}.parquet')
        label_start = int(200 * eeg_label_offset_seconds)
        window = 10000

        # Random shift for training
        if self.mode == "train":
            label_start_shift = int(np.random.uniform(label_start - window // 3,
                                                      label_start + window // 3))
            label_start_shift = np.clip(label_start_shift, 0,
                                        eeg_combined.shape[0] - window)
        else:
            label_start_shift = label_start
        shift = label_start - label_start_shift

        eeg = eeg_combined.iloc[label_start_shift:label_start_shift + window]

        # Double Banana
        eeg_1 = eeg[self.s0].values
        eeg_2 = eeg[self.s1].values

        x = eeg_1 - eeg_2
        x[np.isnan(x)] = 0

        # Random Flip Augmentation
        if self.mode == 'train':
            if np.random.random() > 0.5:
                x = x[::-1].copy()
            if np.random.random() > 0.5:
                x[:, np.arange(x.shape[-1])[1::2]], x[:, np.arange(x.shape[-1])[0::2]] = \
                    x[:, np.arange(x.shape[-1])[0::2]], x[:, np.arange(x.shape[-1])[1::2]]

        x = butter_bandpass_filter(x, self.CFG.butter_high_freq, self.CFG.butter_low_freq, order=self.CFG.butter_order)

        if self.mode == "train" and self.CFG.aug_bandpass_prob > np.random.random():
            filt_idx = np.random.choice(np.arange(x.shape[-1]), 1 + np.random.randint(self.CFG.aug_bandpass_max))
            high_freq_aug = np.random.randint(10, self.CFG.butter_high_freq)
            low_freq_aug = np.random.uniform(self.CFG.butter_low_freq, 2)
            x[:, filt_idx] = butter_bandpass_filter(x[:, filt_idx], high_freq=self.CFG.butter_high_freq, low_freq=self.CFG.butter_low_freq, order=max(self.CFG.butter_order, 1))

        x = x.clip(-1024, 1024)
        x /= 32

        center = shift + window // 2
        return x, center