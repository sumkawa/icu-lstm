class CFG:
  data_folder = "./data/train_eegs/"
  # "/content/drive/MyDrive/ML_EEG/Data/train_eegs/"

  # Dataset Configs:
  vote_ct_ranges = [[0, 2], [3, 8], [8, 9999]]
  targets = ['seizure_vote','lpd_vote',
             'gpd_vote','lrda_vote','grda_vote','other_vote']
  vote_ct_weights = [0.3, 0.5, 1.0]
  vote_ct_weight_decay = 1.2
  vote_ct_weights_min = [0.02, 0.02, 1.0]
  curr_epoch = 0

  # Regularization:
  butter_order = 1
  butter_high_freq = 30
  butter_low_freq = 1.6

  # Preprocessing & Model:
  spec_args = dict(sample_rate=200, n_fft=1024,
                   n_mels=128, f_min=0.53, f_max=40,
                   win_length=128, hop_length=39)
  model_args = dict(drop_rate = 0.2)
  num_classes = 6
  backbone = "mixnet_xl"
  pretrained = True
  in_channels = 1

  use_mixup = True
  mixup_spectrogram = False
  mixup_signal = True
  mixup_beta = 1

  aug_drop_spec_prob = 0.5
  aug_drop_spec_max = 8
  aug_bandpass_prob = 0.2
  aug_bandpass_max = 8

  enlarge_len = 2400
  cut_from_large_mel = 0.2

  pool = "gem"
  gem_p_trainable = True


  # Scheduling + Optim:
  epochs = 20
  eval_epochs = 2
  lr = 0.001
  optimizer = "Adam"
  weight_decay = 0
  clip_grad = 4
