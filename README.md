# EEG Harmful Brain Activity Classification - With 1d, 2d ConvNets

## Index

- [Description](#description)
- [Setup](#setup)

## Description

This repository contains utils, custom DataSet + DataLoader classes, notebooks, etc I've developed for harmful brain activity classification, using data from the HMS Kaggle competition. Goal is also to log things that I've learned along the way with this project.

## What's Next?
For the 2d model I'm looking into some other signal processing augmentation ideas, like focusing in on the middle 10 seconds of the spectrogram as inspired by [this kaggle post](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/472976).

I'm also interested in learning the theory behind time series and classification of time series data using 1d convolutions, so I think I'll develop a 1d conv-net.

Finally, I want to deploy this on my personal website so I can understand what making nice UI/UX for AI interfaces is like (coming very soon).

## Setup
This project uses conda environments.

Clone the repo:

```bash
git clone https://github.com/sumkawa/icu-lstm.git
cd icu-lstm
```

Install dependencies, required packages, and setup conda development env:

```bash
make install
```

Get data from kaggle competition:

```bash
kaggle competitions download -c hms-harmful-brain-activity-classification -p ./data
```

Activate conda env:

```bash
conda activate icu_classifier_env
```

### Converting .parquet to .npy, visualize initial data

Give proper user perms:

```bash
chmod +x visualize.py parq_to_npy.py
```

Seed database by converting parquet to npy for more efficient data loading and model compatibility (this step is only necessary if training with efficient net - mixnet implementations convert raw EEG data on the fly for reasons explained in the research md):

```bash
python parq_to_npy.py
```

(Optional) Visualize Data with matplotlib:

```bash
python visualize.py
```
