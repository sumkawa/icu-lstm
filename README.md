# EEG Harmful Brain Activity Classification - With 1d, 2d ConvNets

## Index

- [Description](#description)
- [Setup](#setup)

## Description

This repository contains utils, custom DataSet + DataLoader classes, notebooks, etc I've developed for harmful brain activity classification, using data from the HMS Kaggle competition. Goal is also to log things that I've learned along the way with this project.

## Setup

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

Seed database by converting parquet to npy for more efficient data loading and model compatibility:

```bash
python parq_to_npy.py
```

(Optional) Visualize Data with matplotlib:

```bash
python visualize.py
```
