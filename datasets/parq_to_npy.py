import os
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from configs.config_1 import CFG

# Ensure the save directories exist
os.makedirs(f"{CFG.NPY_SAVE_DIR}/train_spectrograms", exist_ok=True)
os.makedirs(f"{CFG.NPY_SAVE_DIR}/test_spectrograms", exist_ok=True)


# Define a function to process a single spectrogram ID
def process_spec(spec_id, split="train"):
    spec_path = f"{CFG.BASE_PATH}/{split}_spectrograms/{spec_id}.parquet"
    spec = pd.read_parquet(spec_path)
    spec = spec.fillna(
        0
    ).values[:,
             1:].T  # Fill NaN values with 0, transpose (Time, Freq) -> (Freq, Time)
    spec = spec.astype("float32")
    np.save(f"{CFG.NPY_SAVE_DIR}/{split}_spectrograms/{spec_id}.npy", spec)


# Load train and test data
df = pd.read_csv(f'{CFG.BASE_PATH}/train.csv')
test_df = pd.read_csv(f'{CFG.BASE_PATH}/test.csv')

# Process training data
spec_ids = df["spectrogram_id"].unique()
_ = joblib.Parallel(n_jobs=-1, backend="loky")(
    joblib.delayed(process_spec)(spec_id, "train")
    for spec_id in tqdm(spec_ids, total=len(spec_ids)))

# Process test data
test_spec_ids = test_df["spectrogram_id"].unique()
_ = joblib.Parallel(n_jobs=-1, backend="loky")(
    joblib.delayed(process_spec)(spec_id, "test")
    for spec_id in tqdm(test_spec_ids, total=len(test_spec_ids)))
