import os
import torch
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from scipy.signal import butter, lfilter
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import timm
import random
from datasets.ds_2_c import EEGDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'using device: {device}')

data_path = './data/'

df = pd.read_csv(os.path.join(data_path, 'train.csv'))
# Golden set for validation
df_filtered_golden = pd.read_csv(os.path.join(data_path, 'filtered_data_golden.csv'))

# Regular training set
df_filtered_regular = pd.read_csv(os.path.join(data_path, 'filtered_data_regular.csv'))

# Large training set with low count votes
df_filtered_large = pd.read_csv(os.path.join(data_path, 'filtered_data_large.csv'))

def filter_train(df, filter_ct, drop_dupes=False):
    # Filter rows where the sum of votes across all label columns is greater than filter_ct
    label_columns = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
    df = df[df[label_columns].sum(axis=1) > filter_ct].copy()

    # Create a unique identifier combining eeg_id and label columns
    df['eeg_id_l'] = df[['eeg_id'] + label_columns].astype(str).agg('_'.join, axis=1)

    rows = []
    # Remove overlapping and redundant rows
    for eeg_id_l in tqdm(df['eeg_id_l'].unique(), desc="Processing Groups"):
        df0 = df[df['eeg_id_l'] == eeg_id_l].reset_index(drop=True).copy()
        offsets = df0['spectrogram_label_offset_seconds'].astype(int).values
        x = np.zeros(offsets.max() + 600)
        for o in offsets:
            x[o:o + 600] += 1
        best_idx = np.argmax([x[o:o + 600].sum() for o in offsets])
        rows.append(df0.iloc[best_idx])

    filtered_df = pd.DataFrame(rows)

    # Drop duplicates
    if drop_dupes:
        filtered_df = filtered_df.drop_duplicates(subset='eeg_id').copy()

    return filtered_df

# # Golden set for validation
# df_filtered_golden = filter_train(df.copy(), filter_ct=8, drop_dupes=True)

# # Regular training set
# df_filtered_regular = filter_train(df.copy(), filter_ct=8, drop_dupes=False)

# # Large training set with low count votes
# df_filtered_large = filter_train(df.copy(), filter_ct=3, drop_dupes=False)

# df_filtered_golden.to_csv('filtered_data_golden.csv', index=False)
# df_filtered_regular.to_csv('filtered_data_regular.csv', index=False)
df_filtered_large.to_csv('filtered_data_large.csv', index=False)

# Display the first few rows of the filtered DataFrame
len(df_filtered_large)

train_dataset = EEGDataset(df=df_filtered_large, CFG = CFG)
train_dataloader = DataLoader(train_dataset, batch_size = 8,  shuffle=True)

test_dataset = EEGDataset(df=df_filtered_golden, CFG = CFG)
test_dataloader = DataLoader(test_dataset, batch_size = 12,  shuffle=True)

batch = next(iter(train_dataloader))

# Print out the shapes of the data in the batch
print("Input shape:", batch['input'].shape)
print("Target shape:", batch['target'].shape)
print("Length train:", len(train_dataset))
# If label weights are included in the dataset
if 'label_weight' in batch:
    print("Label weight shape:", batch['label_weight'].shape)

cfg = CFG()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(cfg, device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = CFG.lr)

# Load the checkpoint
checkpoint = torch.load('./models/effnet_pth_checkpoints/best_model_checkpoint_15.pth')

# Load the model state dict
model.load_state_dict(checkpoint['model_state_dict'])

# Load the optimizer state dict
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Load other information
start_epoch = checkpoint['epoch']

# Load a batch from the DataLoader
batch = next(iter(train_dataloader))

# Print out the shapes of the data in the batch
print("Input shape:", batch['input'].shape)
print("Target shape:", batch['target'].shape)

# If label weights are included in the dataset
if 'label_weight' in batch:
    print("Label weight shape:", batch['label_weight'].shape)

# Training loop
best_val_loss = float('inf')

n_epochs = CFG.epochs
for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0

    for batch in train_dataloader:

        optimizer.zero_grad()

        # Move data to the device (GPU/CPU)
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)

        # Forward pass
        outputs = model(batch)
        loss = outputs['loss']

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_dataloader.dataset)
    torch.cuda.empty_cache()
    print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {epoch_loss:.4f}")

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_batch in test_dataloader:
            val_inputs = val_batch['input'].to(device)
            val_targets = val_batch['target'].to(device)

            val_outputs = model(val_batch)
            val_loss += val_outputs['loss'].item() * val_inputs.size(0)

    val_loss = val_loss / len(test_dataloader.dataset)
    print(f"Validation Loss: {val_loss:.4f}");

    # Save checkpoint if validation loss improves
    if val_loss < best_val_loss:
        print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving checkpoint.")
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_val_loss,
        }, f'./models/effnet_pth_checkpoints/best_model_checkpoint_{epoch+1}.pth')

print("Training complete!")