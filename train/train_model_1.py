import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datasets.ds_1 import EEGDataset
from configs.config_1 import CFG, SPEC_DIR, NPY_SAVE_DIR
import pandas as pd
import torch.nn as nn
import timm
import torch.optim as optim
from torch.utils.data import DataLoader

# Define transformations and augmentations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((400, 300))  # Resizing to a fixed size (height, width)
])

augment = transforms.Compose([
    transforms.RandomErasing(
        p=0.5,
        scale=(0.02, 0.1),
        ratio=(8, 12),  # Frequency mask
        inplace=False),
    transforms.RandomErasing(
        p=0.5,
        scale=(0.02, 0.10),
        ratio=(0.5, 1.5),  # Time mask
        inplace=False)
])

# Load data
df = pd.read_csv(f'{CFG.BASE_PATH}/train.csv')
test_df = pd.read_csv(f'{CFG.BASE_PATH}/test.csv')

# Create datasets
train_dataset = EEGDataset(df=df,
                           data_dir=f"{NPY_SAVE_DIR}/train_spectrograms",
                           mode="train",
                           transform=transform,
                           augment=augment)
test_dataset = EEGDataset(df=test_df,
                          data_dir=f"{NPY_SAVE_DIR}/test_spectrograms",
                          mode="test",
                          transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset,
                          batch_size=CFG.batch_size,
                          shuffle=True)
test_loader = DataLoader(test_dataset,
                         batch_size=CFG.batch_size,
                         shuffle=False)

# Example to verify the dataset
spectrogram, label = train_dataset[0]
print(f"Spectrogram shape: {spectrogram.shape}")
print(f"Label shape: {label.shape}")

# Pretrained Timm EffNet:


# Load the EfficientNet-B0 model
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=len(train_dataset.df['class_label'].unique()))

# Move the model to the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for spectrograms, labels in train_loader:
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(spectrograms)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Track loss
        running_loss += loss.item() * spectrograms.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    # Validation Step
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for spectrograms, labels in test_loader:
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            outputs = model(spectrograms)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Validation Accuracy: {100 * correct / total:.2f}%')

print('Finished Training')
