import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple1DCNN(nn.Module):
    def __init__(self, num_channels: int, num_classes: int, dropout_p: float = 0.5):
        super(Simple1DCNN, self).__init__()
        
        # First 1D convolutional block
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Second 1D convolutional block
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Third 1D convolutional block
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 12, 128)  # Assuming input length reduces to 12 after pooling
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Apply convolutional blocks
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))

        # Flatten and apply fully connected layers
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(self.fc1(x))
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)