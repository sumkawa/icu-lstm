import torch
import torch.nn as nn
import timm

class Net(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super(Net, self).__init__()

        self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=0)

        self.fc = nn.Linear(self.backbone.num_features, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)

        x = self.fc(x)
        
        return x
