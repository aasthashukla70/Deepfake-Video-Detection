import torch.nn as nn
from torchvision import models


class DeepFakeCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # Load pretrained ResNet18
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Freeze all pretrained layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.model(x)