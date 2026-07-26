import torch.nn as nn
from torchvision import models


class DeepFakeCNN(nn.Module):

    def __init__(self, use_dropout=False):

        super().__init__()

        self.model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )


        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        for param in self.model.fc.parameters():
            param.requires_grad = True


        in_features = self.model.fc.in_features


        if use_dropout:

            self.model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(
                    in_features,
                    2
                )
            )

        else:

            self.model.fc = nn.Linear(
                in_features,
                2
            )


    def forward(self, x):

        return self.model(x)