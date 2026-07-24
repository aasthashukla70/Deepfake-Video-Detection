import torch
import torch.nn as nn
from torchvision import models
class CNNLSTM(nn.Module):

    def __init__(self):
        super().__init__()

        self.resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )


        # Freeze ResNet weights
        for param in self.resnet.parameters():
            param.requires_grad = False


        self.resnet.fc = nn.Identity()

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=1,
            batch_first=True
        )


        self.classifier = nn.Linear(
            256,
            2
        )

    def forward(self, x):

        batch_size, sequence_length, C, H, W = x.shape


        x = x.view(
            batch_size * sequence_length,
            C,
            H,
            W
        )

        self.resnet.eval()
        features = self.resnet(x)


        features = features.view(
            batch_size,
            sequence_length,
            512
        )


        output, (hidden, cell) = self.lstm(features)


        last_output = output[:, -1, :]


        prediction = self.classifier(last_output)


        return prediction    
    
if __name__ == "__main__":

    model = CNNLSTM()

    dummy_video = torch.randn(
        1,
        60,
        3,
        224,
        224
    )

    output = model(dummy_video)

    print("Output shape:", output.shape)