import torch
import torch.nn as nn

class CNN_BiGRU(nn.Module):
    def __init__(self, input_channels=10, output_channels=4):
        super(CNN_BiGRU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        self.gru = nn.GRU(64, output_channels, bidirectional=True)

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(2, 0, 1)
        x, _ = self.gru(x)
        return x