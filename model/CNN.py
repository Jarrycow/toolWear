import torch.nn as nn
import torch

class CNN(nn.Module):
    def __init__(self, input_channel, output_size, cnnConvKernel, cnnPoolKernel):
        super(CNN, self).__init__()
        self.conv = nn.Conv2d(input_channel, output_size, cnnConvKernel)
        self.pool = nn.MaxPool1d(cnnPoolKernel)

    def forward(self, x):
        x = x.reshape(int(x.shape[0]/4), x.shape[1], int(x.shape[2]*4))
        x = self.conv(x)
        x = self.pool(x.to(torch.int32))
        return x

        # 
