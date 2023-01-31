import torch
import torch.nn as nn

class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        # x should be of shape (batch_size, sequence_length, input_size)
        x = x.to(torch.float32)
        x = x.reshape([x.shape[0], x.shape[2], x.shape[1]])
        output, _ = self.gru(x)
        output = self.fc(output[:, -1, :])
        return output