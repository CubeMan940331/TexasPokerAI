import torch
import torch.nn as nn

class RegretNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(RegretNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.net(x)
