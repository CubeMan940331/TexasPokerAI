import torch
import torch.nn as nn
import torch.nn.functional as F

class StrategyNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(StrategyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)
