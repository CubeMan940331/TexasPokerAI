import torch
import torch.nn as nn
import torch.nn.functional as F

class StrategyNet(nn.Module):
    def __init__(self, input_dim, output_dim=3, dropout=0.2):
        super(StrategyNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.norm1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.norm2 = nn.LayerNorm(128)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.tanh(self.norm1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.tanh(self.norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc_out(x)
        return F.softmax(x, dim=-1)
