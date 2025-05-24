import torch
import torch.nn as nn
import torch.nn.functional as F

class RegretNet(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        """
        Neural network to predict regret values for each action.
        input_dim: dimension of state input vector.
        output_dim: number of actions (default 3 for [raise, call, fold]).
        """
        super(RegretNet, self).__init__()
        # Simple two-hidden-layer feed-forward network
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_out = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # No activation on output (regrets can be positive or negative)
        return self.fc_out(x)
