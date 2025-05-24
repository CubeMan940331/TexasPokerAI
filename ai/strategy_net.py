import torch
import torch.nn as nn
import torch.nn.functional as F

class StrategyNet(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        """
        Neural network to predict strategy (action probabilities) from state.
        input_dim: dimension of state input vector.
        output_dim: number of actions (default 3 for [raise, call, fold]).
        """
        super(StrategyNet, self).__init__()
        # Two-layer feed-forward network
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_out = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Softmax to produce a probability distribution over actions
        x = self.fc_out(x)
        x = F.softmax(x, dim=-1)
        return x
