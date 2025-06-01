import torch
import torch.nn as nn
import torch.nn.functional as F
from poker_env import PokerEnv, PokerStreet

def encode_state(env:PokerEnv) -> torch.Tensor:
    '''
    encode PokerEnv to state vector\\
    one-hot encode hole cards (52)\\
    one-hot encode community cards (52)\\
    one-hot encode street (4)\\
    pot size, player stack, opponent stack (3)
    '''
    result = torch.zeros((111,))
    for i in env.player_cards(env.act_idx): result[i]=1
    for i in env.shared_cards(): result[i+52]=1
    if env.street > PokerStreet.init and env.street < PokerStreet.finish:
        result[104+env.street-1]=1
    result[108]=env.pot()/env.init_stack
    result[109]=env.players[env.act_idx].stack/env.init_stack
    result[110]=env.players[1-env.act_idx].stack/env.init_stack
    return result

HIDDEN = 256
DROPOUT = 0.2
ACTIONS = 3

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc   = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(DROPOUT)
    def forward(self, x):
        y = F.relu(self.norm(self.fc(x)))
        y = self.drop(y)
        return x + y

class RegretNet(nn.Module):
    def __init__(self, input_dim, output_dim=ACTIONS):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.Tanh(),
            nn.Dropout(DROPOUT)
        )
        self.trunk = nn.Sequential(
            ResidualBlock(HIDDEN),
            ResidualBlock(HIDDEN),
            ResidualBlock(HIDDEN)
        )
        self.head = nn.Linear(HIDDEN, output_dim)

    def forward(self, x):
        z = self.proj(x)
        z = self.trunk(z)
        return self.head(z)
    
def regret_matching(state_vec:torch.Tensor, model:None|RegretNet) -> torch.Tensor:
    '''
    return torch.Tensor with size=(ACTIONS,)
    '''
    if model is None:
        # return uniform distribution
        return torch.ones(ACTIONS)/ACTIONS
    model.eval()
    with torch.no_grad():
        V_plus = F.relu(model(state_vec))
        s = torch.sum(V_plus)
    if s==0: return torch.ones(ACTIONS)/ACTIONS
    return V_plus/s

def instantaneous_regret(payoff:torch.Tensor, prob_vec:torch.Tensor) -> torch.Tensor:
    '''
    return torch.Tensor with size=(ACTIONS,)
    '''
    expected_payoff = torch.sum(payoff * prob_vec)
    return payoff - expected_payoff
