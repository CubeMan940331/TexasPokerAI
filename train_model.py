import torch
import torch.nn as nn
import torch.optim as optim

def train_regret_network(regret_net, regret_memory, optimizer, batch_size=128, epochs=1):
    """
    Train the regret network on samples from regret_memory.
    Uses Mean Squared Error loss between predicted and target regrets.
    """
    regret_net.train()
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        random.shuffle(regret_memory.memory)
        for i in range(0, len(regret_memory.memory), batch_size):
            batch = regret_memory.memory[i:i+batch_size]
            state_batch = torch.tensor([s for (s, _) in batch], dtype=torch.float32)
            target_batch = torch.tensor([r for (_, r) in batch], dtype=torch.float32)
            optimizer.zero_grad()
            pred = regret_net(state_batch)
            loss = loss_fn(pred, target_batch)
            loss.backward()
            optimizer.step()

def train_strategy_network(strategy_net, strategy_memory, optimizer, batch_size=128, epochs=1):
    """
    Train the strategy network on samples from strategy_memory.
    Uses MSE loss on action probability distributions.
    """
    strategy_net.train()
    for epoch in range(epochs):
        random.shuffle(strategy_memory.memory)
        for i in range(0, len(strategy_memory.memory), batch_size):
            batch = strategy_memory.memory[i:i+batch_size]
            state_batch = torch.tensor([s for (s, _) in batch], dtype=torch.float32)
            target_batch = torch.tensor([dist for (_, dist) in batch], dtype=torch.float32)
            optimizer.zero_grad()
            pred_probs = strategy_net(state_batch)
            loss = ((pred_probs - target_batch) ** 2).mean()
            loss.backward()
            optimizer.step()
