import torch
import torch.optim as optim
from regret_net import RegretNet
from strategy_net import StrategyNet
from memory import Memory
from game_simulator import simulate_episode

input_size = 4  # Example: 4-dim state
action_size = 3  # fold, call, raise

regret_net = RegretNet(input_size, action_size)
strategy_net = StrategyNet(input_size, action_size)
regret_optimizer = optim.Adam(regret_net.parameters(), lr=1e-3)
strategy_optimizer = optim.Adam(strategy_net.parameters(), lr=1e-3)

regret_memory = Memory()
strategy_memory = Memory()

def traverse_and_store():
    samples = simulate_episode()
    for state, regrets, strategy in samples:
        regret_memory.add((torch.tensor(state), torch.tensor(regrets)))
        strategy_memory.add((torch.tensor(state), torch.tensor(strategy)))

def train_network(net, memory, optimizer, loss_fn, batch_size=64):
    if len(memory) == 0:
        return
    batch = memory.sample(batch_size)
    states, targets = zip(*batch)
    states = torch.stack(states)
    targets = torch.stack(targets)

    preds = net(states)
    loss = loss_fn(preds, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

for iteration in range(1000):
    traverse_and_store()
    train_network(regret_net, regret_memory, regret_optimizer, torch.nn.MSELoss())
    train_network(strategy_net, strategy_memory, strategy_optimizer, torch.nn.KLDivLoss(reduction='batchmean'))

    if iteration % 100 == 0:
        print(f"Iteration {iteration}: regret_mem={len(regret_memory)} strategy_mem={len(strategy_memory)}")
