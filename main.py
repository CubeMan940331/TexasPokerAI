import random
from tqdm import trange, tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ai.regret_net import RegretNet
from ai.strategy_net import StrategyNet
from memory.regret_memory import RegretMemory
from memory.strategy_memory import StrategyMemory
from engine.runner import CFRRunner
from traverse.traverse import compute_counterfactual_regrets

if __name__ == "__main__":
    input_dim = 113  # state vector length (52 hole + 52 board + 4 street + 5 numeric features)
    regret_net = RegretNet(input_dim)
    strategy_net = StrategyNet(input_dim)
    # Initialize replay memories for regret and strategy samples
    regret_memory = RegretMemory()
    strategy_memory = StrategyMemory()
    # Set up the self-play runner with game parameters
    initial_stack = 1000
    small_blind = 10
    ante = 0
    runner = CFRRunner(regret_net, strategy_net, initial_stack=initial_stack, small_blind=small_blind, ante=ante)
    # Hyperparameters for training
    num_iterations = 10
    episodes_per_iteration = 100
    regret_optimizer = optim.Adam(regret_net.parameters(), lr=0.001)
    strategy_optimizer = optim.Adam(strategy_net.parameters(), lr=0.001)
    # Deep CFR training loop
    for it in trange(num_iterations,desc="Training Iterations"):
        # Self-play to collect episodes
        for ep in trange(episodes_per_iteration,desc=f"Self-Play Iter {it+1}", leave=False):
            players, payoffs = runner.play_episode()
            # Compute regrets for this episode
            regret_samples = compute_counterfactual_regrets(players, payoffs,
                                                           initial_stack=runner.initial_stack,
                                                           small_blind=runner.small_blind,
                                                           ante=runner.ante,
                                                           blind_structure=runner.blind_structure)
            # Store regrets and strategy data from the episode
            for state_vec, regret_vec in regret_samples:
                regret_memory.add(state_vec, regret_vec)
            for p in players:
                for decision in p.episode_history:
                    strategy_memory.add(decision["state_vec"], decision["strategy"])
        # Train the regret network on collected regret samples
        if len(regret_memory) > 0:
            loss_fn = nn.MSELoss()
            batch_size = 128
            random.shuffle(regret_memory.memory)
            for i in range(0, len(regret_memory.memory), batch_size):
                batch = regret_memory.memory[i:i+batch_size]
                state_batch = torch.tensor(np.array([s for (s, _) in batch]), dtype=torch.float32)
                target_regret_batch = torch.tensor(np.array([r for (_, r) in batch]), dtype=torch.float32)
                regret_optimizer.zero_grad()
                pred_regret = regret_net(state_batch)
                loss = loss_fn(pred_regret, target_regret_batch)
                loss.backward()
                regret_optimizer.step()
        # Clear regret memory (do not carry over to next iteration)
        regret_memory.clear()
        print(f"Iteration {it+1}/{num_iterations} complete.")
    # After iterations, train the strategy network on all accumulated strategy samples (average strategy)
    if len(strategy_memory) > 0:
        batch_size = 128
        for epoch in range(3):
            random.shuffle(strategy_memory.memory)
            for i in range(0, len(strategy_memory.memory), batch_size):
                batch = strategy_memory.memory[i:i+batch_size]
                state_batch = torch.tensor(np.array([s for (s, _) in batch]), dtype=torch.float32)
                target_strategy_batch = torch.tensor(np.array([dist for (_, dist) in batch]), dtype=torch.float32)
                strategy_optimizer.zero_grad()
                predicted_probs = strategy_net(state_batch)
                # Mean squared error on the distributions
                loss = ((predicted_probs - target_strategy_batch) ** 2).mean()
                loss.backward()
                strategy_optimizer.step()
    # Training complete. The strategy_net can now be used for decisions or evaluation.
    torch.save(strategy_net.state_dict(), "models/strategy_net.pt")
    print("Training complete")
