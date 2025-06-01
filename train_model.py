from poker_env import PokerEnv
from model import RegretNet, encode_state, regret_matching, instantaneous_regret
import random
import copy

import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import trange, tqdm

import multiprocessing

import sys
sys.setrecursionlimit(1000000)

CFR_T = 1000
K = 256

class ReservoirMemory:
    def __init__(self, capacity: int):
        self.capacity = capacity
        manager = multiprocessing.Manager()
        self.datalock = manager.Lock()
        self.data = manager.list()

    def add(self, element):
        """
        Add a new element to the memory using reservoir sampling.
        """
        with self.datalock:
            if len(self.data) < self.capacity:
                self.data.append(element)
            else:
                # Reservoir sampling: replace with decreasing probability
                idx = random.randint(0, len(self.data))
                if idx < self.capacity:
                    self.data[idx] = element

    def sample(self, batch_size: int, epochs:int):
        """
        Sample a batch of tensors uniformly at random.
        Returns a tensor stacked along the first dimension.
        """
        with self.datalock:
            if batch_size > len(self.data):
                raise ValueError(f"Cannot sample {batch_size} elements from memory of size {len(self.data)}")
            local_data = list(self.data)
        for _ in range(epochs):
            yield random.sample(local_data, batch_size)

    def __len__(self):
        return len(self.data)

    def clear(self):
        with self.datalock:
            self.data.clear()

def train_regret_model(mem:ReservoirMemory, batch_size:int, epochs:int) -> nn.Module:
    model=RegretNet(111,3).to('cuda')
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    with trange(epochs, desc=f"loss = 0.000000") as tq:
        for batch, _ in zip(mem.sample(batch_size, epochs), tq):
            states, regrets, t = zip(*batch)
            states = torch.stack(states).to('cuda')
            regrets = torch.stack(regrets).to('cuda')
            t = torch.tensor(t).reshape((batch_size,-1)).to('cuda')
            t = torch.sqrt(t/CFR_T*2)
            
            pred = model(states)
            
            loss:torch.Tensor = nn.MSELoss()(pred*t, regrets*t)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            tq.desc = f"loss = {loss.item(): .6f}"
    
    return model.to('cpu')

def get_payoff(game_env:PokerEnv, p:int, model):
    while (not game_env.is_end) and game_env.round<=500:
        a = None
        if game_env.is_betting:
            state_vec = encode_state(game_env)
            prob_vec = regret_matching(state_vec, model)
            valid_action = torch.tensor(game_env.valid_action())
            weights = prob_vec*valid_action
            if torch.sum(weights)<=0: weights=valid_action
            a = random.choices(range(3),weights= weights)[0]
        game_env.next_state(a)
    return game_env.players[p].stack/game_env.init_stack - 1

def traverse(game_env:PokerEnv, p:int, model:None, regret_mem:ReservoirMemory, strategy_mem:ReservoirMemory, t: int) -> float:
    if game_env.is_end or game_env.round>500:
        # return normalized pay off of p
        return game_env.players[p].stack/game_env.init_stack - 1
    if not game_env.is_betting:
        game_env.next_state(None)
        return traverse(game_env, p, model, regret_mem, strategy_mem, t)
    
    state_vec = encode_state(game_env)
    prob_vec = regret_matching(state_vec, model)
    valid_action = torch.tensor(game_env.valid_action())
    
    if game_env.act_idx==p:
        payoff = torch.zeros((3,))
        for idx, flag in enumerate(valid_action):
            if flag:
                copy_env = copy.deepcopy(game_env)
                copy_env.next_state(idx)
                payoff[idx] = get_payoff(copy_env, p, model)
        r = instantaneous_regret(payoff, prob_vec)
        
        regret_mem.add((state_vec, r, t))
    # else: strategy_mem.add((state_vec, prob_vec, t))

    weights = prob_vec*valid_action
    if torch.sum(weights)<=0: weights=valid_action
    a = random.choices(range(3),weights= weights)[0]
    game_env.next_state(a)
    return traverse(game_env, p, model, regret_mem, strategy_mem, t)

def main():
    model = None
    regret_mem = ReservoirMemory(10_000_000)
    strategy_mem = ReservoirMemory(10_000_000)

    for t in trange(1,CFR_T+1, desc="running CFR iterations"):
        for p in range(2):
            with multiprocessing.Pool(12) as pool:
                pool.starmap(traverse, ((PokerEnv(1,10),p , model, regret_mem, strategy_mem, t) for _ in range(K)))

        model = train_regret_model(regret_mem, 2048, 40000)
        torch.save(model, f"regret_net{t:04}.pt")

if __name__=='__main__':
    main()
