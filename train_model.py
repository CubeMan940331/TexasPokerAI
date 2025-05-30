from pypokerengine.api.game import setup_config
from pypokerengine.api.emulator import Emulator
from pypokerengine.api.emulator import Event
from pypokerengine.engine.table import Table
from pypokerengine.engine.player import Player
from ai.regret_net import RegretNet
from pypokerengine.engine.data_encoder import DataEncoder
from engine.state_encoder import encode_state
from pypokerengine.engine.action_checker import ActionChecker
from players.random_player import RandomPlayer
import torch
import torch.nn as nn
import random

initial_stack = 1000
action_dim = 3

def apply_action(action:dict, game_state, E:Emulator) -> tuple[dict, bool]:
    game_state, events = E.apply_action(game_state, action["action"], action["amount"])
    is_finish = E._is_last_round(game_state, E.game_rule)
    return game_state, is_finish

def regret_matching(model: None|RegretNet, game_state:dict):
    if model==None:
        # return uniform distrabution
        return torch.ones((action_dim))/action_dim
    # get who is playing
    p = game_state["next_player"]
    
    table:Table = game_state["table"]
    player:Player = table.seats.players[p]
    player.hole_card

    player_uuid = player.uuid

    hole_cards = DataEncoder.encode_player(player,holecard=True)["hole_card"]
    round_state = DataEncoder.encode_round_state(game_state)
    
    state_vec = encode_state(player_uuid, hole_cards, round_state)
    
    state_vec = torch.tensor(state_vec)
    V_plus = nn.ReLU(model(state_vec))
    S = torch.sum(V_plus)
    if S==0:
        # return uniform distrabution
        return torch.ones((action_dim))/action_dim
    return V_plus / S

def instantaneous_regret(payoff:torch.Tensor, prob_vec:torch.Tensor):
    expected_payoff = torch.sum(payoff * prob_vec, keepdim=True)
    return payoff - expected_payoff

def traverse(
        game_state:dict, E:Emulator, is_end:bool,
        p:int,
        model:None | RegretNet,
        regret_mem, strategy_mem,
        t: int
    ) -> float:
    table:Table = game_state["table"]
    player:Player=table.seats.players[p]

    if is_end:
        # caculate payoff
        return player.stack-initial_stack
    
    prob_vec = regret_matching(model, game_state)
    
    if game_state["next_player"]==p:
        payoff = torch.zeros((action_dim))
        for idx, action in enumerate(E.generate_possible_actions(game_state)):
            if action["amount"]<0: continue
            nx_game_state, is_finish = apply_action(action, game_state, E)
            payoff[idx] = traverse(
                game_state, E, is_finish, p, model, regret_mem, strategy_mem, t
            )
            r = instantaneous_regret(payoff, prob_vec)
            # TODO : insert (game_state, t, r) to regret_mem
    else:
        # TODO : insert (game_state, t, prob_vec) to strategy_mem

        valid_actions = ActionChecker.legal_actions(
            table.seats.players, p,
            game_state["small_blind_amount"],
            game_state["street"],
            game_state["raise_cnt"]
        )
        if valid_actions[2]["amount"]<0: prob_vec[2]=0
        action = random.choices(valid_actions, weights=prob_vec)
        nx_game_state, is_finish = apply_action(action, game_state, E)

        return traverse(nx_game_state, E, is_finish, p, model, regret_mem, strategy_mem, t)

def main():
    ...

if __name__=='__main__':
    main()
