import torch
import random
from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.players import BasePokerPlayer
from ai.strategy_net import StrategyNet
from engine.state_encoder import encode_state
from players.strategy_player import StrategyPlayer
from players.random_player import RandomPlayer

# Load the trained strategy_net
strategy_net = StrategyNet(input_dim=113)
strategy_net.load_state_dict(torch.load("models/strategy_net.pt"))
strategy_net.eval()

# Configure game
config = setup_config(max_round=100, initial_stack=1000, small_blind_amount=10)
config.register_player(name="StrategyAI", algorithm=StrategyPlayer(strategy_net))
config.register_player(name="RandomAI", algorithm=RandomPlayer())

# Start the game
result = start_poker(config, verbose=1)

# Print outcome
print("\n=== Game Result ===")
for player in result["players"]:
    print(f"{player['name']} - Final Stack: {player['stack']}")

# Play 100 full matches (until one player has 0 stack)
# strategy_full_wins = 0
# total_matches = 100
# initial_stack = 1000
# small_blind = 10

# for match in range(total_matches):
#     strategy_stack = initial_stack
#     random_stack = initial_stack
#     round_count = 0

#     while strategy_stack > 0 and random_stack > 0 and round_count < 1000:
#         config = setup_config(max_round=1, initial_stack=initial_stack, small_blind_amount=small_blind)
#         config.register_player(name="StrategyAI", algorithm=StrategyPlayer(strategy_net))
#         config.register_player(name="RandomAI", algorithm=RandomPlayer())

#         result = start_poker(config, verbose=0)
#         players = {p["name"]: p["stack"] for p in result["players"]}

#         strategy_stack = players["StrategyAI"]
#         random_stack = players["RandomAI"]
#         round_count += 1

#     if strategy_stack > 0 and random_stack == 0:
#         strategy_full_wins += 1

#     print(f"Match {match + 1}: StrategyAI = {strategy_stack}, RandomAI = {random_stack} in {round_count} rounds")

# # Print summary
# print("\n=== Match Summary ===")
# print(f"Total matches: {total_matches}")
# print(f"StrategyAI full wins (opponent busted): {strategy_full_wins}")