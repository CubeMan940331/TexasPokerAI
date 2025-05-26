import torch
import random
from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.players import BasePokerPlayer
from ai.strategy_net import StrategyNet
from engine.state_encoder import encode_state


# Define StrategyPlayer that uses strategy_net
class StrategyPlayer(BasePokerPlayer):
    def __init__(self, model, name="StrategyAI"):
        self.model = model
        self.name = name
        self.uuid = None
    
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, new_action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass
    def receive_game_start_message(self, game_info):
        self.uuid = next(p['uuid'] for p in game_info['seats'] if p['name'] == self.name)

    def declare_action(self, valid_actions, hole_card, round_state):
        state_vec = encode_state(self.uuid, hole_card, round_state)
        state_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
        probs = self.model(state_tensor).detach().numpy()[0]

        legal_actions = [a["action"] for a in valid_actions]
        all_actions = ["raise", "call", "fold"]

        # Build masked strategy
        strategy = {a: probs[i] if a in legal_actions else 0.0 for i, a in enumerate(all_actions)}
        total = sum(strategy.values())
        if total == 0:
            strategy = {a: 1.0 / len(legal_actions) for a in legal_actions}
        else:
            strategy = {a: v / total for a, v in strategy.items()}

        choice = random.choices(list(strategy.keys()), weights=strategy.values(), k=1)[0]

        if choice == "fold":
            return "fold", 0
        elif choice == "call":
            return "call", [a for a in valid_actions if a["action"] == "call"][0]["amount"]
        elif choice == "raise":
            raise_info = [a for a in valid_actions if a["action"] == "raise"][0]["amount"]
            return "raise", int(raise_info["max"]) if isinstance(raise_info, dict) else int(raise_info)

# Define a simple random opponent
class RandomPlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        action = random.choice(valid_actions)
        act_name = action["action"]
        amt = action["amount"]
        if act_name == "raise":
            if isinstance(amt, dict):
                min_raise = int(amt["min"])
                max_raise = int(amt["max"])
                if min_raise >= max_raise:
                    amt = min_raise
                else:
                    amt = random.randint(min_raise, max_raise)
            else:
                amt = int(amt)
        else:
            amt = int(amt)
        return act_name, amt

    def receive_game_start_message(self, game_info):pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, new_action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass


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