import random
import torch
from engine.state_encoder import encode_state
from pypokerengine.players import BasePokerPlayer
from ai.strategy_net import StrategyNet

class StrategyPlayer(BasePokerPlayer):
    def __init__(self, model, name="StrategyAI"):
        self.model = model
        self.name = name
        self.uuid = None
    
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, new_action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass
    def receive_game_start_message(self, game_info): pass

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

# Load the trained strategy_net
strategy_net = StrategyNet(input_dim=113)
strategy_net.load_state_dict(torch.load("models/strategy_net.pt"))
strategy_net.eval()

def setup_ai():
    return StrategyPlayer(strategy_net)
