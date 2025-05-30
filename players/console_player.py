from pypokerengine.players import BasePokerPlayer

class ConsolePlayer(BasePokerPlayer):  # Do not forget to make parent class as "BasePokerPlayer"
    def __init__(self, name="ConsolePlayer"):
        self.name=name
    
    #  we define the logic to make an action through this method. (so this method would be the core of your AI)
    def declare_action(self, valid_actions, hole_card, round_state):
        
        '''
        valid_actions = [
            { "action" : "fold" , "amount" : 0 },
            { "action" : "call" , "amount" : agree_amount},
            { "action" : "raise", "amount" : raise_to_amount}
        ]
        '''
        print(f"\n==== console player: {self.name} acting ====")
        print(valid_actions)
        print(round_state)
        print(hole_card)
        for action in valid_actions:
            if action["action"]=="call": call_amount=action["amount"]
            elif action["action"]=="raise": raise_amount=action["amount"]
        
        action = ""
        amount = -1
        while True:
            op = input("choose an action: ")
            if op=='r':
                if raise_amount<0: print("invalid raise")
                else:
                    action="raise"
                    amount=raise_amount
                    break
            elif op=='c':
                action="call"
                amount=call_amount
                break
            elif op=='f':
                action="fold"
                amount=0
                break
            else: print("invalid action")
        
        print("=============================================\n")
        return action, amount

    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass

def setup_ai():
    return ConsolePlayer()
