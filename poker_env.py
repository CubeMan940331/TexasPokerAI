from phevaluator.evaluator import evaluate_cards
from phevaluator.card import Card
import random

class PokerStreet:
    init:int = 0
    preflop:int = 1
    flop:int = 2
    turn:int = 3
    river:int = 4
    finish:int = 5

class PokerAction:
    a_fold:int = 0
    a_call:int = 1
    a_raise:int = 2

class PokerPlayer:
    def __init__(self, init_stack:int):
        self.stack:int = init_stack
        self.paid:int = 0
        self.is_agree:bool = False
        self.is_fold:bool = False
        self.is_live:bool = True
        self.hand_rank:int = -1

    def pay(self, amount:int):
        assert amount<=self.stack, "Player not enough money"
        self.stack-=amount
        self.paid+=amount
    
    @property
    def total(self):
        return self.stack+self.paid

class PokerEnv:
    def __init__(self, small_blind, init_stack):
        # fixed rule
        self.init_stack:int = init_stack
        self.small_blind:int = small_blind

        # player state
        self.players:list[PokerPlayer] = [
            PokerPlayer(init_stack),
            PokerPlayer(init_stack)
        ]

        # cards
        self.cards:list[int] = [i for i in range(52)]

        # round
        self.round:int = 0
        self.street:int = PokerStreet.init
        self.is_betting:bool = False
        self.agree_amount:int = 0
        
        self.act_idx:int = 0
        self.btn_idx:int = 0
        self.raise_cnt:int = 0
        
        self.is_end:bool = False
    
    
    def print(self):
        print("==== print game state ====")
        print("round:", self.round, "street:", self.street, "is_betting:", self.is_betting)
        print("btn_idx:",self.btn_idx, "act_idx:", self.act_idx, "raise_cnt:", self.raise_cnt)
        print("is_end:",self.is_end)
        print("shared cards:", list(map(str,map(Card,self.shared_cards()))))
        print("player info:")
        for i in range(2):
            print("pos:",i)
            print("\tplayer cards:", list(map(str,map(Card,self.player_cards(i)))))
            print(f"\t{vars(self.players[i])}")
        print("==========================")
    
    def pot(self) -> int:
        return sum(p.paid for p in self.players)
    def shared_cards(self) -> list:
        if self.street==PokerStreet.river or self.street==PokerStreet.finish:
            return self.cards[:5]
        if self.street==PokerStreet.turn:
            return self.cards[:4]
        if self.street==PokerStreet.flop:
            return self.cards[:3]
        return []    
    def player_cards(self, idx:int) -> list:
        if self.street!=PokerStreet.init:
            idx = 5 + idx*2
            return self.cards[idx:idx+2]
        return []
    def raise_to(self) -> int:
        if self.street==PokerStreet.preflop or self.street==PokerStreet.flop:
            return self.agree_amount+self.small_blind*2
        elif self.street==PokerStreet.turn or self.street==PokerStreet.river:
            return self.agree_amount+self.small_blind*4
        return -1
    def valid_action(self) -> list[bool]:
        '''
        return a list of bool that indicate if the action is valid
        '''
        if (
            (not self.is_betting) or
            self.street==PokerStreet.init or
            self.street==PokerStreet.finish
        ): return [False, False, False]
        # fold and call are always valid
        result = [True, True, True]

        if(# the player doesn't have enough money
            (self.raise_to() > self.players[self.act_idx].total)
            or ( # exceed raise_cnt limit
                (self.street==PokerStreet.preflop or self.street==PokerStreet.flop)
                and self.raise_cnt>=3
            ) 
            or ( # exceed raise_cnt limit
                (self.street==PokerStreet.turn or self.street==PokerStreet.river)
                and self.raise_cnt>=4
            )
        ): result[PokerAction.a_raise]=False

        return result
        
    def next_state(self, action:int|None):
        # end
        if self.is_end: return

        # betting
        if self.is_betting:
            assert self.valid_action()[action], "not a valid action"
            # apply the action
            if action==PokerAction.a_fold:
                self.players[self.act_idx].is_fold=True
                
                # goto finish
                self.is_betting=False
                self.street=PokerStreet.finish
                return
            
            elif action==PokerAction.a_call:
                self.players[self.act_idx].is_agree=True
                
                pay_amount = min(
                    self.agree_amount - self.players[self.act_idx].paid,
                    self.players[self.act_idx].stack
                )
                self.players[self.act_idx].pay(pay_amount)
            
            elif action==PokerAction.a_raise:
                for i in range(2):
                    self.players[i].is_agree=False
                
                raise_to = self.raise_to()
                self.agree_amount = raise_to
                pay_amount = raise_to-self.players[self.act_idx].paid
                self.players[self.act_idx].pay(pay_amount)
                
                self.raise_cnt+=1
            
            # if all agree
            all_agree=True
            for p in self.players:
                if not p.is_live: continue
                all_agree = all_agree and (p.is_agree or p.is_fold)
            
            if all_agree:
                # goto next street
                self.is_betting=False
                self.street+=1
                return
            
            # goto betting
            # find the next player
            self.act_idx=(self.act_idx+1)%2
            return
            
        # init
        if self.street==PokerStreet.init:
            live_player_cnt = 0
            for  p in self.players:
                if p.stack>0: live_player_cnt+=1
            if live_player_cnt<2:
                # goto end state
                self.is_end=True
                return
            # goto preflop
            self.round+=1

            # shuffle cards
            random.shuffle(self.cards)
            # pay small blind
            pay_amount = min(self.small_blind, self.players[self.btn_idx].paid)
            self.players[self.btn_idx].pay(pay_amount)
            # pay big blind
            pay_amount = min(self.small_blind*2, self.players[(self.btn_idx+1)%2].paid)
            self.players[(self.btn_idx+1)%2].pay(pay_amount)
            # set agree amount
            self.agree_amount=self.small_blind*2            
            for i in range(2): self.players[i].is_fold=False

            self.street=PokerStreet.preflop
            return
        
        # preflop flop turn river
        if(
            self.street==PokerStreet.preflop or
            self.street==PokerStreet.flop or
            self.street==PokerStreet.turn or
            self.street==PokerStreet.river
        ):
            acting_player = 0
            for i in range(2):
                if self.players[i].is_live and (not self.players[i].is_fold):
                    acting_player+=1
            
            # goto next street
            if acting_player<2:
                self.street+=1
                return

            # goto betting
            if self.street==PokerStreet.preflop:
                self.act_idx = self.btn_idx
            else:
                self.act_idx = (self.btn_idx+1)%2
            self.is_betting=True
            for i in range(2):
                self.players[i].is_agree=False
            self.raise_cnt=0
            return
        if self.street==PokerStreet.finish:
            # move btn
            self.btn_idx = (self.btn_idx+1)%2

            # transfer money
            for i in range(2):
                if (not self.players[i].is_live) or self.players[i].is_fold:
                    self.players[i].hand_rank=-1
                    continue
                self.players[i].hand_rank=evaluate_cards(
                    *(self.player_cards(i)+self.shared_cards())
                )
            min_rank = min((p.hand_rank for p in self.players))
            winners_idx = [
                idx for idx, p in enumerate(self.players) if p.hand_rank==min_rank
            ]
            shared_amount = self.pot()//len(winners_idx)
            remain_amount = self.pot()%len(winners_idx)
            assert remain_amount==0, "expeted no remaining"
            for i in range(2): self.players[i].paid=0
            for i in winners_idx:
                self.players[i].stack+=shared_amount

            # update is_live
            for i in range(2):
                self.players[i].is_live = self.players[i].stack>0
            
            # goto init
            self.street=PokerStreet.init
            return

def main():
    env=PokerEnv(1,100)
    env.print()
    while not env.is_end:
        a = None
        if env.is_betting:
            print(env.valid_action())
            a = int(input("choose an action: "))
        env.next_state(a)
        env.print()

if __name__=='__main__':
    main()
