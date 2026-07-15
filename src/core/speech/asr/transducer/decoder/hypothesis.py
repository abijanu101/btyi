
import torch

import src.core.speech.config as conf

from typing import List, Tuple

LSTMState = Tuple[torch.Tensor, torch.Tensor]

class Hypothesis:
    def __init__(
            self,
            idx:int,
            t:int,
            u:int,

            tokens:List[int],
            score:float = 0.0,  # log probability of hypothesis

            pred_hidden:LSTMState|None = None,
        ):
        'Scores Expected to be in Log-space'

        self.idx = idx
        self.t = t
        self.u = u

        self.tokens = tokens
        self.score = score
        
        self.pred_hidden = pred_hidden


    def extend(
            self,
            logprobs:torch.Tensor,
            pred_hidden:LSTMState
        ) -> List[Hypothesis]:
        'Produces all possible hypothesis extensions'
        return [
            Hypothesis(
                self.idx, 
                self.t, 
                self.u + 1,
                self.tokens + [token],
                self.score + score.item(),
                pred_hidden
            )
            if token != conf.BLANK_IDX else
            Hypothesis(
                self.idx, 
                self.t + 1, 
                self.u,
                self.tokens,
                self.score + score.item(),
                self.pred_hidden    # old pred hidden, no prednet step when only t increases
            )
            for token, score in enumerate(logprobs)
        ]


    def __gt__(self, other) -> bool:
        return self.score > other.score
    def __lt__(self, other) -> bool:
        return self.score < other.score
    # i probably wont be needing __eq__
      
    def __hash__(self) -> int:
        return hash(tuple([self.t, self.u, self.idx] + self.tokens))
