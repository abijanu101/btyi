
import torch

import src.core.speech.config as conf

from typing import List, Tuple

LSTMState = Tuple[torch.Tensor, torch.Tensor]

class Hypothesis:
    def __init__(
            self,
            t:int, u:int,

            tokens:List[int],
            score:float,            # log probability of hypothesis

            hidden:LSTMState|None = None
        ):
        'Scores Expected to be in Log-space'
        self.t = t
        self.u = u

        self.tokens = tokens
        self.score = score
        
        self.hidden = hidden

    def extend(
            self,
            logprobs:torch.Tensor,
            new_hidden:LSTMState
        ) -> List[Hypothesis]:
        'Produces all possible hypothesis extensions'

        top_k = logprobs.topk(conf.BEAM_WIDTH)  # will never need to explore beyond beam width
        indices = top_k.indices.squeeze(0)
        values = top_k.values.squeeze(0)

        return [
            Hypothesis(
                self.t, 
                self.u + 1,
                self.tokens + [token.item()],
                self.score + score.item(),
                new_hidden
            )
            if token.item() != conf.BLANK_IDX else
            Hypothesis(
                self.t + 1, 
                self.u,
                self.tokens,
                self.score + score.item(),
                self.hidden    # old pred hidden, no prednet step when only t increases
            )
            for token, score in zip(indices, values)
        ]

    def __gt__(self, other) -> bool:
        return self.score > other.score
    def __lt__(self, other) -> bool:
        return self.score < other.score
    def __eq__(self, other) -> bool:
        return self.score == other.score

    def __hash__(self) -> int:
        return hash(tuple([self.t, self.u] + self.tokens))
