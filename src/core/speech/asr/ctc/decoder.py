import torch
from .model import CTCNetwork

from typing import Tuple
import src.core.speech.config as conf

class CTCNetworkGreedyDecoder:
    def __init__(self, model:CTCNetwork):
        self.model = model
    
    def decode(self, X:torch.Tensor, hidden:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        'Returns the logprobs'

        logits, hidden = self.model(X, hidden)
        logprobs = torch.log_softmax(logits, dim=-1)
        return logprobs, hidden