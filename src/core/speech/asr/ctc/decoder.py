import torch
from .model import CTCNetwork

from typing import List
import src.core.speech.config as conf

class CTCNetworkGreedyDecoder:
    def __init__(self, model:CTCNetwork):
        self.model = model
    
    def decode(self, X:torch.Tensor) -> List[List[int]]:
        y = self.model(X)
        y = y.argmax(dim=-1)
        return y