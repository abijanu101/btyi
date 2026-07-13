import heapq
import torch

import src.core.speech.config as conf
from src.core.speech.asr.transducer import ConformerTransducer

from typing import List, Tuple

class ConformerTransducerBeamSearchDecoder:
    'FOR UNBATCHED DATA ONLY, EXPECTING SHAPE (1,T,D)'

    def __init__(self, model:ConformerTransducer):
        self.model = model

    def decode(self, X:torch.Tensor, y_ctc: torch.Tensor) -> List[List[int]]:
        t = 0   # encoder timestep
        u = 0   # otuput length

        completed = []
        queue = []
        
        f = self.model.encode(X)


        pass