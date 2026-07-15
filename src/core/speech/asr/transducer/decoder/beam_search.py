import heapq
import torch

import src.core.speech.config as conf
from src.core.speech.asr.transducer import ConformerTransducer

from typing import List, Tuple

class ConformerTransducerBeamSearchDecoder:
    'FOR UNBATCHED DATA ONLY, EXPECTING SHAPE (1,T,D)'

    def __init__(self, model:ConformerTransducer):
        self.model = model

    def decode(
            self,
            X:torch.Tensor,
            ctc_logprobs:torch.Tensor,
            last_y:torch.Tensor,
            prednet_hidden:torch.Tensor|None
        ) -> List[List[int]]:
        'You will NEVER guess what this does... it applies a standard beam search decoding algorithm.'

        t = 0   # encoder timestep
        u = 0   # otuput length

        completed = []
        queue = []

        print(X.shape, ctc_logprobs.shape, prednet_hidden.shape if prednet_hidden is not None else None)

        raise NotImplementedError()
        # f = self.model.encode(X)
        # h = self.model.encode(prednet_hidden)

        pass