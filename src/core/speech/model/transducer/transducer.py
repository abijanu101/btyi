import torch

import time
from typing import Tuple

import src.core.speech.config as conf

from .conformer import Encoder
from .networks import PredictionNetwork, JointNetwork, LinkNetwork
from .lm import LanguageModel

class ConformerTransducer(torch.nn.Module):
    'An LM-Fused Conformer Transducer modified to incorporate outputs from the first pass model as input'
    
    def __init__(self):
        super().__init__()
        self.conformer = Encoder()
        self.prednet = PredictionNetwork()
        self.linknet = LinkNetwork()
        self.jointnet = JointNetwork()
        self.lm = LanguageModel()
        
    def forward(
            self, 
            X:torch.Tensor, 
            link_X:torch.Tensor,
            last_out:torch.Tensor,      # (B,) index array
            prednet_hidden: Tuple|None = None,
            linknet_hidden: Tuple|None = None,
        ) -> Tuple[torch.Tensor, Tuple, Tuple]:
        'Returns (logits, prednet_hidden, linknet_hidden)'

        if last_out is None:
            last_out = conf.BLANK_IDX

        st = time.time()
        f = self.conformer(X)
        ed = time.time()
        print(f'Conformer Completion Time: {ed - st}')
        
        st = time.time()
        g, prednet_hidden = self.prednet(last_out, prednet_hidden)
        ed = time.time()
        print(f'Prednet Completion Time: {ed - st}')
        
        st = time.time()
        h, linknet_hidden = self.linknet(link_X, linknet_hidden)
        ed = time.time()
        print(f'Linknet Completion Time: {ed - st}')

        print(f.shape, g.shape, h.shape)

        st = time.time()
        logits = self.jointnet(torch.cat([f[:, 0, :], g, h[:, 0, :]], dim =-1)) # for temporary speed evaluation obv incorrect
        ed = time.time()
        print(f'Jointnet Completion Time: {ed - st}')
        
        return logits, prednet_hidden, linknet_hidden
    