import torch

import time
from typing import Tuple

import src.core.speech.config as conf

from .conformer import Encoder
from .networks import PredictionNetwork, JointNetwork, LinkNetwork

LSTMState = tuple[torch.Tensor, torch.Tensor]

class ConformerTransducer(torch.nn.Module):
    'An LM-Fused Conformer Transducer modified to incorporate outputs from the first pass model as input'
    
    def __init__(self):
        super().__init__()
        self.conformer = Encoder()
        self.prednet = PredictionNetwork()
        self.linknet = LinkNetwork()
        self.jointnet = JointNetwork()

    def encode(self, X:torch.Tensor) -> torch.Tensor:
        'Takes in Spectrographs, Returns the ConformerEncoder Output'
        return self.conformer(X)

    def predict(
            self,
            X:torch.Tensor,
            hidden:LSTMState|None
        ) -> Tuple[torch.Tensor, LSTMState]:
        'Takes in last predicted token, Returns the PredicitonNetwork Output'
        return self.prednet(X, hidden)

    def link(
            self,
            X:torch.Tensor,
        ) -> torch.Tensor:
        'Takes in CTC Logits, Returns the LinkNetwork Output'
        return self.linknet(X)

    def project(
            self,
            f:torch.Tensor,
            g:torch.Tensor,
            h:torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        'Takes in the outputs of the Encoder, PredNet, and LinkNet; Returns the JointNetwork Output'
        return self.jointnet(f,g,h)

    def project_f(
            self,
            f:torch.Tensor,
        ) -> torch.Tensor:
        return self.jointnet.project_f(f)

    def project_g(
            self,
            g:torch.Tensor,
        ) -> torch.Tensor:
        return self.jointnet.project_g(g)

    def project_h(
            self,
            h:torch.Tensor,
        ) -> torch.Tensor:
        return self.jointnet.project_h(h)
