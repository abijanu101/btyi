import torch

from ..model import ConformerTransducer
from .forward_backward import RNNTLoss

import src.core.speech.config as conf

from typing import Tuple

class TransducerTrainer:
    def __init__(self, model:ConformerTransducer):
        self.loss_fn = RNNTLoss(model)
        self.optim = torch.optim.AdamW(model.parameters())

    def forward(
            self,
            X:torch.Tensor,             # (B, T_raw, M)
            y_ctc:torch.Tensor,         # (B, T_subsampled, V)
            y:torch.Tensor,             # (B, T)
            len_x:torch.Tensor,         # (B,)
            len_y:torch.Tensor          # (B,)
        ) -> torch.Tensor:
        return self.loss_fn(X, y_ctc, y, len_x, len_y)
        
    def zero_grad(self):
        self.optim.zero_grad()
    def step(self):
        self.optim.step()