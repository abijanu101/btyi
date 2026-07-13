import torch

from ..model import ConformerTransducer
import src.core.speech.config as conf

from typing import Tuple

class TransducerTrainer:
    def __init__(self, model:ConformerTransducer):
        self.model = model
        self.optim = torch.optim.AdamW(self.model.parameters())

    def forward(
            self,
            X:torch.Tensor,             # (B,T,M)
            y_ctc:torch.Tensor,         # (B,T,V)
            y:torch.Tensor,             # (B,T)
            len_x:torch.Tensor,         # (B,T)
            len_y:torch.Tensor          # (B,T)
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        f = self.model.encode(X)
        h = self.model.link(y_ctc)
        print(f.shape, h.shape)

        


        # return loss, y_hat


    def zero_grad(self):
        self.optim.zero_grad()
    def step(self):
        self.optim.step()