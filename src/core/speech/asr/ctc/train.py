import torch

from .model import CTCNetwork
import src.core.speech.config as conf

from typing import Tuple

class CTCTrainer:
    def __init__(self, model:CTCNetwork):
        self.model = model
        self.loss = torch.nn.CTCLoss(conf.BLANK_IDX)
        self.optim = torch.optim.AdamW(self.model.parameters())

    def forward(
            self,
            X:torch.Tensor,
            y:torch.Tensor,
            len_x:torch.Tensor,
            len_y:torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        y_hat, _ = self.model(X, None)
        y_hat = y_hat.log_softmax(dim=-1)

        loss = self.loss(
            y_hat.transpose(0, 1), y,
            len_x, len_y
        )
        return loss, y_hat

    def zero_grad(self):
        self.optim.zero_grad()
    def step(self):
        self.optim.step()