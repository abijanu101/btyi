import torch
from typing import Tuple
import src.core.speech.config as conf

from .ctc import CTCTrainer
from .transducer import TransducerTrainer
from .model import ASRModel

class ASRTrainer:
    'Training Logic and Shi'
    
    def __init__(self, model: ASRModel):
        super().__init__()
        self.ctc_trainer = CTCTrainer(model.ctc)
        self.transducer_trainer = TransducerTrainer(model.transducer)

    def step_together(self, X:torch.Tensor, y:torch.Tensor, len_x: torch.Tensor, len_y:torch.Tensor) -> None:
        'Deep Fusion-esque Joint Training'
        self.ctc_trainer.zero_grad()
        self.transducer_trainer.zero_grad()
        
        ctc_loss, ctc_logprobs = self.ctc_trainer.forward(X, y, len_x, len_y)
        transducer_loss, transducer_logits = self.transducer_trainer.forward(X, y, ctc_logprobs, len_x, len_y)

        loss = (
            conf.JOINTLOSS_CTC_FACTOR * ctc_loss +
            conf.JOINTLOSS_TRANSDUCER_FACTOR * transducer_loss
        )
        
        print(loss)
        loss.backward()

        self.ctc_trainer.step()
        self.transducer_trainer.step()

    def step_both(self, X: torch.Tensor, y:torch.Tensor, len_x:torch.Tensor, len_y:torch.Tensor) -> None:
        'Cold Fusion-esque Joint Training'
        logprobs = self.step_ctc(X, y, len_x, len_y)
        self.step_transducer(X, y, logprobs, len_x, len_y)

    def step_ctc(self, X: torch.Tensor, y:torch.Tensor, len_x:torch.Tensor, len_y:torch.Tensor) -> torch.Tensor:
        'Returns logprobs for potential reuse in training transducer'
        self.ctc_trainer.zero_grad()
        loss, logprobs = self.ctc_trainer.forward(X, y, len_x, len_y)
        loss.backward()
        self.ctc_trainer.step()
        print('CTCNetwork:\t', loss)
        return logprobs
    
    def step_transducer(self, X: torch.Tensor, y_ctc:torch.Tensor, y:torch.Tensor, len_x:torch.Tensor, len_y:torch.Tensor) -> None:
        self.transducer_trainer.zero_grad()
        loss, logprobs = self.transducer_trainer.forward(X, y, y_ctc, len_x, len_y)
        loss.backward()
        self.transducer_trainer.step()
        print('Transducer:\t', loss)
    