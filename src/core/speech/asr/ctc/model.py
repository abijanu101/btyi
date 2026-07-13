import torch
from typing import Tuple
import src.core.speech.config as conf

import time

class CTCNetwork(torch.nn.Module):
    'First pass naive predictor for instant feedback'
    
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.LSTM(
            input_size=conf.CTC_IN_SIZE,
            hidden_size=conf.CTC_H_SIZE,
            num_layers=conf.CTC_N_LAYERS,
            bias=True,
            batch_first=True,
            dropout=conf.CTC_DROPOUT,
            bidirectional=False
        )
        self.linear = torch.nn.Linear(
            in_features=conf.CTC_H_SIZE,
            out_features=conf.CTC_OUT_SIZE,
            bias=True
        )

    def forward(self, X:torch.Tensor, hidden:Tuple|None) -> torch.Tensor:
        '[Batch, Time, Mel bins] -> ([Batch, Time, Logits], (h_t, c_t))'
        o, hidden = self.rnn(X, hidden)
        result = self.linear(o)
        return result, hidden