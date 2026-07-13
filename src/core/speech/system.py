import numpy as np
import pandas as pd
import torch

from tokenizers import ByteLevelBPETokenizer
from typing import Iterable, List, Tuple, Callable

from src.config.paths import BILINGUAL_PATH
from src.core.speech.config import SAMPLING_RATE, N_VOCAB

from .data import ASRDataset, collate_fn, log_mel_spectrogram
from .asr import ASRModel, ASRTrainer

class BiASR:
    'This Urdu-English ASR System is Bisexual'

    def __init__(self):
        self.tokenizer = ByteLevelBPETokenizer()
        self.tokenizer.train(
            files=[(BILINGUAL_PATH / 'flattend_corpora.txt').__str__()],
            vocab_size= N_VOCAB
        )
        self.model = ASRModel().to('cuda')
        self.trainer = ASRTrainer(self.model)
        self.n_steps = 0


    def train_single(self, df:pd.DataFrame, batch_size:int, epochs:int, end_to_end:bool = True):
        'Train on a single corpus'
        DEVICE = next(self.model.parameters()).device
        self.model.train()

        ds = ASRDataset(df, self._tokenize, True)
        dl = torch.utils.data.DataLoader(ds, batch_size, shuffle=True, collate_fn=collate_fn)

        for i in range(epochs):
            print(f'[Epoch {i}/{epochs}]  ({self.n_steps} Steps in Total)')
            for X, y, len_x, len_y in dl:
                X = X.to(DEVICE)
                y = y.to(DEVICE)
                len_x = len_x.to(DEVICE)
                len_y = len_y.to(DEVICE)

                self.trainer.step_together(X, y, len_x, len_y) if end_to_end else self.trainer.step_both(X, y, len_x, len_y)
                self.n_steps += 1

    def train_round_robin(self, df1:pd.DataFrame, df2:pd.DataFrame, batch_size1:int, batch_size2:int, n_steps:int, end_to_end:bool = True):
        'Train on two corpora, drawing one batch from df1 and another from df2'
        DEVICE = next(self.model.parameters()).device
        self.model.train()

        ds1 = ASRDataset(df1, self._tokenize, True)
        ds2 = ASRDataset(df2, self._tokenize, True)

        dls = [
            torch.utils.data.DataLoader(ds1, batch_size1, shuffle=True, collate_fn=collate_fn),
            torch.utils.data.DataLoader(ds2, batch_size2, shuffle=True, collate_fn=collate_fn)
        ]
        iters = [iter(dl) for dl in dls]

        X:torch.Tensor
        y:List[List[int]]
        len_x:List[int]
        len_y:List[int]

        for i in range(n_steps):
            print(f'[{i}/{n_steps} Steps] - ({self.n_steps} Steps in Total)')            
            try:
                X, y, len_x, len_y = next(iters[i % 2])
            except StopIteration:
                iters[i % 2] = iter(dls[i % 2])
                X, y, len_x, len_y = next(iters[i % 2])

            X = X.to(DEVICE)
            y = y.to(DEVICE)
            len_x = len_x.to(DEVICE)
            len_y = len_y.to(DEVICE)
            
            self.trainer.step_together(X, y, len_x, len_y) if end_to_end else self.trainer.step_both(X, y, len_x, len_y)
            self.n_steps += 1


    def evaluate(self, df:pd.DataFrame, batch_size:int):
        'Evaluate WER on a provided corpus'
        DEVICE = next(self.model.parameters()).device
        self.model.eval()

        ds = ASRDataset(df, self._tokenize, False)  # no spec aug
        dl = torch.utils.data.DataLoader(ds, batch_size, shuffle=True, collate_fn=collate_fn)


    def start(self):
        'Start streaming ASR system'


    def save(self):
        'Save model state'


    def load(self):
        'Load from presaved'


    def _tokenize(self, text:str):
        return self.tokenizer.encode(text).ids