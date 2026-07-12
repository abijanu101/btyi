import numpy as np
import pandas as pd
import torch

from tokenizers import ByteLevelBPETokenizer
from typing import Iterable, List, Tuple, Callable

from src.config.paths import BILINGUAL_PATH
from src.core.speech.config import SAMPLING_RATE, N_VOCAB

from .data import ASRDataset, collate_fn, log_mel_spectrogram
from .model import ASRModel, ASRTrainer

class BiASR:
    'This Urdu-English ASR System is Bisexual'

    def __init__(self):
        self.tokenizer = ByteLevelBPETokenizer()
        self.tokenizer.train(
            files=[(BILINGUAL_PATH / 'flattend_corpora.txt').__str__()],
            vocab_size= N_VOCAB
        )
        self.model = ASRModel()
        self.trainer = ASRTrainer(self.model)


    def train_single(self, df:pd.DataFrame, batch_size:int, epochs:int):
        'Train on a single corpus'
        self.model.train()

        ds = ASRDataset(df, self._tokenize, True)
        dl = torch.utils.data.DataLoader(ds, batch_size, shuffle=True, collate_fn=collate_fn)

        print('Dataloader instantiated')
        # for i in range(epochs):
            # for X, lens, y in dl:
        
        X, senlen, y = next(dl.__iter__())
        print(self.model(X))


    def train_round_robin(self, df1:pd.DataFrame, df2:pd.DataFrame, batch_size1:int, batch_size2:int, epochs:int):
        'Train on two corpora, drawing one batch from df1 and another from df2'
        self.model.train()

        ds1 = ASRDataset(df1, self._tokenize, True)
        ds2 = ASRDataset(df2, self._tokenize, True)

        dl1 = torch.utils.data.DataLoader(ds1, batch_size1, shuffle=True, collate_fn=collate_fn)
        dl2 = torch.utils.data.DataLoader(ds2, batch_size2, shuffle=True, collate_fn=collate_fn)



    def evaluate(self, df:pd.DataFrame, batch_size:int):
        'Evaluate WER on a provided corpus'
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