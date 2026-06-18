import numpy as np
import pandas as pd

import torch
import torchtext

from abc import ABC, abstractmethod
from src.config.paths import DATA_PATH, PRETRAINED_PATH

    
class _Embeddings(ABC):
    def __init__(self):
        '''Define a max size for your vocab'''
        self.N_DIM = -1

    @abstractmethod
    def __getitem__(self, word:str):
        '''User of the interface is responsible for ensuring it is already tokenized'''


class GloVeEmbeddings(_Embeddings):
    def __init__(self, N_LIMIT:int=None):
        super().__init__()

        self.N_DIM=300

        # loading the embeddings
        gloVe = torchtext.vocab.GloVe(cache=PRETRAINED_PATH)
        words = gloVe.itos
        vectors  = gloVe.vectors

        # control token embedding initizialization
        control_tokens = ['<s>', '</s>', '<unk>', '<pad>']

        # merging and limit enforcing
        words = control_tokens + words
        vectors = torch.cat(
            [
                torch.rand(len(control_tokens) * self.N_DIM).reshape((-1, self.N_DIM)),
                vectors
            ]
        )

        N_VOCAB = len(words) if N_LIMIT is None or len(words) <= N_LIMIT else N_LIMIT
        self.itos = words[:N_VOCAB]
        self.stoi = {v:k for k,v in enumerate(self.itos)}
        self.embeddings = torch.tensor(np.stack(vectors[:N_VOCAB]))


    def get_vocab(self) -> list:
        return self.itos
    
    def __getitem__(self, key:str) -> torch.Tensor:
        if key not in self.itos:
            key = '<unk>'
        return self.embeddings[self.stoi[key]]