import torch
import torchtext

from typing import Tuple 

class Encoder(torch.nn.Module):
    def __init__(self, embeddings:torch.Tensor, CONTEXT_VECTOR_DIM:int):
        super().__init__()
        self._embeddings = torch.nn.Embedding.from_pretrained(embeddings)
        self._model = torch.nn.RNN(embeddings.shape[1], CONTEXT_VECTOR_DIM, batch_first=True)

    def forward(self, X_t:torch.Tensor) -> torch.Tensor:
        '''batch of documents to encode'''
        I_t = self._embeddings(X_t)
        y_t, h_t =  self._model(I_t)
        return h_t

class Decoder(torch.nn.Module):
    def __init__(self, embeddings:torch.Tensor, CONTEXT_VECTOR_DIM:int):
        super().__init__()
        self._embeddings = torch.nn.Embedding.from_pretrained(embeddings)
        self._rnn = torch.nn.RNN(embeddings.shape[1], CONTEXT_VECTOR_DIM, batch_first=True)
        self._outlin = torch.nn.Linear(CONTEXT_VECTOR_DIM, embeddings.shape[0])

    def forward(self, X_t:torch.Tensor, h_t:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''batch of summaries to generate'''
        I_t = self._embeddings(X_t)
        y_t, h_t = self._rnn(I_t, h_t)
        return self._outlin(y_t), h_t
