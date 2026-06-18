import torch
import torchtext

from .vocab import SEUPGloVeFactory
from typing import Iterable, Callable


class SummarizerDataset(torch.utils.data.Dataset):
    def __init__(self, X: Iterable, y:Iterable, encode_fn:Callable):
        assert len(X) == len(y)

        self.X = X
        self.y = y

        self.encode_fn = encode_fn

    def __getitem__(self, idx:int) -> list:
        return self.encode_fn(self.X[idx]), self.encode_fn(self.y[idx])

    def __len__(self) -> int:
        return len(self.y)


class BasicEnglishSummarizer:
    def __init__(self):
        embeddings, self.text_encoder = SEUPGloVeFactory().construct_embeddings()
        self.model = 'not implemented yet'
        pass

    def fit(self, X:Iterable, y:Iterable) -> None:
        n = len(y)
        assert len(X) == n

        # to be parameterized later        
        BATCH_SIZE=32
        ds = SummarizerDataset(X, y, self.text_encoder.encode)
        return torch.utils.data.DataLoader(ds, BATCH_SIZE, shuffle=True, collate_fn=self._collate_fn)



    def _collate_fn(self, batch):
        docs, summaries = zip(*batch)

        docs = [torch.tensor(i) for i in docs]
        docs = torch.nn.utils.rnn.pad_sequence(docs, batch_first=True, padding_value=self.text_encoder._pad)
    
        summaries = [torch.tensor(i) for i in summaries]
        summaries = torch.nn.utils.rnn.pad_sequence(summaries, batch_first=True, padding_value=self.text_encoder._pad)
    
        return docs, summaries
    

    def predict(self, X) -> None:
        pass