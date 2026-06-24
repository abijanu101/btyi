import torch
import sacrebleu

from src.config.paths import TRAINED_PATH
from src.core.text.vocab import SEUPGloVeFactory
from src.core.models.rnn.seq2seq import SEUPSeq2Seq

from typing import List, Iterable, Callable

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PairDataset(torch.utils.data.Dataset):
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
    def __init__(self, CONTEXT_VECTOR_DIM:int=512, VOCAB_LIMIT:int=None, MAX_SUMM_LENGTH:int=None):
        embeddings, self.text_encoder = SEUPGloVeFactory('840B', 300, VOCAB_LIMIT).construct_embeddings()
        print(f"Persisted {len(embeddings)} tokens into TextEncoderSEUP.")

        self.model = SEUPSeq2Seq(embeddings, self.text_encoder.get_control_stoi(), CONTEXT_VECTOR_DIM, MAX_SUMM_LENGTH)
        print(f"SEUPSeq2Seq initialized with {sum(p.numel() for p in self.model.parameters())} parameters.")

    def _collate_fn(self, batch):
        '''For Training DataLoader'''
        docs, summaries = zip(*batch)
        PAD_IDX = self.text_encoder.get_control_stoi()['<PAD>']

        docs = [torch.tensor(i) for i in docs]
        docs = torch.nn.utils.rnn.pad_sequence(docs, batch_first=True, padding_value=PAD_IDX)
    
        summaries = [torch.tensor(i) for i in summaries]
        summaries = torch.nn.utils.rnn.pad_sequence(summaries, batch_first=True, padding_value=PAD_IDX)
    
        return docs, summaries


    def use_gpu(self) -> None:
        self.model = self.model.to(DEVICE)

    def fit(self, X:Iterable, y:Iterable, epochs:int=10, BATCH_SIZE:int=32) -> None:
        n = len(y)
        assert len(X) == n

        ds = PairDataset(X, y, self.text_encoder.encode)
        dl = torch.utils.data.DataLoader(ds, BATCH_SIZE, shuffle=True, collate_fn=self._collate_fn)
    
        for epoch in range(epochs):
            batch = 1
            for X, y in dl:
                X=X.to(DEVICE)
                y=y.to(DEVICE)
                
                print(f'EP{epoch}\tBT{batch} \t{self.model.fit(X, y)}')
                batch+=1


    def evaluate(self, X:Iterable[str], y:Iterable[str], BATCH_SIZE:int=32) -> List[str]:
        ds = PairDataset(X, y, self.text_encoder.encode)
        dl = torch.utils.data.DataLoader(ds, BATCH_SIZE, shuffle=True, collate_fn=self._collate_fn)

        
        for X, y in dl:
            X = X.to(DEVICE)
            pred = self.model.generate(X)

            y_hat = self.text_encoder.decode_batch(pred)
            y_true = self.text_encoder.decode_batch(y)

            bleu = sacrebleu.corpus_bleu(y_hat, [y_true]) 
            print(bleu)
        

    def predict(self, document:str) -> str:
        tokenized = self.text_encoder.encode(document)
        x = torch.tensor(tokenized).reshape((1, len(tokenized))).to(DEVICE)
        y = self.model.generate(x)[0]
        return self.text_encoder.decode(y)
