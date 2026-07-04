import pandas as pd
import torch

from typing import List

from src.config.paths import TRAINED_PATH
from src.config.lex import N_VOCAB, N_ENCODERS, N_HEADS, D_MODEL, D_FEEDFORWARD, LRATE

from src.core.text.lex.encoder import Encoder
from src.core.text.lex.dataset import LexDataset


class SentenceEmbeddingModel:
    def __init__(self):
        self.model = Encoder(N_VOCAB, D_MODEL, N_HEADS, D_FEEDFORWARD, N_ENCODERS).to('cuda')   
        self.optim = torch.optim.Adam(self.model.parameters(), LRATE)
        self.loss_fn = torch.nn.TripletMarginLoss()
        self.epochs = 0 # true trained epochs

    def fit(self, ds:LexDataset, epochs:int) -> None:
        dl = torch.utils.data.DataLoader(ds, batch_size=None, shuffle=True)
        
        n_chunks = len(ds)

        self.model.train()
        for epoch in range(epochs):
            chunk_num = 0
            print(f"Epoch {self.epochs}")
            for chunk in dl:
                self.optim.zero_grad()

                anchors, positives, negatives = self.model(chunk)
                loss = self.loss_fn(anchors, positives, negatives)

                chunk_num += 1

                if chunk_num % 5 == 0:
                    print(f'[{chunk_num}/{n_chunks}]', loss)

                loss.backward()
                self.optim.step()
            self.save()
            self.epochs += 1

    def predict(self, input:List[torch.Tensor]):
        self.model.eval()
        with torch.no_grad():
            return self.model.predict(input)
    
    def save(self) -> None:
        model_dir = TRAINED_PATH / 'lex' / 'model.pt'
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optim.state_dict(),
                'epochs' : self.epochs
            }, model_dir.__str__()
        )
    
    def load(self) -> None:
        model_dir = TRAINED_PATH / 'lex' / 'model.pt'
        checkpoint = torch.load(model_dir.__str__())
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epochs = checkpoint['epochs']
