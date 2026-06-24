import torch

from .base import Encoder, Decoder
from typing import List

class SEUPSeq2Seq(torch.nn.Module):
    def __init__(self, embeddings:torch.Tensor, control_stoi:dict, CONTEXT_VECTOR_DIM:int=512, max_length=None):
        super().__init__()
        self.encoder = Encoder(embeddings, CONTEXT_VECTOR_DIM)
        self.decoder = Decoder(embeddings, CONTEXT_VECTOR_DIM)

        self._stoi = control_stoi
        self.max_length = max_length

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self._stoi['<PAD>'])
        self.optim = torch.optim.Adam(self.parameters())


    def forward(self, X:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        '''teacher forced forward pass on (B, S, E) shaped Document-Summary inputs'''
        context_vector = self.encoder(X)
        logits, _ = self.decoder(y, context_vector)
    
        return logits

    def generate(self, X:torch.Tensor) -> List[List[int]]:
        '''autoregressive inference on (B, S, E) shaped Document Batch inputs'''
        BATCH_SIZE = X.shape[0]
        DEVICE = next(self.parameters()).device
      
        h_t = self.encoder(X)
        x_t = torch.full((BATCH_SIZE, 1), self._stoi['<START>'])
        x_t = x_t.to(DEVICE)

        is_complete = torch.zeros(BATCH_SIZE, dtype=bool, device=DEVICE)
        results : List[torch.Tensor] = []

        l = 0
        while not all(is_complete):
            x_t[is_complete] = self._stoi['<PAD>']

            y_t, h_t = self.decoder(x_t, h_t)
            x_t = torch.argmax(y_t, 2, keepdim=False)

            is_complete |= (x_t.squeeze(1) == self._stoi['<END>'])
            results.append(x_t.reshape((-1,)))

            l += 1
            if self.max_length and l > self.max_length:
                break

        return torch.stack(results, 1)


    def fit(self, X:torch.Tensor, y:torch.Tensor) -> float:
        labels = y[:, :-1]
        out = self.forward(X[:, 1:], labels)
        loss = self.loss_fn(out.reshape((-1, out.shape[2])), labels.reshape((-1,)))

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        
        return loss