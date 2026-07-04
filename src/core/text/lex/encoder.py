import torch
from typing import List, Tuple

class Encoder(torch.nn.Module):
    def __init__(self, n_vocab, d_model, n_heads, dim_feedforward, n_encoders=3):
        super().__init__()

        self.emb = torch.nn.Embedding(n_vocab, d_model)
        self.d_model = d_model

        enc = torch.nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, norm_first=True)
        self.enc = torch.nn.TransformerEncoder(enc, n_encoders)

    def forward(self, chunk:List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        DEVICE = next(self.parameters()).device

        context = torch.zeros(self.d_model).to(DEVICE)
        results = []
        pos_ids = []
        neg_ids = []

        for sentence, pos_id, neg_id in chunk:
            sentence = sentence.to(DEVICE)
            context = context.unsqueeze(0)
            inp = torch.cat([context, self.emb(sentence)]).to(DEVICE)
            context = self.enc(inp).mean(axis=0)

            results.append(context)
            
            pos_ids.append(pos_id)
            neg_ids.append(neg_id)

        anchors = torch.stack(results)
        positives = torch.stack([results[i] for i in pos_ids])
        negatives = torch.stack([results[i] for i in neg_ids])
        
        return (anchors, positives, negatives)
    
    def predict(self, chunk:List[torch.Tensor]) -> torch.Tensor:
        DEVICE = next(self.parameters()).device

        context = torch.zeros(self.d_model).to(DEVICE)
        results = []

        for sentence in chunk:
            sentence = sentence.to(DEVICE)
            context = context.unsqueeze(0)
            inp = torch.cat([context, self.emb(sentence)]).to(DEVICE)
            context = self.enc(inp).mean(axis=0)

            results.append(context)
            
        return torch.stack(results)
