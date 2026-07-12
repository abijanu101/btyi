import torch
import src.core.speech.config as conf

class FFNNModule(torch.nn.Module):
    'The Forward-Feeding Neural Network Module for the Conformer as described in the original paper'
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.LayerNorm((conf.CNF_D_MODEL,)),
            torch.nn.Linear(
                in_features=conf.CNF_D_MODEL,
                out_features=conf.CNF_D_MODEL * 4),
            torch.nn.SiLU(),
            torch.nn.Dropout(conf.CNF_FFNN_DROPOUT),
            torch.nn.Linear(
                in_features=conf.CNF_D_MODEL * 4,
                out_features=conf.CNF_D_MODEL),
            torch.nn.Dropout(conf.CNF_FFNN_DROPOUT)
        )

    def forward(self, X):
        return self.model(X)