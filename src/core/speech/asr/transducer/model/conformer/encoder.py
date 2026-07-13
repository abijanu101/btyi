import torch
import src.core.speech.config as conf

from .subsampler import Subsampler
from .ffnn import FFNNModule
from .mhsa import MHSAModule, SinusoidalPositionalEncoding
from .conv import ConvModule

import time

class ConformerBlock(torch.nn.Module):
    'A single conformer block as described in the original paper'
    def __init__(self, posenc:SinusoidalPositionalEncoding):
        super().__init__()
        self.ffnn1 = FFNNModule()
        self.mhsa = MHSAModule(posenc)
        self.conv = ConvModule()
        self.ffnn2 = FFNNModule()
        self.norm = torch.nn.LayerNorm(conf.CNF_D_MODEL)

    def forward(self, X):
        X = X + 1/2 * self.ffnn1(X)
        X = self.mhsa(X)
        X = self.conv(X)
        X = X + 1/2 * self.ffnn2(X)
        return self.norm(X)

class Encoder(torch.nn.Module):
    'For second pass chunk-wise processing of input frames'
    def __init__(self):
        super().__init__()
        self.subsampler = Subsampler()
        self.pe = SinusoidalPositionalEncoding()
        self.conformers = torch.nn.Sequential(
            *[ConformerBlock(self.pe) for _ in range(conf.CNF_N_BLOCKS)]
        )
        self.project = torch.nn.Linear(
            in_features=conf.CNF_D_MODEL,
            out_features=conf.CNF_OUT_SIZE
        )

    def forward(self, X):
        subsampled = self.subsampler(X)
        conformed = self.conformers(subsampled)
        projected = self.project(conformed)
        return projected