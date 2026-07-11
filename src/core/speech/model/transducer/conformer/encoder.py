import torch
import src.core.speech.config as conf

from .subsampler import Subsampler
from .ffnn import FFNNModule
from .mhsa import MHSAModule, RSPE
from .conv import ConvModule

class ConformerBlock(torch.nn.Module):
    'A single conformer block as described in the original paper'
    def __init__(self, posenc:RSPE):
        super().__init__()
        self.ffnn1 = FFNNModule()
        self.mhsa = MHSAModule(posenc)
        self.conv = ConvModule()
        self.ffnn2 = FFNNModule()
        self.norm = torch.nn.LayerNorm(conf.CNF_D_MODEL)

    def forward(self, X):
        print('X: ', X.shape)
        X = self.ffnn1(X)
        print('AFTER FFNN: ', X.shape)
        X = self.mhsa(X)
        print('AFTER MHSA: ', X.shape)
        X = self.conv(X)
        print('AFTER CONV: ', X.shape)
        X = self.ffnn2(X)
        print('AFTER FFNN2: ', X.shape)
        return self.norm(X)

class Encoder(torch.nn.Module):
    'For second pass chunk-wise processing of input frames'
    def __init__(self):
        super().__init__()
        self.subsampler = Subsampler()
        self.rspe = RSPE()
        self.conformers = torch.nn.Sequential(
            *[ConformerBlock(self.rspe) for _ in range(conf.CNF_N_BLOCKS)]
        )

    def forward(self, X):
        return self.conformers(self.subsampler(X))
