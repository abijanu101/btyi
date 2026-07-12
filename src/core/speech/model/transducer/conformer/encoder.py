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
        st = time.time()
        X = X + 1/2 * self.ffnn1(X)
        ed = time.time()
        print(f'\t\tFFNN1 Completion Time: {ed - st}')

        st = time.time()
        X = self.mhsa(X)
        ed = time.time()
        print(f'\t\tMHSA Completion Time: {ed - st}')
        
        st = time.time()
        X = self.conv(X)
        ed = time.time()
        print(f'\t\tConv Completion Time: {ed - st}')
        
        st = time.time()
        X = X + 1/2 * self.ffnn2(X)
        ed = time.time()
        print(f'\t\tFFNN2 Completion Time: {ed - st}')
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
        st = time.time()
        subsampled = self.subsampler(X)
        ed = time.time()
        print(f'\tSubsampler Completion Time: {ed - st}')
        
        print('Conformer Run Started:')
        st = time.time()
        conformed = self.conformers(subsampled)
        ed = time.time()
        print(f'\tConformer Complete Completion Time: {ed - st}')
        
        st = time.time()
        projected = self.project(conformed)
        ed = time.time()
        print(f'\tConformer Projection Time: {ed - st}')
        return projected