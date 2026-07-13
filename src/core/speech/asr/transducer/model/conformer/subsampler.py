import torch
import math

from typing import List
import src.core.speech.config as conf

class Subsampler(torch.nn.Module):
    '''Subsamples a Log Mel Spectrograph into a (B, T/4, D_MODEL) tensor'''
    def __init__(self):
        super().__init__()

        self.convs = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=conf.CNF_SUBSAMPLER_N_FILTERS,
                kernel_size=conf.CNF_SUBSAMPLER_KERNEL_SIZE,
                stride=conf.CNF_SUBSAMPLER_STRIDE,
                padding=conf.CNF_SUBSAMPLER_PAD,
                padding_mode='replicate'
            ), 
            torch.nn.Conv2d(
                in_channels=conf.CNF_SUBSAMPLER_N_FILTERS,
                out_channels=conf.CNF_SUBSAMPLER_N_FILTERS,
                kernel_size=conf.CNF_SUBSAMPLER_KERNEL_SIZE,
                stride=conf.CNF_SUBSAMPLER_STRIDE,
                padding=conf.CNF_SUBSAMPLER_PAD,
                padding_mode='replicate'
            )
        )

       # rn we got (B, C, T/4, d/4)
       # permute in forward pass to go to (B, T/4, C, d/4) for convenient flattening

        self.project = torch.nn.Sequential(
            torch.nn.Flatten(2), # (B, d/4, _)
            torch.nn.Linear(
                in_features=math.ceil(conf.N_MELS / 4) * conf.CNF_SUBSAMPLER_N_FILTERS,
                out_features=conf.CNF_D_MODEL
            ), 
            torch.nn.Dropout(conf.CNF_SUBSAMPLER_PROJ_DROPOUT)
        )
        

    def forward(self, X:torch.Tensor):
        'Expects X to be shape (B, T, self.last_dim_size)'
        X = X.unsqueeze(1)                          # conv expects (B, C, T, d)
        X = self.convs(X)                           # (B, N, T/4, d/4)
        subsampled = torch.permute(X, (0,2,1,3))    # (B, T/4, N, d/4)
        return self.project(subsampled)             # (B, T/4, N * d/4)
    

    def shrinked_lengths(seqlens: List[int]) -> List[int]:
        'Get the shrinked sequence lens after subsampling is applied'
        return [
            (i + 2 * conf.SUBSAMPLER_PAD - conf.SUBSAMPLER_KERNEL_SIZE) // conf.SUBSAMPLER_STRIDE + 1 
            for i in seqlens
        ]
