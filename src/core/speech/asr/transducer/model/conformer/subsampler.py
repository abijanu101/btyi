import torch

from typing import List
import src.core.speech.config as conf

class Subsampler(torch.nn.Module):
    '''Subsamples a Log Mel Spectrograph into a (B, T/4, D_MODEL) tensor'''
    def __init__(self):
        super().__init__()

        self.convs = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=conf.SUBSAMPLER_N_FILTERS,
                kernel_size=conf.SUBSAMPLER_KERNEL_SIZE,
                stride=conf.SUBSAMPLER_STRIDE,
                padding=conf.SUBSAMPLER_PAD,
                padding_mode='replicate'
            ), 
            torch.nn.Conv2d(
                in_channels=conf.SUBSAMPLER_N_FILTERS,
                out_channels=conf.SUBSAMPLER_N_FILTERS,
                kernel_size=conf.SUBSAMPLER_KERNEL_SIZE,
                stride=conf.SUBSAMPLER_STRIDE,
                padding=conf.SUBSAMPLER_PAD,
                padding_mode='replicate'
            )
        )

       # rn we got (B, C, T/4, Mel/4)
       # permute in forward pass to go to (B, T/4, C, Mel/4) for convenient flattening

        self.project = torch.nn.Sequential(
            torch.nn.Flatten(2), # (B, T/4, _)
            torch.nn.Linear(
                in_features=conf.N_MELS // 4 * conf.SUBSAMPLER_N_FILTERS,
                out_features=conf.CNF_D_MODEL
            ), 
            torch.nn.Dropout(conf.SUBSAMPLER_PROJ_DROPOUT)
        )
        

    def forward(self, X:torch.Tensor):
        'Expects X to be shape (B, T, Mel)'
        X = X.unsqueeze(1)  # conv expects (B, C, X, Y)
        subsampled = torch.permute(self.convs(X), (0,2,1,3))
        return self.project(subsampled)
    

    def shrinked_lengths(seqlens: List[int]) -> List[int]:
        'Get the shrinked sequence lens after subsampling is applied'
        return [
            (i + 2 * conf.SUBSAMPLER_PAD - conf.SUBSAMPLER_KERNEL_SIZE) // conf.SUBSAMPLER_STRIDE + 1 
            for i in seqlens
        ]
