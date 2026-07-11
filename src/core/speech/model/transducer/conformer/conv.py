import torch
import src.core.speech.config as conf

class ConvModule(torch.nn.Module):
    'The Convolution Module for the Conformer as described in the original paper'
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.LayerNorm(conf.CNF_D_MODEL)
        self.model = torch.nn.Sequential(
            #(B, D, T)
            torch.nn.Conv1d(
                in_channels=conf.CNF_D_MODEL,
                out_channels=conf.CNF_CONV_N_FILTERS,
                kernel_size=1,              # point-wise Convolution i.e., Kernel Size of 1
                stride=1,
                padding=0,
                groups=1,
                padding_mode='zeros'
            ),
            #(B, F, T)
            torch.nn.GLU(dim=1),
            #(B, F // 2, T)
            torch.nn.Conv1d(
                in_channels=conf.CNF_CONV_N_FILTERS // 2,
                out_channels=conf.CNF_D_MODEL,
                kernel_size=conf.CNF_CONV_KERNEL_SIZE,
                stride=conf.CNF_CONV_STRIDE,
                padding=conf.CNF_CONV_PAD,
                groups=conf.CNF_D_MODEL,    # depthwise convolution
                padding_mode='zeros'
            ),
            #(B, D, T)
            torch.nn.BatchNorm1d(conf.CNF_D_MODEL),
            torch.nn.SiLU(),
            torch.nn.Conv1d(
                in_channels=conf.CNF_D_MODEL,
                out_channels=conf.CNF_D_MODEL,
                kernel_size=1,              # point-wise Convolution i.e., Kernel Size of 1
                stride=1,
                padding=0,
                groups=1,
                padding_mode='zeros'
            ),
            torch.nn.Dropout(conf.CNF_CONV_DROPOUT)
        )


    def forward(self, X:torch.Tensor) -> torch.Tensor:
        '(B, T, D)'
        X = self.norm(X).transpose(1,2)         # (B,D,T)
        return self.model(X).transpose(1,2)
