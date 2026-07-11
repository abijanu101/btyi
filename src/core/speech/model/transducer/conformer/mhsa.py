import math
import torch

from typing import Tuple
import src.core.speech.config as conf

class RSPE(torch.nn.Module):
    '''A modification in Transformer-XL's Relative Sinusoidal Positional Encoding to support Bidirectional Attention Application'''
    def __init__(self): 
        super().__init__()
        U, self.offset = self._generate()
        self.register_buffer('U', U)

    def fetch_slice(self, l:int) -> torch.Tensor:
        'Get a (T,T,d) matrix, R, that holds the relative positional encoding for i-j at R[i,j]'
        distances = torch.arange(0, l)
        indices = self.offset + (distances[:, None] - distances[None, :])

        return self.U[indices]

    def _generate(self) -> Tuple[torch.Tensor, int]:
        'Generates the Bidirectional Relative Sinusoidal Positional Encoding Table'
        T = conf.CNF_CHUNK_MAX_LEN
        D = conf.CNF_D_MODEL

        U_vanilla = torch.zeros(T, D, dtype=torch.float32)

        for pos in range(T):
            for i in range(D // 2):
                U_vanilla[pos][2*i] = math.sin(pos / 10_000**(2*i / D))
                U_vanilla[pos][2*i + 1] = math.cos(pos / 10_000**(2*i / D))

        U_neg = U_vanilla.clone()       # bidirectionality
        U_neg[:, ::2] *= -1

        U = torch.cat([U_neg[1:].flip(0), U_vanilla], dim=0)
        return U, T-1

class SelfAttentionWithRSPE(torch.nn.Module):
    'The Transformer-XL Self Attention Mechanism modified to support bidirectional attention'
    def __init__(self, posenc:RSPE):
        super().__init__()
        self.rspe = posenc

        self.W_q = torch.nn.Linear(
            in_features=conf.CNF_D_MODEL, 
            out_features=conf.CNF_MHSA_D_HEAD, 
            bias=False
        )
        self.W_kE = torch.nn.Linear(
            in_features=conf.CNF_D_MODEL, 
            out_features=conf.CNF_MHSA_D_HEAD, 
            bias=False
        )
        self.W_v = torch.nn.Linear(
            in_features=conf.CNF_D_MODEL, 
            out_features=conf.CNF_MHSA_D_HEAD, 
            bias=False
        )

        self.u = torch.nn.Parameter(torch.rand(conf.CNF_MHSA_D_HEAD))
        self.v = torch.nn.Parameter(torch.rand(conf.CNF_MHSA_D_HEAD))

        self.W_kR = torch.nn.Linear(
            in_features=conf.CNF_D_MODEL, 
            out_features=conf.CNF_MHSA_D_HEAD, 
            bias=False
        )
        
    def forward(self, X:torch.Tensor) -> torch.Tensor:
        'scaled dot product self attention'
        Q = self.W_q(X)                         # (B, T, d)
        K_E = self.W_kE(X)                      # (B, T, d)
        V = self.W_v(X)                         # (B, T, d)
        
        R = self.rspe.fetch_slice(X.shape[-2])  # (T, T, d_model)
        K_R = self.W_kR(R)                      # (T, T, d)

        # (B, T, d) @ (B, d, T) = (B, T, T)
        A = Q @ K_E.transpose(-2,-1)
        # (B, T, d) and (T, T, d) cant really multiply even if u transpose
        B = torch.einsum(
            'btd,tjd->btj',
            Q, K_R
        )
        # (1, 1, d) @ (B, d, T) = (B, 1, T)
        C = self.u.view(1, 1, -1) @ K_E.transpose(-2,-1)        
        # (d) and (T, T, d) have the same problem
        D = torch.einsum(
            'd,tjd->tj',
            self.v, K_R
        ).unsqueeze(0)

        scaled = (A + B + C + D) / math.sqrt(conf.CNF_MHSA_D_HEAD)        # (B, T, T)
        return torch.softmax(scaled, dim=-1) @ V                            # (B, T, d)
        
class MHSAModule(torch.nn.Module):
    'The Mult-head Self Attention Module for the Conformer as described in the original paper'
    def __init__(self, posenc:RSPE):
        super().__init__()
        self.rspe = posenc
        self.norm = torch.nn.LayerNorm(conf.CNF_D_MODEL)
        self.heads = torch.nn.ModuleList(
            [SelfAttentionWithRSPE(self.rspe) for _ in range(conf.CNF_MHSA_N_HEADS)]
        )
        self.project = torch.nn.Linear(
            in_features=conf.CNF_MHSA_D_HEAD * conf.CNF_MHSA_N_HEADS,
            out_features=conf.CNF_D_MODEL
        )
        self.dropout = torch.nn.Dropout(conf.CNF_MHSA_DROPOUT)

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        normalized = self.norm(X)
        h = torch.cat([head(normalized) for head in self.heads], dim=-1)
        y = self.project(h)
        return self.dropout(y)
