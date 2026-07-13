import math
import torch

from typing import Tuple
import src.core.speech.config as conf

class SinusoidalPositionalEncoding(torch.nn.Module):
    '''Vanilla Absolute Sinusoidal Positional Encoding modified to facilitate Bidirectionality'''
    def __init__(self): 
        super().__init__()
        U_pos = self._generate()        
        U_neg = U_pos.clone()
        U_neg[:, ::2] *= -1

        self.register_buffer('U_pos', U_pos)
        self.register_buffer('U_neg', U_neg)

    def slice(self, l:int) -> Tuple[torch.Tensor, torch.Tensor]:
        'Instead of (MAX_LEN, d), get a smaller matrix slice (T,d)'
        return self.U_pos[:l], self.U_neg[:l]

    def _generate(self) -> torch.Tensor:
        'Generates the Bidirectional Relative Sinusoidal Positional Encoding Table'
        T = conf.CNF_CHUNK_MAX_LEN
        D = conf.CNF_D_MODEL

        U = torch.zeros(T, D, dtype=torch.float32)
        for pos in range(T):
            for i in range(D // 2):
                U[pos][2*i] = math.sin(pos / 10_000**(2*i / D))
                U[pos][2*i + 1] = math.cos(pos / 10_000**(2*i / D))
        
        return U


class SelfAttentionWithRSPE(torch.nn.Module):
    'The Transformer-XL Self Attention Mechanism modified to support bidirectional attention'
    def __init__(self, posenc:SinusoidalPositionalEncoding):
        super().__init__()
        self.pe = posenc

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
        
        # (B, T, d) @ (B, d, T) = (B, T, T)
        A = Q @ K_E.transpose(-2,-1)
        # (1, 1, d) @ (B, d, T) = (B, 1, T)
        C = self.u.view(1, 1, -1) @ K_E.transpose(-2,-1)        
        
        U_pos, U_neg = self.pe.slice(X.shape[-2])   # (T, d_model)
        K_R_pos = self.W_kR(U_pos)                      # (T, d)
        K_R_neg = self.W_kR(U_neg)                      # (T, d)

        # ((B, T, d) + (d,)) @ (d, T) = (B, T, T)
        BD_tilde_pos = (Q + self.v) @ K_R_pos.T
        BD_tilde_neg = (Q + self.v) @ K_R_neg.T
        BD = self._relative_shift(BD_tilde_pos, BD_tilde_neg)

        scaled = (A + C+ BD) / math.sqrt(conf.CNF_MHSA_D_HEAD)        # (B, T, T)
        return torch.softmax(scaled, dim=-1) @ V                          # (B, T, d)
        
    def _relative_shift(self, BD_tilde_pos:torch.Tensor, BD_tilde_neg:torch.Tensor) -> torch.Tensor:
        'A modification of the method described in Appendix B of the TXL Paper to support bidirectional attention'
        DEVICE = next(self.parameters()).device

        # input shape is (B, T, T)
        B, T, _ = BD_tilde_neg.shape
        
        idx = torch.arange(T, device=DEVICE)

        q = idx[:, None]
        k = idx[None, :]

        delta =  q - k
        pos_idx = delta.clamp(min=0).unsqueeze(0).expand(B, -1, -1)
        neg_idx = (-delta).clamp(min=0).unsqueeze(0).expand(B, -1, -1)

        BD_pos = torch.gather(BD_tilde_pos, dim=-1, index=pos_idx)
        BD_neg = torch.gather(BD_tilde_neg, dim=-1, index=neg_idx)

        BD = torch.where((delta >= 0).unsqueeze(0), BD_pos, BD_neg)
        return BD

class MHSAModule(torch.nn.Module):
    'The Mult-head Self Attention Module for the Conformer as described in the original paper'
    def __init__(self, posenc:SinusoidalPositionalEncoding):
        super().__init__()
        self.pe = posenc
        self.norm = torch.nn.LayerNorm(conf.CNF_D_MODEL)
        self.heads = torch.nn.ModuleList(
            [SelfAttentionWithRSPE(self.pe) for _ in range(conf.CNF_MHSA_N_HEADS)]
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
