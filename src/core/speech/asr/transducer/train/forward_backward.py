import torch

from ..model import ConformerTransducer
import src.core.speech.config as conf

from typing import Tuple

class RNNTLoss:
    def __init__(self, model:ConformerTransducer):
        self.model = model

    def __call__(
            self,
            X:torch.Tensor,             # (B, T_raw, M)
            y_ctc:torch.Tensor,         # (B, T_subsampled, V)
            y:torch.Tensor,             # (B, T)
            len_x:torch.Tensor,         # (B,)
            len_y:torch.Tensor          # (B,)
        ) -> torch.Tensor:
        'The forward pass from the RNN-T forward-backward algorithm, but for Batched Samples'
        DEVICE = next(self.model.parameters()).device

        f = self.model.encode(X)
        g = self.model.predict(y)               # teacher forcing
        h = self.model.link(y_ctc)

        f, g, h = self.model.project(f, g, h)   # (B,D) -> (B, V+1)

        B = X.shape[0]
        T = torch.ceil(len_x / 4).max()         # This makes it match the subsampling the encoder does
        U = len_y.max()

        alpha = torch.zeros(
            B,      # Batched Alphas
            U + 1,  # u: Tokens Generated
            T + 1,   # +1 for terminal state
            device=DEVICE
        )
    
        alpha[:, 0, 0] = 1
        # The First U anti-diagonals that include the left most column's entries (excluding the base case u=0, t=0)
        for i in range(1, U+1):
            for j in range(i+1):
                u, t = i-j, j
                alpha[:, u][:, t] = self._alpha(u,t, alpha, y, f,g,h)
        
        # The remaining T anti-diagonals that include the top most row's entries (excluding the primary anti-diagoanal)        
        for i in range(1, T+1):
            u, t = U, i
            while t <= T and u >= 0:
                alpha[:, u][:, t] = self._alpha(u,t, alpha, y, f,g,h)
                t += 1
                u -= 1

        terminals = alpha.gather(
            1,
            len_y.view(-1, 1, -1)
        ).squeeze(1)
        terminals = alpha.gather(
            1,
            len_x.view(-1, 1)
        ).squeeze(1)
        
        losses = terminals
        return losses

    def _alpha(
            self,
            u:int, t:int,
            alpha:torch.Tensor,
            y:torch.Tensor,
            f:torch.Tensor, g:torch.Tensor, h:torch.Tensor
        ) -> torch.Tensor | int:
        '''The recurrence'''
        
        joined = f[:, t] + g[:, u] + h[:, t]
        y_hat = torch.softmax(joined, dim=-1)   # (B, V+1)

        blank_probs = y_hat[:, conf.BLANK_IDX]  # (B,)

        labels = y[:, u].unsqueeze(1)           # (B, 1)
        label_probs = y_hat.gather(
            dim=1,
            index=labels
        ).squeeze(1)                            # (B,)

        return (
            # blank term
            (   
                # probability of having emitted everything necessary already
                alpha[:, u][:, t-1] *
                # probability of emiting blank
                blank_probs
            ) if t != 0 else 0
            +
            # generation term 
            (
                # probability of having emitted everything except the u-th token
                alpha[:, u-1][:, t] *
                # probability of having emitted the u-th token
                label_probs
            ) if u != 0 else 0
        )