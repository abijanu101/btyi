import numpy as np
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
        'The forward pass from the RNN-T forward-backward algorithm'
        DEVICE = next(self.model.parameters()).device


        B = X.shape[0]

        # neural modules
        prednet_in = torch.cat([
            torch.tensor([conf.BLANK_IDX] * B, device=DEVICE).unsqueeze(1),
            y
        ], dim=1)

        f = self.model.encode(X)
        g, _ = self.model.predict(prednet_in, None)
        h = self.model.link(y_ctc)

        f, g, h = self.model.project(f, g, h)       # (B,D) -> (B, V+1)
        
        # DP
        T = int(torch.ceil(len_x / 4).max().item())     # match the subsampling in the Encoder and LinkNet
        U = y.shape[1]

        alpha = np.empty((U+1, T+1),dtype=object)       # (U+1, T+1) pointer array to (B,) tensors
        alpha[0, 0] = torch.zeros(B, device=DEVICE)

        # The First U anti-diagonals that include the left most column's entries 
        # (excluding the base case and the primary anti-diagonal)
        for i in range(1, U):
            for j in range(min(i+1, T+1)):
                u, t = i-j, j
                alpha[u, t] = self._alpha(u,t, U,T, alpha, y, f,g,h)

        # The remaining T anti-diagonals that include the top most row's entries        
        for i in range(T+1):
            u, t = U, i
            while t <= T and u >= 0:
                alpha[u, t] = self._alpha(u,t, U,T, alpha, y, f,g,h)
                t += 1
                u -= 1

        # adjust for padding
        sum_logprobs = torch.zeros(1, device=DEVICE)
        for i in range(B):
            u = int(len_y[i].item())
            t = int(torch.ceil(len_x[i] / 4).item())
            
            sum_logprobs += alpha[u, t][i]
        losses = -sum_logprobs.squeeze() / B

        return losses

    def _alpha(
            self,
            u:int, t:int,
            U:int, T:int,
            alpha:torch.Tensor,
            y:torch.Tensor,
            f:torch.Tensor, g:torch.Tensor, h:torch.Tensor
        ) -> torch.Tensor | int:
        '''The recurrence'''
        terms = []

        # Term 1            (included except for left most column)
        if t != 0:                  
            joined = f[:, t-1] + g[:, u] + h[:, t-1]
            y_hat = torch.log_softmax(joined, dim=-1)   # (B, V+1)
            
            blank_probs = y_hat[:, conf.BLANK_IDX]      # (B,)
            
            terms.append(
                # probability of having emitted everything necessary already
                alpha[u, t-1] +
                # probability of emiting blank
                blank_probs
            )

        # Term 2            (included except for Bottom and Top rows, and Terminal States)
        if u > 0 and u <= U and t != T:
            joined = f[:, t] + g[:, u - 1] + h[:, t]
            y_hat = torch.log_softmax(joined, dim=-1)   # (B, V+1)

            labels = y[:, u-1].unsqueeze(1)             # (B, 1)
            label_probs = y_hat.gather(
                dim=1,
                index=labels
            ).squeeze(1)                                # (B,)

            terms.append(
                # probability of having emitted everything except the u-th token
                alpha[u-1, t] +
                # probability of having emitted the u-th token
                label_probs
            )

        return torch.logsumexp(
            torch.stack(terms, dim=0),
            dim=0
        )
