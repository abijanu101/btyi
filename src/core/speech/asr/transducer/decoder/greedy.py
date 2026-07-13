import torch

import src.core.speech.config as conf
from src.core.speech.asr.transducer import ConformerTransducer

from typing import List, Tuple

LSTMState = Tuple[torch.Tensor, torch.Tensor]

class ConformerTransducerGreedyDecoder:
    'FOR BATCHED DATA'
    def __init__(self, model:ConformerTransducer):
        self.model = model

    @torch.no_grad
    def decode(
            self,
            X:torch.Tensor,
            y_ctc: List[List[int]],     # decoded and collased
            prednet_hidden: LSTMState | None = None,
            linknet_hidden: LSTMState | None = None
        ) -> List[List[int]]:
        B, T_RAW, _ = X.shape
        T_SS = T_RAW // 4

        f = self.model.encode(X)
        h, linknet_hidden = self.model.link(torch.tensor(y_ctc), linknet_hidden)
        
        t_raw = torch.zeros(B, dtype=torch.long)
        t_ss = torch.zeros(B, dtype=torch.long)
        
        y = torch.tensor([[conf.BLANK_IDX]] * B)
        last_tokens = y[:, -1].view(-1)
        full_h, full_c = prednet_hidden
        while any(t_ss < T_SS):
            active = torch.nonzero(t_ss < T_SS, as_tuple=True)[0]

            active_h = full_h[:, active]
            active_c = full_c[:, active]

            g, (active_h, active_c) = self.model.predict(
                last_tokens[active], 
                (active_h, active_c)
            )

            full_h[:, active] = active_h
            full_c[:, active] = active_c

            logits = torch.zeros(B, conf.LINK_OUT_SIZE)
            logits[active] = self.model.join(
                f[active, t_ss[active]],
                g,
                h[active, t_raw[active]]
            )

            next_tokens = logits.argmax(dim=-1).view(B, -1) 
            y = torch.cat([y, next_tokens], dim=-1)

            last_tokens = next_tokens
            t_ss += (last_tokens == conf.BLANK_IDX)
            t_raw += 4 * (last_tokens == conf.BLANK_IDX)

        return y, (full_h, full_c), linknet_hidden
