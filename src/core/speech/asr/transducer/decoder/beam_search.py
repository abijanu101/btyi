import torch

import src.core.speech.config as conf
from src.core.speech.asr.transducer import ConformerTransducer

from .hypothesis import Hypothesis

from typing import List, Tuple

class ConformerTransducerBeamSearchDecoder:
    '''
        Stopping Criteria:
        - max(complete) > max(queue)
        - k Hypotheses Completed
        - Amount of Hypotheses blocked by the MAX_LENGTH constraint exceeds a certain threshold
        - Queue Emptied
    '''

    def __init__(self, model:ConformerTransducer):
        self.model = model

    def decode(
            self,
            X:torch.Tensor,
            ctc_logprobs:torch.Tensor,
            last_y:int,
            prednet_hidden:torch.Tensor|None
        ) -> Tuple[List[int], Tuple]:
        'You will NEVER guess what this does... it applies a standard beam search decoding algorithm.'
        DEVICE = next(self.model.parameters()).device

        
        f = self.model.encode(X)                            # (B,T,_)
        f = self.model.project_f(f).squeeze(0)     # (T,V+1)

        h = self.model.link(ctc_logprobs)           # (B,T,_)
        h = self.model.project_h(h).squeeze(0)     # (B,T,_)


        T = f.shape[0]                    # max encoder timesteps
        U = conf.BEAM_LENGTH_LIMIT        # max decoder timesteps


        # we will ignore the first element of our generated tokens before decoding so its chill
        beam = [Hypothesis(0, 0, [last_y], 0, prednet_hidden)]
        completed = []
        blocked = []        # hypothesis stoppede from growing because a limit has been reached

        # THE LOOP
        y = None
        while (beam):
            # Early Stopping
            if completed and (
                beam[-1] < completed[-1] or
                len(completed) >= conf.BEAM_COMPLETED_THRESH
            ):
                y = completed[-1]
                break

            if len(blocked) >= conf.BEAM_BLOCKED_THRESH:
                y = blocked[-1]
                break

            # The Actual Loop Body
            hyp = beam.pop()
            
            prev_y = torch.tensor(hyp.tokens[-1], device=DEVICE).view(1,1)
            g, new_hidden = self.model.predict(prev_y, hyp.hidden)  # (1,1)
            g = self.model.project_g(g).squeeze(0)

            y_hat = torch.log_softmax(f[hyp.t] + g + h[hyp.t], dim=-1)
            
            for extension in hyp.extend(y_hat, new_hidden):
                if extension.u >= U:
                    blocked.append(extension)
                    blocked.sort()
                    continue

                if extension.t >= T:
                    completed.append(extension)
                    completed.sort()
                    continue
                                    
                beam.append(extension)    
                beam.sort()
                beam = beam[-conf.BEAM_WIDTH:]

        # Edge Case Handling
        if not y:
            if (
                blocked and (
                    not completed or
                    completed and blocked[-1] > completed[-1]
                )
            ):
                y = blocked[-1]
            else:
                y = completed[-1]

        return y.tokens[1:], y.hidden