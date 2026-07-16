import torch
import numpy as np
import src.core.speech.config as conf

from typing import Tuple, List, Iterable

from .ctc import CTCNetworkGreedyDecoder
from .transducer import ConformerTransducerGreedyDecoder, ConformerTransducerBeamSearchDecoder
from .model import ASRModel

class StreamingDecoder:
    'Responsible for Endpointing, and Sequence Collapse'

    def __init__(self, model:ASRModel):
        self.ctc_greedy = CTCNetworkGreedyDecoder(model.ctc)
        self.transducer_beam = ConformerTransducerBeamSearchDecoder(model.transducer)

        self.device = next(model.parameters()).device

        self.max_blank_count = 0
        self.blank_count = 0

        # Stuff the Second Pass Needs
        self.X_buffer:torch.Tensor|None = None              # (B, T, M) tensor of unprocessed frames
        self.ctc_buffer:torch.Tensor|None = None            # (B, T, V+1) tensor of  unprocessed ctc_logprobs 
        
        self.ctc_hidden = None
        self.transducer_hidden = None

        self.transcription = []

    def get_transcription(self) -> List[int]:
        'get the complete transcription'
        return self.transcription

    def get_in_buffer_transcription_snippet(self) -> List[int]:
        'unlikely to ever be used but for a gui implementation this could be handy lol'
        return self._ctc_collapse(self.ctc_buffer, False) if self.ctc_buffer else []


    def decode(self, X:np.ndarray) -> Tuple[List[int], bool]:
        '''Uses the ASR Model to generate trnascriptions, returns True at index 1 if Second Pass was invoked'''
        ctc_out = self._first_pass(X)
        if self.max_blank_count < conf.SWITCH_THRESHOLD:
            return ctc_out, False
        
        self.max_blank_count = 0
        self.blank_count = 0
        return self._second_pass(), True

    
    def _first_pass(self, X:np.ndarray) -> List[int]:
        'Returns the collapsed CTCnet output. Also maintains the (B,T,M) input and (B,T,V+1) logprob tensors'

        X = torch.tensor(X, device=self.device).unsqueeze(0)    # (B, T, M)
        y, self.ctc_hidden = self.ctc_greedy.decode(X, self.ctc_hidden)
        collapsed = self._ctc_collapse(y, True)
        
        # Buffer Management
        if self.X_buffer is not None:     # no need for a self.ctc_buffer is None check since these update together
            self.X_buffer = torch.cat([self.X_buffer, X], dim=1)
            self.ctc_buffer = torch.cat([self.ctc_buffer, y], dim=1)
        else:
            self.X_buffer = X
            self.ctc_buffer = y        
        return collapsed


    def _second_pass(self) -> List[int]:
        'Consumes the Buffers Populated by the First Pass iterations to produce a more refined transcription'

        collapsed, self.transducer_hidden = self.transducer_beam.decode(
            X=self.X_buffer,
            ctc_logprobs=self.ctc_buffer,
            last_y=self.transcription[-1] if self.transcription else conf.BLANK_IDX,
            prednet_hidden=self.transducer_hidden
        )
        
        # Buffer Management
        self.X_buffer = None
        self.ctc_buffer = None

        self.transcription.extend(collapsed)

        return collapsed


    def _ctc_collapse(self, logprobs:torch.Tensor, track_blanks:bool) -> List[int]:
        'More efficient to track blanks as you do it instead of a separate loop'

        y = torch.argmax(logprobs, dim=-1).squeeze(0)           # (T,)
        prev = None
        collapsed = []
        for tkn in y:
            if tkn == conf.BLANK_IDX:
                prev = tkn
                if track_blanks:
                    self.blank_count += 1
                    self.max_blank_count = max(self.max_blank_count, self.blank_count)
                continue

            if track_blanks:
                self.blank_count = 0            
            if tkn != prev:
                collapsed.append(tkn.item())
            prev = tkn
        
        return collapsed