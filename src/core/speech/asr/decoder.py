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

        self.ctc_hidden = None
        self.transducer_hidden = None

        self.device = next(model.parameters()).device

        self.buffer = []
        self.transcription = []

        self.max_blank_count = 0
        self.blank_count = 0


    def decode(self, X:np.ndarray):
        X = torch.tensor(X, device=self.device)
        
        # FIRST PASS
        logprobs, self.ctc_hidden = self.ctc_greedy.decode(X.unsqueeze(0), self.ctc_hidden)
        y_ctc = torch.argmax(logprobs, dim=-1)
        y_ctc = y_ctc.squeeze(0)

        prev = None
        collapsed_ctc = []
        for tkn in y_ctc:
            if tkn == conf.BLANK_IDX:
                prev = tkn
                self.blank_count += 1
                self.max_blank_count = max(self.max_blank_count, self.blank_count)
                continue

            self.blank_count = 0            
            if tkn != prev:
                collapsed_ctc.append(i)
            prev = tkn
        
        self.buffer.extend(collapsed_ctc)

        if self.max_blank_count < conf.SWITCH_THRESHOLD:
            self.max_blank_count = 0
            self.blank_count = 0
            return collapsed_ctc

        # SECOND PASS
        


        return [923, 402]


    def get_buffer(self) -> Tuple[List[int], List[int]]:
        pass

    def get_transcription(self):
        pass
