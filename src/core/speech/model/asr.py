import torch

from typing import Tuple, List, Iterable
import src.config.asr as conf

from .ctc import CTCNetwork
from .transducer import ConformerEncoder, PredictionNetwork, JointNetwork
from .lm import LanguageModel

class ASRModel(torch.nn.Module):
    '''Orchestrates the actual model pipeline'''

    def __init__(self):
        super().__init__()
        self.ctc = CTCNetwork()
        self.conformer = ConformerEncoder()
        self.prednet = PredictionNetwork()
        self.jointnet = JointNetwork()
        self.lm = LanguageModel()

        self.ctc_hidden = None

    def forward(self, X, ctc_hidden:Tuple[torch.Tensor, torch.Tensor]|None = None):
        if ctc_hidden is None:
            ctc_hidden = self.ctc_hidden

        y, self.ctc_hidden =  self.ctc(X, ctc_hidden)
        return y.argmax(dim=2)
                
        # for now we only have it behave like a minimal CTC net wrapper
    
    def _collapse(self, t: List[List[int]] | List[int], collapse_repeats:bool) -> List[List[int]] | List[int]:
        assert isinstance(t, List)

        # for batching
        if len(t) and isinstance(t[0], List):
            return [self._collapse(i, collapse_repeats) for i in t]
        
        prev = None
        result = []
        for i in t:
            if i != conf.BLANK_IDX and (i != prev or not collapse_repeats):
                result.append(i)
            prev = i
        return result