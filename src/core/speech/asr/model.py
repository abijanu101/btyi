import torch

from typing import Tuple, List, Iterable
import src.core.speech.config as conf

from .ctc import CTCNetwork
from .transducer import ConformerTransducer

class ASRModel(torch.nn.Module):
    '''Orchestrates the actual model pipeline'''

    def __init__(self):
        super().__init__()
        self.ctc = CTCNetwork()
        self.transducer = ConformerTransducer()

        
    def forward(self, X):  
        y_1, ctc_hidden = self.ctc(X, None)
        y_2, prednet_hidden, linknet_hidden = self.transducer.forward(
            X, y_1, torch.tensor([conf.BLANK_IDX for i in range(len(X))])
        )
        
        return y_2
                
                
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
    
    def print_params(self) -> None:
        params = {}
        params['CTC Network'] = self._get_stats(self.ctc)
        params['Conformer'] = self._get_stats(self.transducer.conformer)
        params['Prediction Net'] = self._get_stats(self.transducer.prednet)
        params['Link Network'] = self._get_stats(self.transducer.linknet)
        params['Joint Network'] = self._get_stats(self.transducer.jointnet)

        sum_n = 0 
        sum_s = 0 

        for k, (n, s) in params.items():
            print(f'{k}\t {n/10**6:.2f}M  \t({s/2**20:.2f}MB) ')
            sum_n += n
            sum_s += s
        print('--------------------------------------------------')
        print(f'Total\t\t {sum_n/10**6:.2f}M  \t({sum_s/2**20:.2f}MB) ')

    def _get_stats(self, model: torch.nn.Module) -> Tuple[int, int]:
        n_params = sum(i.numel() for i in model.parameters())
        s_params = sum(i.numel() * i.element_size() for i in model.parameters())

        return n_params, s_params