import torch

from typing import Tuple, List, Iterable
import src.core.speech.config as conf

from .ctc import CTCNetwork, CTCTrainer, CTCNetworkGreedyDecoder
from .transducer import (
    ConformerTransducer, TransducerTrainer,
    ConformerTransducerGreedyDecoder, ConformerTransducerBeamSearchDecoder
)

class ASRModel(torch.nn.Module):
    '''Orchestrates the actual model pipeline'''

    def __init__(self):
        super().__init__()
        self.ctc = CTCNetwork()

        self.transducer = ConformerTransducer()
        

    def predict(self, X):
        'For Realtime Predictions'
        


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