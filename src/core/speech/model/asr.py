import torch

from .ctc import CTCNetwork
from .transducer import ConformerEncoder, PredictionNetwork, JointNetwork
from .lm import LanguageModel

class ASRModel(torch.nn.Module):
    '''Orchestrates the actual model pipeline'''

    def __init__(self):
        pass

    def forward(self):
        pass