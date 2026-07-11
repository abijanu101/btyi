import torch

from .conformer import Encoder
from .networks import PredictionNetwork, JointNetwork, LinkNetwork
from .lm import LanguageModel

class ConformerTransducer(torch.nn.Module):
    ''
    
    def __init__(self):
        super().__init__()
        self.conformer = Encoder()
        self.prednet = PredictionNetwork()
        self.linknet = LinkNetwork()
        self.jointnet = JointNetwork()
        self.lm = LanguageModel()
        
        self.prednet_hidden = None
        self.linknet_hidden = None

    def forward(self):
        pass