import torch


class ConformerEncoder(torch.nn.Module):
    'For second pass chunk-wise processing of input frames'
    def __init__(self):
        super().__init__()

    def forward(self, X):
        pass


class PredictionNetwork(torch.nn.Module):
    'For second pass processing of past outputs'
    def __init__(self):
        super().__init__()

    def forward(self, X):
        pass


class JointNetwork(torch.nn.Module):
    '3 Way Joint Network merging the First Pass CTC Network with the Second Pass Conformer-Transducer architecture'
    def __init__(self):
        super().__init__()

    def forward(self, X):
        pass