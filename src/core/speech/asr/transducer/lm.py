import torch

class LanguageModel(torch.nn.Module):
    'For rescoring on second pass'
    def __init__(self):
        super().__init__()

    def forward(self, X):
        pass