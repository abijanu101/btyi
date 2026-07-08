import torch

from .asr import ASRModel

class ASRTrainer:
    def __init__(self, model:ASRModel):
        self.model = model
        self.optim = torch.optim.AdamW(model.parameters())

    def train(self, dl: torch.utils.data.DataLoader):
        pass