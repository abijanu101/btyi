import torch

from .model import CTCNetwork
import src.core.speech.config as conf

class CTCTrainer:
    def __init__(self, model:CTCNetwork):
        self.model = model
        self.loss = torch.nn.CTCLoss(conf.BLANK_IDX)
        self.optim = torch.optim.AdamW()

    def train(self, dataloader:torch.utils.data.DataLoader) -> None:
        pass
