import torch

from ..model import ConformerTransducer
import src.core.speech.config as conf


class TransducerTrainer:
    def __init__(self, model:ConformerTransducer):
        self.model = model
        self.optim = torch.optim.AdamW()

    def train(self, dataloader:torch.utils.data.DataLoader) -> None:
        pass
    