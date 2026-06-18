import torch

from abc import ABC, abstractmethod
from typing import Tuple, List


class TextEncoder(ABC):
    '''manages tokenization, control token insertion and replacement, and stoi encoding'''
    def __init__(self):
        pass
    
    @abstractmethod
    def encode(self, text:str) -> List[int]:
        raise NotImplementedError()


class EmbeddingsFactory(ABC):
    def __init__(self):
        '''Define Embedding Details'''

    @abstractmethod
    def construct_embeddings(self) -> Tuple[torch.Tensor, TextEncoder]:
        '''Return embeddings matrix and an encoder'''
        
