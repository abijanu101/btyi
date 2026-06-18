import numpy as np
import pandas as pd

from collections import Counter
from typing import Iterable

import torch
import torchtext

from src.core.text.embeddings import GloVeEmbeddings


class BasicEnglishSummarizer:
    def __init__(self):        
        self.vocab = GloVeEmbeddings(2**20)
        self.tokenize_fn = torchtext.data.get_tokenizer('basic_english')
        self.model = None
        pass

    def fit(X:Iterable, y:Iterable) -> None:
        pass

    def predict(X) -> None:
        pass

    
    def _load_embeddings(self):
        pass