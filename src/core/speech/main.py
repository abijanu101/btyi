import numpy as np
import pandas as pd
import torch

from tokenizers import ByteLevelBPETokenizer
from typing import Iterable, List, Tuple, Callable

from src.config.paths import BILINGUAL_PATH
from src.config.asr import SAMPLING_RATE, N_VOCAB, PAD_TOKEN

from .data import ASRDataset, collate_fn, log_mel_spectrogram
from .model import ASRModel, ASRTrainer

class BiASR:
    'This Urdu-English ASR System is Bisexual'

    def __init__(self):
        self.tokenizer = ByteLevelBPETokenizer()
        self.tokenizer.train(
            files=[(BILINGUAL_PATH / 'flattened_corpora.txt').__str__()],
            vocab_size= N_VOCAB
        )
        # self.blank_idx = self.tokenizer.get_vocab_length()

    def train_single(self, df:pd.DataFrame, batch_size:int, epochs:int):
        'Train on a single corpus'

    def train_round_robin(self, df1:pd.DataFrame, df2:pd.DataFrame, batch_size1:int, batch_size2:int, epochs:int):
        'Train on two corpora, drawing one batch from df1 and another from df2'

    def evaluate(self, df:pd.DataFrame):
        'Evaluate WER on a provided corpus'

    def start(self):
        'Start streaming ASR system'

    def save(self):
        'Save model state'

    def load(self):
        'Load from presaved'
