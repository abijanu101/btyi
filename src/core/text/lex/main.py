import pandas as pd
import torch
import re


import wtpsplit
from tokenizers import ByteLevelBPETokenizer
from sklearn.cluster import AgglomerativeClustering
from typing import List, Tuple

from src.config.paths import DATA_PATH, TRAINED_PATH
from src.config.lex import N_VOCAB
from src.core.text.lex.sentemb import SentenceEmbeddingModel
from src.core.text.lex.dataset import LexDataset

import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

class Lex:
    def __init__(self):
        self.tokenizer = ByteLevelBPETokenizer()
        self.tokenizer.train(
            files=[DATA_PATH.joinpath('flattened_corpora.txt').__str__()],
            vocab_size=N_VOCAB,
            min_frequency=2,
            special_tokens=['[BOS]', '[SEP]']
        )
        print(f"Trained Tokenizer for a {N_VOCAB} token Vocabulary.")
        
        self.sentemb = SentenceEmbeddingModel()
        print(f"Initialized Embeddings Model with {sum(p.numel() for p in self.sentemb.model.parameters())} parameters.")
        
        self.sat = wtpsplit.SaT('sat-3l-sm')
        print("Loaded SaT-3l-sm.")

    def fit(self, df: pd.DataFrame, epochs:int = 10) -> None:
        ds = LexDataset(df, self._tokenize_fn, self._mutate_fn)
        self.sentemb.fit(ds, epochs)
    
    def predict(self, text:str) -> List[Tuple[int, str]]:
        sentences = self.sat.split(text)

        if len(sentences) < 2:
            return [0]

        X = [ torch.tensor(self._tokenize_fn(sentence)) for sentence in sentences ]
        y = self.sentemb.predict(X).cpu()

        clustering = AgglomerativeClustering(n_clusters=None,compute_full_tree=True, linkage='ward', distance_threshold=13)
        clustering.fit(y)

        return [(int(label), sentences[i]) for i, label in enumerate(clustering.labels_)]


    def save(self) -> None:
        tokenizer_dir = TRAINED_PATH / 'lex' / 'tokenizer'
        tokenizer_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer.save_model(tokenizer_dir.__str__())
        self.sentemb.save()

        print('Saved Successfully.')

    def load(self) -> None:
        tokenizer_dir = TRAINED_PATH / 'lex' / 'tokenizer'
        self.tokenizer = ByteLevelBPETokenizer.from_file(str(tokenizer_dir / 'vocab.json'), str(tokenizer_dir / 'merges.txt'))

        self.sentemb.load()

        print("Loaded Successfully")

    def _tokenize_fn(self, text:str):
        return self.tokenizer.encode(text).ids
        
    def _mutate_fn(self, text:str) -> str:
        '''Mutation function for robustness training'''
        rand = torch.rand(1)[0]
        if rand > 0.30:
            return text
        
        if (rand < 0.30 and rand > 0.20) or rand < 0.10:
            text = text.lower()
        if rand < 0.20:
            text = ''.join(re.split(r'[.,;:!?]+', text))
        
        return text