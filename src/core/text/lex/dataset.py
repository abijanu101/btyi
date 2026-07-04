import pandas as pd
import torch

from typing import Callable, Iterable, List, Tuple
from src.config.lex import CHUNK_SIZE


class LexDataset(torch.utils.data.Dataset):
    def __init__(self, df:pd.DataFrame, tokenize_fn:Callable[[str], List[int]], mutate_fn:Callable[[str], str] | None):
        '''
        df must have
        - fixed size chunks of in-sequence conversation
        - premined in-chunk positive and negative examples (and as a corrolary filtered for multi-speaker chunks)
        '''

        self.df = df
        self.stoi = { id : i % CHUNK_SIZE for i, id in enumerate(df.index) }
        self.n_chunks = df['chunk_id'].nunique()

        self.tokenize_fn = tokenize_fn
        self.mutate_fn = mutate_fn
    

    def __getitem__(self, idx : int) -> List[Tuple[torch.Tensor, int, int]]:
        '''Get In-sequence Conversation Chunk alongside Positives and Negatives for each Chunk Segment'''

        chunk = self.df[self.df['chunk_id'] == idx]

        texts = chunk['text'].to_list()
        pos = chunk['pos_id']
        neg = chunk['neg_id']
        
        pos = [self.stoi[i] for i in pos]
        neg = [self.stoi[i] for i in neg]

        if self.mutate_fn:
            texts = [self.mutate_fn(sent) for sent in texts]

        prompts = []
        for sp, sc in zip(['[BOS]'] + texts[:-1], texts):
            prompts.append(torch.tensor(self.tokenize_fn(sp + '[SEP]' + sc)))

        return list(zip(prompts, pos, neg))


    def __len__(self) -> int:
        '''Get number of chunks'''
        return self.n_chunks
