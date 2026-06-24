import torch
import torchtext

from src.config.paths import PRETRAINED_PATH
from .base import TextEncoder, EmbeddingsFactory

from typing import Literal, Tuple, List, Dict, Callable, Iterable

CONTROL_TOKENS = ['<START>', '<END>', '<UNK>', '<PAD>']

class TextEncoderSEUP(TextEncoder):
    '''TextEncoder for 4 Control Tokens: <START>, <END>, <UNK>, <PAD>'''
    
    def __init__(self, itos:list, stoi:Dict[str, int], tokenize_fn:Callable):
        self.itos = itos
        self.stoi = stoi
        self.tokenize_fn = tokenize_fn
    
    def encode(self, text:str) -> List[int]:
        ''' str becomes stoi(insert_control_tokens(tokenize.str())) '''
        # TO DO: add memoization flag to constructor and build a cache to save recomputation for multi-epoch training
        return [self.stoi['<START>']] + [self.stoi.get(i, self.stoi['<UNK>']) for i in self.tokenize_fn(text)] + [self.stoi['<END>']]


    def decode(self, sequence:Iterable[int]) -> str:
        return ' '.join([
            (
                self.itos[idx] 
                if idx in range(len(self.itos))
                else '<UNK>'    # out of bounds index should never happen, this is just precautionary
            )
            for idx in sequence
            if idx != self.stoi['<PAD>']    # skip padding
        ])

    def decode_batch(self, batch:torch.Tensor) -> List[str]:
        return [self.decode(i) for i in batch] 

    def get_control_stoi(self) -> dict:
        return {
            i: self.stoi[i]
            for i in ['<START>', '<END>', '<UNK>', '<PAD>']
        }


class SEUPGloVeFactory(EmbeddingsFactory):    
    '''Embeddings Loader and Vocabulary Manager for GloVe + 4 Control Tokens: <START>, <END>, <UNK>, <PAD>'''

    def __init__(self, name:Literal['42B', '840B', 'twitter.27B', '6B']='840B', n_dim:int=300, n_vocab:int|None=None):
        self.name = name
        self.n_dim = n_dim
        self.n_vocab = n_vocab


    def construct_embeddings(self) -> Tuple[torch.Tensor, TextEncoderSEUP]:   
        # loading
        gloVe_base = torchtext.vocab.GloVe(name=self.name, dim=self.n_dim, cache=PRETRAINED_PATH)
        print(f"Loaded {len(gloVe_base.itos)} GloVe embeddings from 840B.300d Common Crawl.")

        # limit enforcing
        true_vocab = len(gloVe_base.itos)
        WORDS_PLUS_CTRL = true_vocab + len(CONTROL_TOKENS)

        if self.n_vocab is not None:
            LIMIT_MINUS_CTRL = self.n_vocab - len(CONTROL_TOKENS)
            if WORDS_PLUS_CTRL > self.n_vocab:
                true_vocab = LIMIT_MINUS_CTRL

        words : List[str] = gloVe_base.itos[:true_vocab]
        vectors :torch.Tensor = gloVe_base.vectors[:true_vocab]


        # merging with control tokens
        itos = CONTROL_TOKENS + words
        stoi = {v:k for k,v in enumerate(itos)}
        vectors = torch.cat([
            torch.rand(len(CONTROL_TOKENS) * self.n_dim).reshape((-1, self.n_dim)),
            vectors
        ])

        return vectors, TextEncoderSEUP(itos, stoi, torchtext.data.get_tokenizer('basic_english'))