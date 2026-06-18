import torch
import torchtext

from .base import TextEncoder, EmbeddingsFactory
from src.config.paths import PRETRAINED_PATH

from typing import Literal, Tuple, List, Dict, Callable

CONTROL_TOKENS = ['<START>', '<END>', '<UNK>', '<PAD>']

class TextEncoderSEUP(TextEncoder):
    '''TextEncoder for 4 Control Tokens, <START>, <END>, <UNK>, <PAD>'''
    
    def __init__(self, itos:list, stoi:Dict[str, int], tokenize_fn:Callable):
        self.itos = itos
        self.stoi = stoi
        self.tokenize_fn = tokenize_fn

        self._start = self.stoi['<START>']
        self._end = self.stoi['<END>']
        self._unk = self.stoi['<UNK>']
        self._pad = self.stoi['<PAD>']


    def encode(self, text:str) -> List[int]:
        ''' str becomes stoi(insert_control_tokens(tokenize.str())) '''
        # TO DO: add memoization, <START>, <END> insertion
        return [self._start] + [self.stoi.get(i, self._unk) for i in self.tokenize_fn(text)] + [self._end]


class SEUPGloVeFactory(EmbeddingsFactory):    
    def __init__(self, name:Literal['42B', '840B', 'twitter.27B', '6B']='840B', n_dim:int=300, n_vocab:int|None=None):
        self.name = name
        self.n_dim = n_dim
        self.n_vocab = n_vocab


    def construct_embeddings(self) -> Tuple[torch.Tensor, TextEncoderSEUP]:   
        # loading
        gloVe_base = torchtext.vocab.GloVe(name=self.name, dim=self.n_dim, cache=PRETRAINED_PATH)
        

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