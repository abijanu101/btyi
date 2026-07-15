import torch
import sounddevice
import pandas as pd

from tokenizers import ByteLevelBPETokenizer, InputSequence
from typing import Iterable, List, Tuple, Callable

from src.config.paths import BILINGUAL_PATH, TRAINED_PATH

from .data import ASRDataset, collate_fn, log_mel_spectrogram
from .asr import ASRModel, ASRTrainer, StreamingDecoder
from .config import N_VOCAB, SAMPLING_RATE, STREAM_CHUNK_SAMPLES, STREAM_CHUNK_DURATION

class BiASR:
    'This Urdu-English ASR System is Bisexual'

    def __init__(self):
        self.tokenizer = ByteLevelBPETokenizer()
        self.tokenizer.train(
            files=[(BILINGUAL_PATH / 'flattend_corpora.txt').__str__()],
            vocab_size= N_VOCAB,
        )
        self.model = ASRModel().to('cuda')
        self.trainer = ASRTrainer(self.model)

    # @torch.no_grad
    def stream(self, dur_limit:int|None = 60, sentinel_phrase:str|None = "bye gang"):
        'Start streaming ASR system'
        assert dur_limit is None or dur_limit > 0
        assert sentinel_phrase is None or isinstance(sentinel_phrase, str) and sentinel_phrase != ""

        if sentinel_phrase:
            sentinel_phrase = sentinel_phrase.lower()

        stream = sounddevice.InputStream(
            samplerate=SAMPLING_RATE,
            blocksize=STREAM_CHUNK_SAMPLES,
            channels=1  # Mono only
        )

        decoder = StreamingDecoder(self.model)

        dur_elapsed = 0
        stream.start()
        while dur_limit is None or dur_elapsed < dur_limit:
            X, _ = stream.read(STREAM_CHUNK_SAMPLES)
            X = X[:, 0]             # Mono Channel
            
            X = log_mel_spectrogram(X, SAMPLING_RATE).T
            y = decoder.decode(X)
            y = self.tokenizer.decode(y)

            print(y, end='')

            dur_elapsed += STREAM_CHUNK_DURATION
        stream.stop()

    def train_single(self, df:pd.DataFrame, batch_size:int, epochs:int, end_to_end:bool = True) -> None:
        'Train on a single corpus'
        DEVICE = next(self.model.parameters()).device
        self.model.train()

        ds = ASRDataset(df, self._tokenize, True)
        dl = torch.utils.data.DataLoader(ds, batch_size, shuffle=True, collate_fn=collate_fn)

        for i in range(epochs):
            print(f'[Epoch {i}/{epochs}]')
            j = 0
            for X, y, len_x, len_y in dl:
                X = X.to(DEVICE)
                y = y.to(DEVICE)
                len_x = len_x.to(DEVICE)
                len_y = len_y.to(DEVICE)

                j+=1
                print(j, end=': ' if end_to_end else ':\n')
                self.trainer.step_together(X, y, len_x, len_y) if end_to_end else self.trainer.step_both(X, y, len_x, len_y)

                if j % 10 == 0:
                    self.save()
        self.save()

    def train_round_robin(
            self,
            df1:pd.DataFrame, df2:pd.DataFrame,
            batch_size1:int, batch_size2:int, 
            n_steps:int, end_to_end:bool = True
        ) -> None:
        'Train on two corpora, drawing one batch from df1 and another from df2'
        DEVICE = next(self.model.parameters()).device
        self.model.train()

        ds1 = ASRDataset(df1, self._tokenize, True)
        ds2 = ASRDataset(df2, self._tokenize, True)

        dls = [
            torch.utils.data.DataLoader(ds1, batch_size1, shuffle=True, collate_fn=collate_fn),
            torch.utils.data.DataLoader(ds2, batch_size2, shuffle=True, collate_fn=collate_fn)
        ]
        iters = [iter(dl) for dl in dls]

        X:torch.Tensor
        y:List[List[int]]
        len_x:List[int]
        len_y:List[int]

        for i in range(n_steps):
            print(f'[{i+1}/{n_steps} steps,  df{i%2 + 1}] ', end='' if end_to_end else '\n')            
            try:
                X, y, len_x, len_y = next(iters[i % 2])
            except StopIteration:
                iters[i % 2] = iter(dls[i % 2])
                X, y, len_x, len_y = next(iters[i % 2])

            X = X.to(DEVICE)
            y = y.to(DEVICE)
            len_x = len_x.to(DEVICE)
            len_y = len_y.to(DEVICE)
            
            self.trainer.step_together(X, y, len_x, len_y) if end_to_end else self.trainer.step_both(X, y, len_x, len_y)
            if (i+1) % 10 == 0:
                self.save()
        self.save()


    @torch.no_grad
    def evaluate(self, df:pd.DataFrame, batch_size:int):
        'Evaluate WER on a provided corpus'
        DEVICE = next(self.model.parameters()).device
        self.model.eval()

        ds = ASRDataset(df, self._tokenize, False)  # no spec aug
        dl = torch.utils.data.DataLoader(ds, batch_size, shuffle=True, collate_fn=collate_fn)


    def save(self):
        'Save model state'
        root_dir = TRAINED_PATH / 'asr'
        tokenizer_dir = root_dir / 'tokenizer'
        tokenizer_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer.save_model(str(tokenizer_dir))
        torch.save(
            {
                'model': self.model.state_dict(),
                'ctc optim': self.trainer.ctc_trainer.optim.state_dict(),
                'transducer optim': self.trainer.transducer_trainer.optim.state_dict(),
            },
            str(root_dir / 'model.pt')
        )

        print('Saved Successfully.')

    def load(self):
        'Load from presaved'
        root_dir = TRAINED_PATH / 'asr'
        tokenizer_dir = root_dir / 'tokenizer'

        self.tokenizer = ByteLevelBPETokenizer.from_file(
            str(tokenizer_dir / 'vocab.json'), str(tokenizer_dir / 'merges.txt')
        )
        checkpoint = torch.load(str(root_dir / 'model.pt'))

        self.model.load_state_dict(checkpoint['model'])
        self.trainer.ctc_trainer.optim.load_state_dict(checkpoint['ctc optim'])
        self.trainer.transducer_trainer.optim.load_state_dict(checkpoint['transducer optim'])

        print('Loaded Successfully.')


    def _tokenize(self, text:str):
        return self.tokenizer.encode(text).ids