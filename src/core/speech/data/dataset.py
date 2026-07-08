import numpy as np
import pandas as pd
import librosa
import torch

from typing import Iterable, List, Tuple, Callable
from src.config.asr import SAMPLING_RATE
from src.config.paths import CVUR_PATH, LIBRI_PATH
from src.core.speech.data.preproc import log_mel_spectrogram, spec_augment


class ASRDataset(torch.utils.data.Dataset):
    'Has the potential to end the world if used improperly'
    
    def __init__(self, df:pd.DataFrame, tokenize_fn:Callable, spec_augment_enabled:bool=True):
        self.df = df
        self.tokenize_fn = tokenize_fn
        self.otop = {
            'mcv-urdu': CVUR_PATH / 'clips',
            'libri-train-clean-100h': LIBRI_PATH / 'train-clean-100',
            'libri-test-clean': LIBRI_PATH / 'test-clean',
            'libri-test-other': LIBRI_PATH / 'test-other'
        }
        self.spec_augment_enabled = spec_augment_enabled

    def __getitem__(self, idx:int) -> Tuple[np.ndarray, List]:
        row = self.df.iloc[idx]

        file = row['file']
        transcription = row['transcription']
        origin = row['origin']
        speaker_id = row['speaker_id']
        passage_id = row['passage_id']
        
        path = self.otop[origin]
        if origin != 'mcv-urdu':
            path = path / speaker_id / passage_id
        path = path / file

        signal, _ = librosa.load(path, sr=SAMPLING_RATE)
        log_mel = log_mel_spectrogram(signal, SAMPLING_RATE)
        if self.spec_augment_enabled:
            log_mel = spec_augment(log_mel)
        
        return log_mel, self.tokenize_fn(transcription)

    def __len__(self) -> int:
        return len(self.df)


def collate_fn(batch: Iterable[Tuple[np.ndarray, List]]) -> Tuple[torch.Tensor, List[int], List[List[int]]]:
    log_mels, transcripts = zip(*batch)
    input_lengths = [len(i) for i in log_mels]
    
    # transposing because pad sequence applies padding at first non batch axis, and its currently (Mel, T) - lowkey no reason to go back cuz (T, Mel) makes more sense
    log_mels = [torch.tensor(i).T for i in log_mels]
    log_mels = torch.nn.utils.rnn.pad_sequence(log_mels, batch_first=True)

    return log_mels, input_lengths, transcripts