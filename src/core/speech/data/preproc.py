import numpy as np
import librosa

from typing import Iterable

import src.core.speech.config as conf
from src.core.speech.data.transforms import apply_frequency_masking, apply_time_masking, apply_time_warping


def log_mel_spectrogram(signal:Iterable, sr:int) -> np.ndarray:
    return librosa.power_to_db(
        librosa.feature.melspectrogram(
            y=signal,
            sr=sr,
            n_fft=conf.N_FFT,
            hop_length=conf.HOP_LEN,
            win_length=conf.WIN_LEN,
            n_mels=conf.N_MELS,
            power=2     # power is better than raw amplitudes or something?
        )
    )

def spec_augment(log_mel:np.ndarray) -> np.ndarray:
    log_mel = log_mel.copy()

    log_mel = apply_time_warping(log_mel, conf.TW_P, conf.TW_LIM)             # upto 0.5s of distortion
    log_mel = apply_time_masking(log_mel, conf.TM1_P, conf.TM1_LIM)             # upto 150 ms
    log_mel = apply_time_masking(log_mel, conf.TM2_P, conf.TM2_LIM)             # upto 150 ms
    log_mel = apply_frequency_masking(log_mel, conf.FM_P, conf.FM_LIM)        # upto 10 frequency bands

    return log_mel