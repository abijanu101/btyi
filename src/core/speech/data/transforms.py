import numpy as np
import cv2

from typing import Iterator, Tuple
import src.core.speech.config as conf

def time_warp(log_mel:np.ndarray, low:int, high:int, scale_factor:float, pivot_ratio:float, squish_right:bool) -> np.ndarray:
    assert low >= 0 and low <= high
    assert scale_factor > 1 and pivot_ratio > 0
    assert isinstance(squish_right, bool)

    if low + 1 >= high:
        return log_mel

    B = log_mel.shape[0]
    pivot = low + round((high - low) / pivot_ratio)

    unchanged_left = log_mel[:, :low]
    unchanged_right = log_mel[:, high:]

    to_squish = log_mel[:, low:pivot]
    to_stretch = log_mel[:, pivot:high]

    if squish_right:
        to_squish, to_stretch = to_stretch, to_squish

    squished = cv2.resize(to_squish, (round((pivot - low) / scale_factor), B))
    stretched = cv2.resize(to_stretch, (round((high - pivot) * scale_factor), B))

    return np.hstack(
        [unchanged_left, stretched, squished, unchanged_right]
        if squish_right
        else [unchanged_left, squished, stretched, unchanged_right]
    )

def generate_ranges(N:int, p_m:float, lim_m:int) -> Iterator[Tuple[int, int]]:
    assert p_m > 0 and p_m < 1 and lim_m > 1 and isinstance(lim_m, int)

    mask_starts = np.random.randint(0, N, int(p_m * N))
    mask_durations = np.random.randint(1, lim_m, len(mask_starts))
    mask_ends = mask_durations + mask_starts
    mask_ends[mask_ends >= N] = N - 1  
    
    return zip(mask_starts, mask_ends)

def apply_frequency_masking(log_mel: np.ndarray, p_m:float, lim_m:int) -> np.ndarray:
    N = log_mel.shape[0]
    mask_value = log_mel.mean()
    
    for i_s, i_e in generate_ranges(N, p_m, lim_m):
        log_mel[i_s : i_e] = mask_value
    
    return log_mel

def apply_time_masking(log_mel: np.ndarray, p_m:float, lim_m:int) -> np.ndarray:
    N = log_mel.shape[1]
    mask_value = log_mel.mean()
    
    for i_s, i_e in generate_ranges(N, p_m, lim_m):
        log_mel[:, i_s : i_e] = mask_value
    
    return log_mel

def apply_time_warping(log_mel:np.ndarray, p_m:float, lim_m:int) -> np.ndarray:
    for i_s, i_e in generate_ranges(log_mel.shape[1], p_m, lim_m):
        scl, piv, dir = np.random.random(3)

        scl = scl * conf.TW_SCL_RANGE + conf.TW_SCL_MIN
        piv = piv * conf.TW_PIV_RANGE + conf.TW_PIV_MIN
        dir = bool(dir > 0.5)


        i_s = min(i_s, log_mel.shape[1] - 1)    # rescaling edge cases
        i_e = min(i_e, log_mel.shape[1] - 1)    # rescaling edge cases

        log_mel = time_warp(log_mel, i_s, i_e, scl, piv, dir)        

    return log_mel
