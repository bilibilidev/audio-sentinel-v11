"""Pure NumPy MFCC front-end used by the v11 event features.

No speech synthesis, templates, DTW matching or keyword classification.
"""
from __future__ import annotations

import math
import numpy as np

SAMPLE_RATE = 16000
FRAME_LEN = 400      # 25ms at 16kHz
HOP_LEN = 160        # 10ms at 16kHz
NUM_MFCC = 13        # 13 MFCC coefficients
NUM_FILTERS = 26     # Mel filter bank size
NFFT = 512


def _hz_to_mel(hz: float) -> float:
    """Convert frequency in Hz to Mel scale."""
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    """Convert Mel scale to frequency in Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(num_filters: int, nfft: int, sample_rate: int) -> np.ndarray:
    """Pre-compute Mel filter bank matrix."""
    high_freq = sample_rate / 2
    mel_min = _hz_to_mel(0)
    mel_max = _hz_to_mel(high_freq)
    mel_points = np.linspace(mel_min, mel_max, num_filters + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((nfft + 1) * hz_points / sample_rate).astype(int)

    fbank = np.zeros((num_filters, nfft // 2 + 1), dtype=np.float32)
    for m in range(1, num_filters + 1):
        left = bin_points[m - 1]
        center = bin_points[m]
        right = bin_points[m + 1]
        for k in range(left, center):
            if center > left:
                fbank[m - 1, k] = (k - left) / (center - left)
        for k in range(center, right):
            if right > center:
                fbank[m - 1, k] = (right - k) / (right - center)
    return fbank


# DCT-II matrix for MFCC computation
def _dct_matrix(num_mfcc: int, num_filters: int) -> np.ndarray:
    """Pre-compute DCT-II matrix."""
    matrix = np.zeros((num_mfcc, num_filters), dtype=np.float32)
    for k in range(num_mfcc):
        for n in range(num_filters):
            matrix[k, n] = math.cos(math.pi * k * (2 * n + 1) / (2 * num_filters))
    matrix *= math.sqrt(2.0 / num_filters)
    matrix[0, :] *= math.sqrt(0.5)  # Scaling for k=0
    return matrix


# Pre-compute tables
_MEL_FB = _mel_filterbank(NUM_FILTERS, NFFT, SAMPLE_RATE)
_DCT_MAT = _dct_matrix(NUM_MFCC, NUM_FILTERS)


def extract_mfcc(pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Extract MFCC features from PCM audio.

    Args:
        pcm: int16 mono audio samples
        sample_rate: sample rate (must be 16000)

    Returns:
        MFCC matrix of shape (num_frames, NUM_MFCC)
    """
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE}Hz, got {sample_rate}Hz")

    # Normalize to float
    x = pcm.astype(np.float32) / 32768.0

    # Pre-emphasis filter (high-pass): y[n] = x[n] - 0.97 * x[n-1]
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])

    # Frame the signal
    num_frames = max(1, 1 + (len(x) - FRAME_LEN) // HOP_LEN)
    frames = np.zeros((num_frames, FRAME_LEN), dtype=np.float32)
    for i in range(num_frames):
        start = i * HOP_LEN
        end = min(start + FRAME_LEN, len(x))
        frames[i, :end - start] = x[start:end]

    # Apply Hamming window
    hamming = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(FRAME_LEN) / (FRAME_LEN - 1))
    frames *= hamming

    # FFT (magnitude spectrum)
    fft_result = np.fft.rfft(frames, n=NFFT, axis=1)
    power_spectrum = np.abs(fft_result) ** 2

    # Mel filter bank
    mel_energy = power_spectrum @ _MEL_FB.T  # (num_frames, NUM_FILTERS)
    mel_energy = np.where(mel_energy < 1e-10, 1e-10, mel_energy)
    log_mel = np.log(mel_energy)

    # DCT to get MFCC
    mfcc = log_mel @ _DCT_MAT.T  # (num_frames, NUM_MFCC)

    return mfcc
