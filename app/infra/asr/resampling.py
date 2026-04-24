from __future__ import annotations

from math import gcd

import numpy as np


def resample_audio(audio: np.ndarray, *, sample_rate: int, target_rate: int = 16000) -> np.ndarray:
    """Resample mono float audio with an anti-aliasing filter for common downsampling paths."""
    signal = np.asarray(audio, dtype=np.float32).reshape(-1)
    if signal.size == 0:
        return signal
    if sample_rate <= 0 or target_rate <= 0 or sample_rate == target_rate:
        return signal.astype(np.float32, copy=False)
    if signal.size <= 1:
        return signal.astype(np.float32, copy=False)

    if sample_rate > target_rate and sample_rate % target_rate == 0:
        factor = sample_rate // target_rate
        return _downsample_integer(signal, factor=factor)

    if sample_rate > target_rate:
        common = gcd(int(sample_rate), int(target_rate))
        up = target_rate // common
        down = sample_rate // common
        if up <= 8 and down <= 32:
            return _resample_polyphase(signal, up=up, down=down)

    return _resample_linear(signal, sample_rate=sample_rate, target_rate=target_rate)


def _downsample_integer(audio: np.ndarray, *, factor: int) -> np.ndarray:
    if factor <= 1:
        return audio.astype(np.float32, copy=False)
    filtered = _lowpass_for_downsample(audio, down_factor=factor)
    return filtered[::factor].astype(np.float32, copy=False)


def _resample_polyphase(audio: np.ndarray, *, up: int, down: int) -> np.ndarray:
    if up <= 0 or down <= 0:
        return audio.astype(np.float32, copy=False)
    if up == 1:
        return _downsample_integer(audio, factor=down)

    upsampled = np.zeros(int(audio.size * up), dtype=np.float32)
    upsampled[::up] = audio
    filtered = _lowpass_for_downsample(upsampled, down_factor=down, gain=float(up))
    expected_len = max(1, int(round(audio.size * up / down)))
    return filtered[::down][:expected_len].astype(np.float32, copy=False)


def _lowpass_for_downsample(audio: np.ndarray, *, down_factor: int, gain: float = 1.0) -> np.ndarray:
    taps = _lowpass_kernel(down_factor=down_factor) * float(gain)
    return np.convolve(audio.astype(np.float32, copy=False), taps, mode="same").astype(np.float32, copy=False)


def _lowpass_kernel(*, down_factor: int) -> np.ndarray:
    factor = max(1, int(down_factor))
    # Keep speech bandwidth while attenuating content that would alias into the
    # 16 kHz Whisper input. 63 taps is inexpensive for real-time chunks and far
    # cleaner than straight interpolation/decimation.
    num_taps = 63 if factor <= 3 else 95
    cutoff = 0.46 / float(factor)
    n = np.arange(num_taps, dtype=np.float64) - ((num_taps - 1) / 2.0)
    kernel = 2.0 * cutoff * np.sinc(2.0 * cutoff * n)
    window = np.hamming(num_taps)
    kernel *= window
    kernel /= max(float(np.sum(kernel)), 1e-12)
    return kernel.astype(np.float32)


def _resample_linear(audio: np.ndarray, *, sample_rate: int, target_rate: int) -> np.ndarray:
    if sample_rate <= 0 or target_rate <= 0 or sample_rate == target_rate or audio.size <= 1:
        return audio.astype(np.float32, copy=False)
    src_len = int(audio.shape[0])
    dst_len = max(1, int(round(src_len * target_rate / sample_rate)))
    src_x = np.linspace(0.0, 1.0, src_len, endpoint=False)
    dst_x = np.linspace(0.0, 1.0, dst_len, endpoint=False)
    return np.interp(dst_x, src_x, audio).astype(np.float32, copy=False)


__all__ = ["resample_audio"]
