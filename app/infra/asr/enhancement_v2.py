from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(slots=True)
class EnhancementChunk:
    audio: np.ndarray
    noise_floor_rms: float
    suppression_ratio: float
    music_likelihood: float
    spectral_flatness: float


class AsrSpeechEnhancerV2:
    """Lightweight speech enhancement for desktop realtime ASR.

    This is intentionally cheaper than full denoise/source-separation models.
    It provides:
    - adaptive stationary-noise suppression
    - conservative tonal/music-bed attenuation
    - chunk-to-chunk noise floor tracking
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        noise_reduce_strength: float = 0.42,
        noise_adapt_rate: float = 0.18,
        min_gain: float = 0.18,
        music_suppress_strength: float = 0.2,
    ) -> None:
        self._enabled = bool(enabled)
        self._noise_reduce_strength = max(0.0, min(1.0, float(noise_reduce_strength)))
        self._noise_adapt_rate = max(0.01, min(0.8, float(noise_adapt_rate)))
        self._min_gain = max(0.05, min(1.0, float(min_gain)))
        self._music_suppress_strength = max(0.0, min(1.0, float(music_suppress_strength)))
        self._noise_profile: np.ndarray | None = None
        self._last_stats = {
            "noise_floor_rms": 0.0,
            "suppression_ratio": 0.0,
            "music_likelihood": 0.0,
            "spectral_flatness": 0.0,
        }

    def reset(self) -> None:
        self._noise_profile = None
        self._last_stats = {
            "noise_floor_rms": 0.0,
            "suppression_ratio": 0.0,
            "music_likelihood": 0.0,
            "spectral_flatness": 0.0,
        }

    def stats(self) -> dict[str, float]:
        return dict(self._last_stats)

    def process(self, audio: np.ndarray, sample_rate: int, *, speech_ratio: float) -> EnhancementChunk:
        signal = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not self._enabled or signal.size == 0 or sample_rate <= 0:
            rms = _rms(signal)
            chunk = EnhancementChunk(
                audio=signal.astype(np.float32, copy=False),
                noise_floor_rms=rms,
                suppression_ratio=0.0,
                music_likelihood=0.0,
                spectral_flatness=1.0,
            )
            self._last_stats = {
                "noise_floor_rms": round(chunk.noise_floor_rms, 5),
                "suppression_ratio": round(chunk.suppression_ratio, 4),
                "music_likelihood": round(chunk.music_likelihood, 4),
                "spectral_flatness": round(chunk.spectral_flatness, 4),
            }
            return chunk

        frame = max(256, int(sample_rate * 0.025))
        hop = max(128, int(sample_rate * 0.01))
        n_fft = 1 << math.ceil(math.log2(frame))
        window = np.hanning(frame).astype(np.float32)
        frames = _frame_audio(signal, frame=frame, hop=hop)
        if frames.size == 0:
            rms = _rms(signal)
            return EnhancementChunk(
                audio=signal.astype(np.float32, copy=False),
                noise_floor_rms=rms,
                suppression_ratio=0.0,
                music_likelihood=0.0,
                spectral_flatness=1.0,
            )

        windowed = frames * window[None, :]
        spectrum = np.fft.rfft(windowed, n=n_fft, axis=1)
        magnitude = np.abs(spectrum).astype(np.float32, copy=False)
        phase = np.angle(spectrum).astype(np.float32, copy=False)

        flatness = _spectral_flatness(magnitude)
        flux = _spectral_flux(magnitude)
        noise_candidate = np.percentile(magnitude, 30.0 if speech_ratio < 0.3 else 18.0, axis=0).astype(np.float32)
        if self._noise_profile is None or self._noise_profile.shape != noise_candidate.shape:
            self._noise_profile = noise_candidate.copy()
        else:
            adapt_rate = self._noise_adapt_rate * (0.45 if speech_ratio >= 0.35 else 1.0)
            self._noise_profile = (
                self._noise_profile * (1.0 - adapt_rate) + noise_candidate * adapt_rate
            ).astype(np.float32, copy=False)

        noise_profile = np.maximum(self._noise_profile, 1e-6)
        gain = 1.0 - self._noise_reduce_strength * (noise_profile[None, :] / (magnitude + noise_profile[None, :]))
        gain = np.clip(gain, self._min_gain, 1.0).astype(np.float32, copy=False)

        music_likelihood = _music_likelihood(flatness=flatness, flux=flux, speech_ratio=speech_ratio)
        if music_likelihood > 0.01 and self._music_suppress_strength > 0.0:
            freq_mask = _music_bin_mask(sample_rate=sample_rate, n_fft=n_fft, bins=magnitude.shape[1])
            if np.any(freq_mask):
                local_avg = _local_average(magnitude)
                tonal_mask = magnitude > (local_avg * (1.18 + (0.22 * music_likelihood)))
                attenuation = 1.0 - (self._music_suppress_strength * music_likelihood * 0.4)
                gain[:, freq_mask & tonal_mask.any(axis=0)] *= attenuation
                gain = np.clip(gain, self._min_gain, 1.0)

        enhanced_spec = magnitude * gain * np.exp(1j * phase)
        enhanced_frames = np.fft.irfft(enhanced_spec, n=n_fft, axis=1)[:, :frame].astype(np.float32, copy=False)
        enhanced = _overlap_add(enhanced_frames, frame=frame, hop=hop, signal_len=signal.size, window=window)
        enhanced = np.clip(enhanced, -1.0, 1.0).astype(np.float32, copy=False)

        input_rms = _rms(signal)
        output_rms = _rms(enhanced)
        noise_floor_rms = float(np.sqrt(np.mean(np.square(noise_profile), dtype=np.float32)))
        suppression_ratio = max(0.0, 1.0 - (output_rms / max(input_rms, 1e-6)))
        chunk = EnhancementChunk(
            audio=enhanced,
            noise_floor_rms=noise_floor_rms,
            suppression_ratio=suppression_ratio,
            music_likelihood=music_likelihood,
            spectral_flatness=flatness,
        )
        self._last_stats = {
            "noise_floor_rms": round(chunk.noise_floor_rms, 5),
            "suppression_ratio": round(chunk.suppression_ratio, 4),
            "music_likelihood": round(chunk.music_likelihood, 4),
            "spectral_flatness": round(chunk.spectral_flatness, 4),
        }
        return chunk


def _frame_audio(signal: np.ndarray, *, frame: int, hop: int) -> np.ndarray:
    if signal.size == 0:
        return np.zeros((0, frame), dtype=np.float32)
    if signal.size < frame:
        padded = np.pad(signal, (0, frame - signal.size))
        return padded.reshape(1, frame).astype(np.float32, copy=False)
    starts = range(0, max(1, signal.size - frame + 1), hop)
    frames = [signal[start : start + frame] for start in starts]
    if not frames:
        return np.zeros((0, frame), dtype=np.float32)
    last_end = (len(frames) - 1) * hop + frame
    if last_end < signal.size:
        tail = np.pad(signal[-frame:], (0, 0))
        frames.append(tail)
    return np.stack(frames).astype(np.float32, copy=False)


def _overlap_add(
    frames: np.ndarray,
    *,
    frame: int,
    hop: int,
    signal_len: int,
    window: np.ndarray,
) -> np.ndarray:
    total_len = max(signal_len, (frames.shape[0] - 1) * hop + frame)
    out = np.zeros((total_len,), dtype=np.float32)
    norm = np.zeros((total_len,), dtype=np.float32)
    for idx, frame_audio in enumerate(frames):
        start = idx * hop
        end = start + frame
        out[start:end] += frame_audio * window
        norm[start:end] += np.square(window)
    norm = np.maximum(norm, 1e-6)
    merged = out / norm
    return merged[:signal_len].astype(np.float32, copy=False)


def _spectral_flatness(magnitude: np.ndarray) -> float:
    if magnitude.size == 0:
        return 1.0
    eps = 1e-6
    per_frame = np.exp(np.mean(np.log(magnitude + eps), axis=1)) / np.maximum(np.mean(magnitude + eps, axis=1), eps)
    return float(np.clip(np.mean(per_frame), 0.0, 1.0))


def _spectral_flux(magnitude: np.ndarray) -> float:
    if magnitude.shape[0] <= 1:
        return 0.0
    diff = np.diff(magnitude, axis=0)
    numer = np.linalg.norm(diff, axis=1)
    denom = np.linalg.norm(magnitude[:-1], axis=1) + 1e-6
    return float(np.clip(np.mean(numer / denom), 0.0, 1.5))


def _music_likelihood(*, flatness: float, flux: float, speech_ratio: float) -> float:
    tonal = max(0.0, 1.0 - flatness)
    score = 0.0
    score += max(0.0, tonal - 0.42) * 1.3
    score += max(0.0, 0.32 - speech_ratio) * 1.6
    score += max(0.0, 0.16 - flux) * 1.8
    return float(np.clip(score, 0.0, 1.0))


def _music_bin_mask(*, sample_rate: int, n_fft: int, bins: int) -> np.ndarray:
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(max(1, sample_rate)))
    freqs = freqs[:bins]
    return (freqs >= 80.0) & (freqs <= 3200.0)


def _local_average(magnitude: np.ndarray) -> np.ndarray:
    if magnitude.shape[1] < 3:
        return magnitude
    left = np.pad(magnitude[:, :-2], ((0, 0), (1, 1)), mode="edge")
    center = magnitude
    right = np.pad(magnitude[:, 2:], ((0, 0), (1, 1)), mode="edge")
    return ((left + center + right) / 3.0).astype(np.float32, copy=False)


def _rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float32)))


__all__ = ["AsrSpeechEnhancerV2", "EnhancementChunk"]
