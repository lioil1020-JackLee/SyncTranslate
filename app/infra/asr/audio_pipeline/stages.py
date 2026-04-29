"""All audio-processing pipeline stage classes in one module.

Previously these were split across five micro-files:
  identity.py, highpass.py, noise_reduction.py, loudness.py, music_suppression.py
"""
from __future__ import annotations

import numpy as np

from app.infra.asr.enhancement_v2 import AsrSpeechEnhancerV2


class IdentityStage:
    """No-op passthrough — used as a placeholder or fallback."""

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        return audio

    def reset(self) -> None:
        pass


class HighpassStage:
    """First-order IIR high-pass / pre-emphasis filter.

    Removes DC offset and boosts high-frequency content, which helps ASR
    models trained on pre-emphasized speech.

    Parameters
    ----------
    alpha:
        Filter coefficient in [0, 1).  Higher = stronger high-pass.
        0.96 is a common value for speech pre-emphasis.
    enabled:
        When False this stage acts as an identity passthrough.
    """

    def __init__(self, *, alpha: float = 0.96, enabled: bool = True) -> None:
        self._alpha = min(0.999, max(0.0, float(alpha)))
        self._enabled = bool(enabled)
        self._prev_input: float = 0.0
        self._prev_output: float = 0.0

    def reset(self) -> None:
        self._prev_input = 0.0
        self._prev_output = 0.0

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:  # noqa: ARG002
        if not self._enabled or audio.size == 0:
            return audio
        audio = np.asarray(audio, dtype=np.float32)
        out = np.empty_like(audio)
        alpha = self._alpha
        prev_in = self._prev_input
        prev_out = self._prev_output
        for i in range(len(audio)):
            x = audio[i]
            y = x - alpha * prev_in + alpha * prev_out
            out[i] = y
            prev_in = x
            prev_out = y
        self._prev_input = float(prev_in)
        self._prev_output = float(prev_out)
        return out


class NoiseReductionStage:
    """Adaptive stationary-noise suppression.

    Thin wrapper around :class:`AsrSpeechEnhancerV2` that disables music
    suppression so it focuses purely on stationary noise.

    Parameters
    ----------
    strength:
        Noise reduction strength in [0, 1].
    adapt_rate:
        How fast the noise floor estimate adapts.
    enabled:
        When False this stage acts as an identity passthrough.
    """

    def __init__(
        self,
        *,
        strength: float = 0.42,
        adapt_rate: float = 0.18,
        enabled: bool = True,
    ) -> None:
        self._enabled = bool(enabled)
        self._enhancer = AsrSpeechEnhancerV2(
            enabled=enabled,
            noise_reduce_strength=strength,
            noise_adapt_rate=adapt_rate,
            music_suppress_strength=0.0,  # handled by MusicSuppressionStage
        )

    def reset(self) -> None:
        self._enhancer.reset()

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self._enabled or audio.size == 0:
            return audio
        result = self._enhancer.process(np.asarray(audio, dtype=np.float32), sample_rate, speech_ratio=0.5)
        return result.audio

    def stats(self) -> dict[str, float]:
        return self._enhancer.stats()


class LoudnessStage:
    """Mild per-chunk automatic gain control.

    Boosts quiet speech towards *target_rms* without exceeding *max_gain*,
    and hard-clips to [-1, 1] to prevent downstream distortion.

    Parameters
    ----------
    target_rms:
        Desired RMS of the output chunk.
    max_gain:
        Maximum amplification factor.
    enabled:
        When False this stage acts as an identity passthrough.
    """

    def __init__(
        self,
        *,
        target_rms: float = 0.05,
        max_gain: float = 3.0,
        enabled: bool = True,
    ) -> None:
        self._target_rms = max(0.001, float(target_rms))
        self._max_gain = max(1.0, float(max_gain))
        self._enabled = bool(enabled)
        self._last_gain: float = 1.0

    def reset(self) -> None:
        self._last_gain = 1.0

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:  # noqa: ARG002
        if not self._enabled or audio.size == 0:
            return audio
        audio = np.asarray(audio, dtype=np.float32)
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms > 1e-7:
            gain = min(self._max_gain, self._target_rms / rms)
        else:
            gain = 1.0
        self._last_gain = gain
        out = np.clip(audio * gain, -1.0, 1.0)
        return out.astype(np.float32)


class MusicSuppressionStage:
    """Conservative tonal/music-bed attenuation.

    Uses the music detection inside :class:`AsrSpeechEnhancerV2` but disables
    stationary noise reduction so the two stages remain independent.

    Parameters
    ----------
    strength:
        Music suppression strength in [0, 1].
    enabled:
        When False this stage acts as an identity passthrough.
    """

    def __init__(self, *, strength: float = 0.2, enabled: bool = True) -> None:
        self._enabled = bool(enabled)
        self._enhancer = AsrSpeechEnhancerV2(
            enabled=enabled,
            noise_reduce_strength=0.0,  # handled by NoiseReductionStage
            noise_adapt_rate=0.18,
            music_suppress_strength=strength,
        )

    def reset(self) -> None:
        self._enhancer.reset()

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self._enabled or audio.size == 0:
            return audio
        result = self._enhancer.process(np.asarray(audio, dtype=np.float32), sample_rate, speech_ratio=0.5)
        return result.audio

    def stats(self) -> dict[str, float]:
        return self._enhancer.stats()


__all__ = [
    "HighpassStage",
    "IdentityStage",
    "LoudnessStage",
    "MusicSuppressionStage",
    "NoiseReductionStage",
]
