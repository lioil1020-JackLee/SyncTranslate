"""Stationary noise reduction stage (wraps AsrSpeechEnhancerV2)."""
from __future__ import annotations

import numpy as np

from app.infra.asr.enhancement_v2 import AsrSpeechEnhancerV2


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


__all__ = ["NoiseReductionStage"]
