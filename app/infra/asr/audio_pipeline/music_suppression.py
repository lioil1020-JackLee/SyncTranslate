"""Music suppression stage."""
from __future__ import annotations

import numpy as np

from app.infra.asr.enhancement_v2 import AsrSpeechEnhancerV2


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


__all__ = ["MusicSuppressionStage"]
