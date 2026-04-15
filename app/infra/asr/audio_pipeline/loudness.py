"""Loudness / AGC stage — mild automatic gain control."""
from __future__ import annotations

import numpy as np


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

    @property
    def last_gain(self) -> float:
        return self._last_gain


__all__ = ["LoudnessStage"]
