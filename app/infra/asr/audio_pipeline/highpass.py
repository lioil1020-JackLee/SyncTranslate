"""High-pass filter stage (first-order IIR pre-emphasis)."""
from __future__ import annotations

import numpy as np


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


__all__ = ["HighpassStage"]
