"""Identity stage — passthrough, no processing."""
from __future__ import annotations

import numpy as np


class IdentityStage:
    """No-op stage that returns the input unchanged."""

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        return audio

    def reset(self) -> None:
        pass


__all__ = ["IdentityStage"]
