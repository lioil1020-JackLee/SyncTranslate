from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class AudioLevel:
    rms: float
    peak: float


def measure_level(audio: np.ndarray) -> AudioLevel:
    if audio.size == 0:
        return AudioLevel(rms=0.0, peak=0.0)
    data = audio.astype(np.float32, copy=False)
    rms = float(np.sqrt(np.mean(np.square(data))))
    peak = float(np.max(np.abs(data)))
    return AudioLevel(rms=rms, peak=peak)


__all__ = ["AudioLevel", "measure_level"]
