from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time

import numpy as np


class ChannelPolicy(str, Enum):
    MONO = "mono"
    STEREO = "stereo"
    STEREO_OR_MONO = "stereo_or_mono"
    NATIVE = "native"


class CaptureKind(str, Enum):
    INPUT = "input"
    OUTPUT_LOOPBACK = "output_loopback"
    VIRTUAL_REMOTE = "virtual_remote"


@dataclass(frozen=True, slots=True)
class AudioFormat:
    sample_rate: int
    channels: int
    dtype: str = "float32"
    layout: str = "unknown"


@dataclass(slots=True)
class AudioFrame:
    samples: np.ndarray
    sample_rate: int
    channels: int
    source_id: str
    role: str
    timestamp_monotonic: float
    layout: str = "unknown"
    source_type: str = "input"

    @classmethod
    def from_samples(
        cls,
        samples: np.ndarray,
        *,
        sample_rate: int,
        source_id: str,
        role: str,
        source_type: str = "input",
        layout: str | None = None,
        timestamp_monotonic: float | None = None,
    ) -> "AudioFrame":
        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio.reshape((-1, 1))
        elif audio.ndim > 2:
            audio = audio.reshape((audio.shape[0], -1))
        channels = int(audio.shape[1]) if audio.ndim == 2 else 1
        resolved_layout = layout or ("mono" if channels == 1 else "stereo" if channels == 2 else "unknown")
        return cls(
            samples=audio.astype(np.float32, copy=False),
            sample_rate=int(sample_rate),
            channels=channels,
            source_id=str(source_id or ""),
            role=str(role or ""),
            timestamp_monotonic=time.monotonic() if timestamp_monotonic is None else float(timestamp_monotonic),
            layout=resolved_layout,
            source_type=str(source_type or "input"),
        )


__all__ = ["AudioFormat", "AudioFrame", "CaptureKind", "ChannelPolicy"]
