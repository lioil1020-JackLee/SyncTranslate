"""Base protocols for audio processing stages.

Each stage is a stateful processor that takes a mono float32 ndarray and
sample rate, and returns a processed ndarray of the same dtype.

Design goals
------------
- Protocol-based for duck typing; no hard inheritance required.
- ``ReferenceAwareAudioStage`` is the extension point for future AEC:
  it accepts a ``reference`` audio array (e.g. speaker playback) alongside
  the microphone signal. Wire in playback audio here to enable echo
  cancellation in a future stage.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class AudioProcessingStage(Protocol):
    """Minimal protocol all pipeline stages must satisfy."""

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Process *audio* and return the processed ndarray.

        Parameters
        ----------
        audio:
            Mono float32 ndarray, typically 16 kHz.
        sample_rate:
            Sample rate in Hz of *audio*.

        Returns
        -------
        np.ndarray
            Processed mono float32 audio.  May be the same length or shorter
            than the input (e.g. after resampling).
        """
        ...

    def reset(self) -> None:
        """Reset internal state (called between utterances / on channel restart)."""
        ...


@runtime_checkable
class ReferenceAwareAudioStage(Protocol):
    """Extension point for AEC and reference-signal-aware stages.

    AEC integration point
    ---------------------
    When playback / speaker reference audio is available (e.g. from a
    VoiceMeeter loopback or a rendered mix-down buffer), pass it as
    ``reference``.  A concrete subclass can implement WebRTC APM or any
    other echo cancellation algorithm here.

    Currently no built-in concrete implementation is provided; this protocol
    exists so that future AEC stages can slot in without changing the
    pipeline assembly code.
    """

    def process_with_reference(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        reference: np.ndarray | None = None,
    ) -> np.ndarray:
        """Process *audio* with an optional speaker reference signal.

        Parameters
        ----------
        audio:
            Mono float32 mic signal.
        sample_rate:
            Sample rate in Hz.
        reference:
            Optional mono float32 speaker playback signal, same sample rate.
            Pass ``None`` when playback reference is unavailable.
        """
        ...

    def reset(self) -> None: ...


__all__ = ["AudioProcessingStage", "ReferenceAwareAudioStage"]
