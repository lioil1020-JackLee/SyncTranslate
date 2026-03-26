from __future__ import annotations

from typing import Callable

import numpy as np

from app.infra.audio.capture import AudioCapture, CaptureStats


class AudioInputManager:
    def __init__(self, *, local_capture: AudioCapture, remote_capture: AudioCapture) -> None:
        self._captures = {
            "local": local_capture,
            "remote": remote_capture,
        }

    def start(self, source: str, device_name: str, sample_rate: int, chunk_ms: int) -> None:
        capture = self._capture_of(source)
        capture.start(device_name, sample_rate=sample_rate, chunk_ms=chunk_ms)

    def stop(self, source: str) -> None:
        self._capture_of(source).stop()

    def stop_all(self) -> None:
        for capture in self._captures.values():
            capture.stop()

    def add_consumer(self, source: str, consumer: Callable[[np.ndarray, float], None]) -> None:
        self._capture_of(source).add_consumer(consumer)

    def remove_consumer(self, source: str, consumer: Callable[[np.ndarray, float], None]) -> None:
        self._capture_of(source).remove_consumer(consumer)

    def set_gain(self, source: str, gain: float) -> None:
        self._capture_of(source).set_gain(gain)

    def stats(self) -> dict[str, CaptureStats]:
        return {
            "local": self._captures["local"].stats(),
            "remote": self._captures["remote"].stats(),
        }

    def _capture_of(self, source: str) -> AudioCapture:
        if source == "remote":
            return self._captures["remote"]
        return self._captures["local"]


AudioRoutingManager = AudioInputManager

__all__ = ["AudioInputManager", "AudioRoutingManager", "CaptureStats", "AudioCapture"]
