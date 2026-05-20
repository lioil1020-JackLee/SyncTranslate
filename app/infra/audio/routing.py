from __future__ import annotations

from typing import Callable

import numpy as np

from app.infra.audio.capture import AudioCapture, CaptureStats


class _AudioInputEndpoint:
    def start(self, device_name: str, sample_rate: int, chunk_ms: int) -> None: ...

    def stop(self) -> None: ...

    def add_consumer(self, consumer: Callable[[np.ndarray, float], None]) -> None: ...

    def remove_consumer(self, consumer: Callable[[np.ndarray, float], None]) -> None: ...

    def set_gain(self, gain: float) -> None: ...

    def stats(self) -> CaptureStats: ...


class AudioInputManager:
    def __init__(
        self,
        *,
        local_source: _AudioInputEndpoint | None = None,
        remote_source: _AudioInputEndpoint | None = None,
        local_capture: AudioCapture | None = None,
        remote_capture: AudioCapture | None = None,
    ) -> None:
        local_endpoint = local_source or local_capture
        remote_endpoint = remote_source or remote_capture
        if local_endpoint is None or remote_endpoint is None:
            raise TypeError("AudioInputManager requires local_source/remote_source or local_capture/remote_capture")
        self._captures = {
            "local": local_endpoint,
            "remote": remote_endpoint,
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

    def _capture_of(self, source: str) -> _AudioInputEndpoint:
        if source == "remote":
            return self._captures["remote"]
        return self._captures["local"]


AudioRoutingManager = AudioInputManager

__all__ = ["AudioInputManager", "AudioRoutingManager", "CaptureStats", "AudioCapture"]
