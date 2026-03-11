from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Callable

import numpy as np
import sounddevice as sd

from app.audio_device_selection import list_indexed_devices, pick_best_device


@dataclass(slots=True)
class CaptureStats:
    running: bool
    muted: bool
    sample_rate: float
    frame_count: int
    level: float
    last_error: str


class AudioCapture:
    def __init__(self) -> None:
        self._stream: sd.InputStream | None = None
        self._lock = Lock()
        self._running = False
        self._sample_rate = 0.0
        self._frame_count = 0
        self._level = 0.0
        self._last_error = ""
        self._muted = False
        self._current_device_name = ""
        self._current_sample_rate: int | None = None
        self._consumers: list[Callable[[np.ndarray, float], None]] = []

    def start(self, device_name: str, sample_rate: int | None = None) -> None:
        if not device_name:
            raise ValueError("Input device name is required.")

        self.stop()

        device_index, device_info = self._find_input_device(device_name)
        max_input_channels = int(device_info["max_input_channels"])
        channels = 1 if max_input_channels >= 1 else 0
        if channels <= 0:
            raise ValueError(f"Device has no input channels: {device_name}")

        requested_sample_rate = float(sample_rate) if sample_rate else None
        default_sample_rate = float(device_info["default_samplerate"])
        resolved_sample_rate = self._resolve_supported_input_sample_rate(
            device_index=device_index,
            channels=channels,
            requested_sample_rate=requested_sample_rate,
            default_sample_rate=default_sample_rate,
        )
        with self._lock:
            self._frame_count = 0
            self._level = 0.0
            self._sample_rate = resolved_sample_rate
            self._last_error = ""
            self._current_device_name = device_name
            self._current_sample_rate = int(round(resolved_sample_rate))

        self._stream = sd.InputStream(
            device=device_index,
            channels=channels,
            samplerate=resolved_sample_rate,
            callback=self._on_audio,
        )
        self._stream.start()
        with self._lock:
            self._running = True

    def stop(self) -> None:
        if not self._stream:
            with self._lock:
                self._running = False
            return

        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None
            with self._lock:
                self._running = False

    def set_muted(self, muted: bool) -> None:
        with self._lock:
            self._muted = muted

    def rebind(self) -> None:
        with self._lock:
            device_name = self._current_device_name
            sample_rate = self._current_sample_rate
        if not device_name:
            raise ValueError("No previous device to rebind.")
        self.start(device_name, sample_rate)

    def stats(self) -> CaptureStats:
        with self._lock:
            return CaptureStats(
                running=self._running,
                muted=self._muted,
                sample_rate=self._sample_rate,
                frame_count=self._frame_count,
                level=self._level,
                last_error=self._last_error,
            )

    def add_consumer(self, consumer: Callable[[np.ndarray, float], None]) -> None:
        with self._lock:
            if consumer not in self._consumers:
                self._consumers.append(consumer)

    def remove_consumer(self, consumer: Callable[[np.ndarray, float], None]) -> None:
        with self._lock:
            if consumer in self._consumers:
                self._consumers.remove(consumer)

    def _on_audio(self, indata: np.ndarray, frames: int, _time: object, status: sd.CallbackFlags) -> None:
        level = float(np.sqrt(np.mean(np.square(indata)))) if indata.size else 0.0
        with self._lock:
            self._frame_count += int(frames)
            self._level = 0.0 if self._muted else level
            if status:
                self._last_error = str(status)
            consumers = list(self._consumers)

        for consumer in consumers:
            try:
                consumer(indata.copy(), self._sample_rate)
            except Exception:
                continue

    @staticmethod
    def _find_input_device(device_name: str) -> tuple[int, dict[str, object]]:
        devices = list_indexed_devices()
        exact_matches = [
            (idx, item)
            for idx, item in devices
            if str(item["name"]) == device_name and int(item["max_input_channels"]) > 0
        ]
        best_exact = pick_best_device(exact_matches)
        if best_exact:
            return best_exact

        partial_matches = [
            (idx, item)
            for idx, item in devices
            if device_name.lower() in str(item["name"]).lower() and int(item["max_input_channels"]) > 0
        ]
        best_partial = pick_best_device(partial_matches)
        if best_partial:
            return best_partial
        raise ValueError(f"Input device not found: {device_name}")

    @staticmethod
    def _resolve_supported_input_sample_rate(
        *,
        device_index: int,
        channels: int,
        requested_sample_rate: float | None,
        default_sample_rate: float,
    ) -> float:
        candidate_rates: list[float] = []
        if requested_sample_rate and requested_sample_rate > 0:
            candidate_rates.append(requested_sample_rate)
        if default_sample_rate > 0 and default_sample_rate not in candidate_rates:
            candidate_rates.append(default_sample_rate)

        errors: list[str] = []
        for rate in candidate_rates:
            try:
                sd.check_input_settings(device=device_index, channels=channels, samplerate=rate)
                return rate
            except Exception as exc:
                errors.append(f"{int(round(rate))}Hz -> {exc}")

        details = "; ".join(errors) if errors else "no valid candidate sample rate"
        raise ValueError(f"Input device does not support requested sample rate(s): {details}")
