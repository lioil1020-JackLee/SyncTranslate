from __future__ import annotations

from typing import Callable

import numpy as np
import sounddevice as sd

from app.infra.audio.capture import AudioCapture, CaptureStats
from app.infra.audio.device_registry import (
    device_tokens,
    list_indexed_devices,
    normalize_device_text,
    parse_device_selector,
    preferred_hostapi_index_for_platform,
)
from app.infra.audio.frame import ChannelPolicy


class WasapiLoopbackCaptureSource:
    def __init__(self) -> None:
        self._stream: sd.InputStream | None = None
        self._capture_stats = CaptureStats(False, False, 0.0, 0, 0.0, "")
        self._channels = 0
        self._consumers: list[Callable[[np.ndarray, float], None]] = []

    def start(
        self,
        device_name: str,
        sample_rate: int,
        chunk_ms: int,
        channels_policy: str = ChannelPolicy.STEREO_OR_MONO.value,
    ) -> None:
        self.stop()
        if not hasattr(sd, "WasapiSettings"):
            raise RuntimeError("WASAPI loopback is only available on Windows sounddevice builds")
        device_index, device_info = self._resolve_output_device(device_name)
        max_output_channels = int(device_info.get("max_output_channels", 0) or 0)
        channels = 2 if max_output_channels >= 2 else 1
        if str(channels_policy).lower() == ChannelPolicy.MONO.value:
            channels = 1
        rate = float(sample_rate or device_info.get("default_samplerate") or 48000)
        extra_settings = sd.WasapiSettings(loopback=True)
        self._stream = sd.InputStream(
            device=device_index,
            channels=channels,
            samplerate=rate,
            blocksize=max(256, int(round(rate * max(5, int(chunk_ms)) / 1000.0))),
            dtype="float32",
            extra_settings=extra_settings,
            callback=self._on_audio,
        )
        self._stream.start()
        self._channels = int(channels)
        self._capture_stats = CaptureStats(True, False, rate, 0, 0.0, "", channels=int(channels))

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
        self._capture_stats = CaptureStats(False, False, self._capture_stats.sample_rate, self._capture_stats.frame_count, self._capture_stats.level, self._capture_stats.last_error, channels=self._channels)

    def add_consumer(self, consumer: Callable[[np.ndarray, float], None]) -> None:
        if consumer not in self._consumers:
            self._consumers.append(consumer)

    def remove_consumer(self, consumer: Callable[[np.ndarray, float], None]) -> None:
        if consumer in self._consumers:
            self._consumers.remove(consumer)

    def set_gain(self, gain: float) -> None:
        del gain

    def stats(self) -> CaptureStats:
        return self._capture_stats

    def _on_audio(self, indata: np.ndarray, frames: int, _time: object, status: sd.CallbackFlags) -> None:
        audio = np.asarray(indata, dtype=np.float32).copy()
        level = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
        self._capture_stats = CaptureStats(
            True,
            False,
            self._capture_stats.sample_rate,
            int(self._capture_stats.frame_count) + int(frames),
            level,
            str(status) if status else "",
            channels=int(audio.shape[1]) if audio.ndim == 2 else 1,
        )
        for consumer in list(self._consumers):
            consumer(audio, self._capture_stats.sample_rate)

    @staticmethod
    def _resolve_output_device(device_name: str) -> tuple[int, dict[str, object]]:
        hostapi_name, requested_name = parse_device_selector(device_name)
        preferred_hostapi = preferred_hostapi_index_for_platform()
        normalized_target = normalize_device_text(requested_name)
        target_tokens = device_tokens(requested_name)
        ranked: list[tuple[int, int, int, dict[str, object]]] = []
        for idx, item in list_indexed_devices():
            if int(item.get("max_output_channels", 0) or 0) <= 0:
                continue
            item_name = str(item.get("name", ""))
            normalized_name = normalize_device_text(item_name)
            name_tokens = device_tokens(item_name)
            score = 0
            if item_name == requested_name:
                score = 500
            elif normalized_name == normalized_target:
                score = 450
            elif normalized_target and normalized_target in normalized_name:
                score = 350
            elif target_tokens and target_tokens.issubset(name_tokens):
                score = 300
            if score <= 0:
                continue
            item_hostapi = int(item.get("hostapi", -1))
            hostapi_rank = 0 if (hostapi_name and str(sd.query_hostapis()[item_hostapi].get("name", "")) == hostapi_name) or (not hostapi_name and item_hostapi == preferred_hostapi) else 1
            ranked.append((hostapi_rank, -score, idx, item))
        if not ranked:
            raise ValueError(f"Output loopback device not found or unavailable: {device_name}")
        ranked.sort()
        _, _, idx, item = ranked[0]
        return idx, item


__all__ = ["WasapiLoopbackCaptureSource"]
