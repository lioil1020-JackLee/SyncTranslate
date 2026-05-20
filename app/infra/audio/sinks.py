from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from app.infra.audio.playback import AudioPlayback


class AudioSink(Protocol):
    def play(self, audio: np.ndarray, sample_rate: float, *, blocking: bool = False) -> None: ...

    def stop(self) -> None: ...

    def set_volume(self, volume: float) -> None: ...

    def push_passthrough(self, audio: np.ndarray, sample_rate: float) -> None: ...


class SoundDevicePlaybackSink:
    def __init__(self, playback: AudioPlayback, output_device_provider: Callable[[], str]) -> None:
        self._playback = playback
        self._output_device_provider = output_device_provider

    def play(self, audio: np.ndarray, sample_rate: float, *, blocking: bool = False) -> None:
        output_device_name = str(self._output_device_provider() or "").strip()
        if not output_device_name:
            return
        self._playback.play(
            audio=np.asarray(audio, dtype=np.float32),
            sample_rate=int(sample_rate),
            output_device_name=output_device_name,
            blocking=bool(blocking),
        )

    def stop(self) -> None:
        self._playback.stop()

    def set_volume(self, volume: float) -> None:
        self._playback.set_volume(float(volume))

    def push_passthrough(self, audio: np.ndarray, sample_rate: float) -> None:
        output_device_name = str(self._output_device_provider() or "").strip()
        if not output_device_name:
            return
        self._playback.push_passthrough(
            audio=np.asarray(audio, dtype=np.float32),
            sample_rate=float(sample_rate),
            output_device_name=output_device_name,
        )


class VirtualMicrophoneSink:
    def __init__(self, bridge_client) -> None:
        self._bridge = bridge_client
        self._volume = 1.0
        self._write_failures = 0
        self._silence_fallback_writes = 0
        self._dropped_frames = 0
        self._last_error = ""
        self._backpressure_flushes = 0
        self._backpressure_probe_interval_sec = 0.2
        self._next_backpressure_probe_at = 0.0

    def play(self, audio: np.ndarray, sample_rate: float, *, blocking: bool = False) -> None:
        del blocking
        payload = np.asarray(audio, dtype=np.float32)
        if payload.size == 0:
            return
        if self._volume != 1.0:
            payload = payload.copy()
            payload *= self._volume
            np.clip(payload, -1.0, 1.0, out=payload)
        try:
            self._maybe_flush_for_backpressure(payload)
            self._bridge.write_virtual_microphone(payload, sample_rate=int(sample_rate))
            self._last_error = ""
        except Exception as exc:
            self._write_failures += 1
            self._last_error = str(exc)
            self._dropped_frames += int(payload.shape[0]) if payload.ndim else int(payload.size)
            # 當 bridge 異常時，避免把錯誤往上層傳遞導致整個 session 中斷。
            try:
                silence = np.zeros_like(payload, dtype=np.float32)
                self._bridge.write_virtual_microphone(silence, sample_rate=int(sample_rate))
                self._silence_fallback_writes += 1
            except Exception:
                pass

    def _maybe_flush_for_backpressure(self, payload: np.ndarray) -> None:
        try:
            import time

            now = time.monotonic()
            if now < self._next_backpressure_probe_at:
                return
            self._next_backpressure_probe_at = now + self._backpressure_probe_interval_sec
            stats_fn = getattr(self._bridge, "stats", None)
            if not callable(stats_fn):
                return
            stats = stats_fn()
            buffered = int(getattr(stats, "virtual_microphone_buffered_frames", 0) or 0)
            capacity = int(getattr(stats, "virtual_microphone_buffer_capacity_frames", 0) or 0)
            if capacity <= 0:
                return
            incoming_frames = int(payload.shape[0]) if payload.ndim else int(payload.size)
            if buffered + max(0, incoming_frames) < capacity:
                return
            self._bridge.flush_virtual_microphone()
            self._backpressure_flushes += 1
        except Exception:
            # Backpressure probing is best-effort and should never interrupt audio flow.
            return

    def stop(self) -> None:
        self._bridge.flush_virtual_microphone()

    def set_volume(self, volume: float) -> None:
        self._volume = max(0.0, float(volume))

    def push_passthrough(self, audio: np.ndarray, sample_rate: float) -> None:
        self.play(audio, sample_rate, blocking=False)

    def diagnostic_stats(self) -> dict[str, object]:
        return {
            "write_failures": int(self._write_failures),
            "silence_fallback_writes": int(self._silence_fallback_writes),
            "dropped_frames": int(self._dropped_frames),
            "backpressure_flushes": int(self._backpressure_flushes),
            "last_error": str(self._last_error or ""),
        }


__all__ = ["AudioSink", "SoundDevicePlaybackSink", "VirtualMicrophoneSink"]
