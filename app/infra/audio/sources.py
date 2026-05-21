from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from app.infra.audio.capture import AudioCapture, CaptureStats


AudioConsumer = Callable[[np.ndarray, float], None]


class AudioSource(Protocol):
    def start(self, device_name: str, sample_rate: int, chunk_ms: int) -> None: ...

    def stop(self) -> None: ...

    def add_consumer(self, consumer: AudioConsumer) -> None: ...

    def remove_consumer(self, consumer: AudioConsumer) -> None: ...

    def set_gain(self, gain: float) -> None: ...

    def stats(self) -> CaptureStats: ...


class SoundDeviceCaptureSource:
    def __init__(self, capture: AudioCapture | None = None) -> None:
        self._capture = capture or AudioCapture()

    def start(self, device_name: str, sample_rate: int, chunk_ms: int) -> None:
        self._capture.start(device_name, sample_rate=sample_rate, chunk_ms=chunk_ms)

    def stop(self) -> None:
        self._capture.stop()

    def add_consumer(self, consumer: AudioConsumer) -> None:
        self._capture.add_consumer(consumer)

    def remove_consumer(self, consumer: AudioConsumer) -> None:
        self._capture.remove_consumer(consumer)

    def set_gain(self, gain: float) -> None:
        self._capture.set_gain(gain)

    def stats(self) -> CaptureStats:
        return self._capture.stats()


class VirtualSpeakerSource:
    def __init__(self, bridge_client) -> None:
        self._bridge = bridge_client
        self._consumers: list[AudioConsumer] = []
        self._running = False
        self._sample_rate = 0
        self._chunk_frames = 0
        self._frame_count = 0
        self._level = 0.0
        self._last_error = ""
        self._gain = 1.0
        self._pending = np.zeros((0, 1), dtype=np.float32)

    def start(self, device_name: str, sample_rate: int, chunk_ms: int) -> None:
        import logging
        logger = logging.getLogger(__name__)
        
        if not device_name:
            logger.error("VirtualSpeakerSource.start() called with empty device_name")
            raise ValueError("device_name cannot be empty for VirtualSpeakerSource")
        
        # CRITICAL: Remote speaker audio must always be captured at 48kHz
        # to preserve quality. ASR sample rate (e.g., 16kHz) applies only to
        # local audio capture. Do NOT resample remote audio to ASR rates.
        remote_sample_rate = 48000  # Bridge outputs speaker audio at 48kHz
        
        logger.info(
            f"VirtualSpeakerSource.start: device_name={device_name!r}, "
            f"requested_sample_rate={sample_rate}, remote_sample_rate={remote_sample_rate}, chunk_ms={chunk_ms}"
        )
        
        try:
            self._bridge.start_remote_input(
                sample_rate=remote_sample_rate,
                device_name=str(device_name or ""),
                chunk_ms=int(chunk_ms),
            )
            self._sample_rate = remote_sample_rate
            self._chunk_frames = max(1, int(self._sample_rate * max(5, int(chunk_ms)) / 1000))
            self._pending = np.zeros((0, 1), dtype=np.float32)
            self._running = True
            self._last_error = ""
        except Exception as exc:
            self._running = False
            self._last_error = str(exc)
            logger.exception(f"VirtualSpeakerSource.start() failed: {exc}")
            raise

    def stop(self) -> None:
        try:
            self._bridge.stop_remote_input()
        finally:
            self._running = False
            self._pending = np.zeros((0, 1), dtype=np.float32)

    def add_consumer(self, consumer: AudioConsumer) -> None:
        import logging
        logger = logging.getLogger(__name__)
        
        if consumer not in self._consumers:
            self._consumers.append(consumer)
            logger.info(f"VirtualSpeakerSource.add_consumer: total_consumers={len(self._consumers)}")
        
        # Always register bridge callback to ensure remote input thread is started/maintained
        logger.debug("Registering bridge remote_input_consumer callback")
        self._bridge.add_remote_input_consumer(self._dispatch_audio)

    def remove_consumer(self, consumer: AudioConsumer) -> None:
        if consumer in self._consumers:
            self._consumers.remove(consumer)
        if not self._consumers:
            self._bridge.remove_remote_input_consumer(self._dispatch_audio)

    def set_gain(self, gain: float) -> None:
        self._gain = max(0.0, float(gain))

    def stats(self) -> CaptureStats:
        return CaptureStats(
            running=self._running,
            muted=False,
            sample_rate=float(self._sample_rate),
            frame_count=self._frame_count,
            level=self._level,
            last_error=self._last_error,
        )

    def _dispatch_audio(self, audio: np.ndarray, sample_rate: float) -> None:
        processed = audio.astype(np.float32, copy=True)
        if processed.ndim == 1:
            processed = processed.reshape((-1, 1))
        if self._gain != 1.0:
            processed *= self._gain
            np.clip(processed, -1.0, 1.0, out=processed)
        if self._pending.size:
            processed = np.concatenate((self._pending, processed), axis=0)

        chunk_frames = max(1, int(self._chunk_frames or 0))
        emitted_any = False
        while processed.shape[0] >= chunk_frames:
            chunk = processed[:chunk_frames]
            processed = processed[chunk_frames:]
            self._emit_chunk(chunk, sample_rate)
            emitted_any = True

        self._pending = processed
        if not emitted_any and processed.size:
            # 低流量時仍允許即時送出，避免尾端音訊被長時間卡在 pending。
            self._emit_chunk(processed, sample_rate)
            self._pending = np.zeros((0, processed.shape[1]), dtype=np.float32)

    def _emit_chunk(self, chunk: np.ndarray, sample_rate: float) -> None:
        self._frame_count += int(chunk.shape[0]) if chunk.ndim else int(chunk.size)
        self._level = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
        for consumer in list(self._consumers):
            consumer(chunk, sample_rate)


__all__ = [
    "AudioConsumer",
    "AudioSource",
    "SoundDeviceCaptureSource",
    "VirtualSpeakerSource",
]
