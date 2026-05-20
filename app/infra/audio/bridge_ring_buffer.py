from __future__ import annotations

import struct
import uuid
from dataclasses import dataclass
from multiprocessing import shared_memory
from threading import Lock

import numpy as np

from app.infra.audio.bridge_protocol import audio_frame_count


@dataclass(frozen=True, slots=True)
class RingBufferStats:
    capacity_frames: int
    buffered_frames: int
    total_written_frames: int
    dropped_frames: int
    shared_memory_name: str = ""


class MockPcmRingBuffer:
    """In-process PCM ring buffer used when shared memory is unavailable."""

    def __init__(self, *, capacity_frames: int = 48000 * 10, channels: int = 1) -> None:
        self.capacity_frames = max(1, int(capacity_frames))
        self.channels = max(1, int(channels))
        self._lock = Lock()
        self._buffer = np.zeros((self.capacity_frames, self.channels), dtype=np.float32)
        self._write_pos = 0
        self._buffered_frames = 0
        self._total_written_frames = 0
        self._dropped_frames = 0

    def write(self, audio: np.ndarray) -> int:
        payload = self._normalize(audio)
        frames = audio_frame_count(payload)
        if frames <= 0:
            return 0
        if frames > self.capacity_frames:
            dropped = frames - self.capacity_frames
            payload = payload[-self.capacity_frames :]
            frames = self.capacity_frames
        else:
            dropped = max(0, self._buffered_frames + frames - self.capacity_frames)

        with self._lock:
            end = self._write_pos + frames
            if end <= self.capacity_frames:
                self._buffer[self._write_pos : end] = payload
            else:
                first = self.capacity_frames - self._write_pos
                self._buffer[self._write_pos :] = payload[:first]
                self._buffer[: end % self.capacity_frames] = payload[first:]
            self._write_pos = end % self.capacity_frames
            self._buffered_frames = min(self.capacity_frames, self._buffered_frames + frames)
            self._total_written_frames += frames
            self._dropped_frames += dropped
        return frames

    def snapshot(self) -> np.ndarray:
        with self._lock:
            frames = self._buffered_frames
            if frames <= 0:
                return np.zeros((0, self.channels), dtype=np.float32)
            start = (self._write_pos - frames) % self.capacity_frames
            if start + frames <= self.capacity_frames:
                return self._buffer[start : start + frames].copy()
            first = self.capacity_frames - start
            return np.concatenate((self._buffer[start:], self._buffer[: frames - first]), axis=0)

    def clear(self) -> None:
        with self._lock:
            self._write_pos = 0
            self._buffered_frames = 0

    def stats(self) -> RingBufferStats:
        with self._lock:
            return RingBufferStats(
                capacity_frames=self.capacity_frames,
                buffered_frames=self._buffered_frames,
                total_written_frames=self._total_written_frames,
                dropped_frames=self._dropped_frames,
            )

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        payload = np.asarray(audio, dtype=np.float32)
        if payload.ndim == 0:
            payload = payload.reshape((1, 1))
        elif payload.ndim == 1:
            payload = payload.reshape((-1, 1))
        if payload.shape[1] == self.channels:
            return np.ascontiguousarray(payload)
        if payload.shape[1] > self.channels:
            return np.ascontiguousarray(payload[:, : self.channels])
        repeats = self.channels - payload.shape[1]
        padding = np.repeat(payload[:, -1:], repeats, axis=1)
        return np.ascontiguousarray(np.concatenate((payload, padding), axis=1))


_HEADER_FORMAT = "<6q"
_HEADER_SIZE = 64


class SharedMemoryPcmRingBuffer:
    """Shared-memory PCM ring buffer used as the driver-facing contract."""

    def __init__(
        self,
        *,
        memory: shared_memory.SharedMemory,
        capacity_frames: int,
        channels: int,
        owner: bool,
    ) -> None:
        self.memory = memory
        self.capacity_frames = max(1, int(capacity_frames))
        self.channels = max(1, int(channels))
        self.owner = owner
        self._lock = Lock()
        self._data_offset = _HEADER_SIZE
        self._data_shape = (self.capacity_frames, self.channels)
        self._data_nbytes = self.capacity_frames * self.channels * np.dtype(np.float32).itemsize
        self._buffer = np.ndarray(self._data_shape, dtype=np.float32, buffer=self.memory.buf, offset=self._data_offset)
        if owner:
            self._buffer.fill(0.0)
            self._write_header(write_pos=0, buffered_frames=0, total_written_frames=0, dropped_frames=0)

    @classmethod
    def create(
        cls,
        *,
        name: str | None = None,
        capacity_frames: int = 48000 * 10,
        channels: int = 1,
    ) -> "SharedMemoryPcmRingBuffer":
        capacity = max(1, int(capacity_frames))
        channel_count = max(1, int(channels))
        memory_name = name or f"synctranslate_audio_{uuid.uuid4().hex}"
        size = _HEADER_SIZE + capacity * channel_count * np.dtype(np.float32).itemsize
        memory = shared_memory.SharedMemory(name=memory_name, create=True, size=size)
        return cls(memory=memory, capacity_frames=capacity, channels=channel_count, owner=True)

    @classmethod
    def attach(
        cls,
        *,
        name: str,
        capacity_frames: int,
        channels: int,
    ) -> "SharedMemoryPcmRingBuffer":
        memory = shared_memory.SharedMemory(name=name, create=False)
        return cls(memory=memory, capacity_frames=capacity_frames, channels=channels, owner=False)

    def write(self, audio: np.ndarray) -> int:
        payload = self._normalize(audio)
        frames = audio_frame_count(payload)
        if frames <= 0:
            return 0

        with self._lock:
            write_pos, buffered_frames, total_written_frames, dropped_frames = self._read_positions()
            if frames > self.capacity_frames:
                dropped = frames - self.capacity_frames
                payload = payload[-self.capacity_frames :]
                frames = self.capacity_frames
            else:
                dropped = max(0, buffered_frames + frames - self.capacity_frames)

            end = write_pos + frames
            if end <= self.capacity_frames:
                self._buffer[write_pos:end] = payload
            else:
                first = self.capacity_frames - write_pos
                self._buffer[write_pos:] = payload[:first]
                self._buffer[: end % self.capacity_frames] = payload[first:]

            self._write_header(
                write_pos=end % self.capacity_frames,
                buffered_frames=min(self.capacity_frames, buffered_frames + frames),
                total_written_frames=total_written_frames + frames,
                dropped_frames=dropped_frames + dropped,
            )
        return frames

    def snapshot(self) -> np.ndarray:
        with self._lock:
            write_pos, buffered_frames, _total_written_frames, _dropped_frames = self._read_positions()
            if buffered_frames <= 0:
                return np.zeros((0, self.channels), dtype=np.float32)
            start = (write_pos - buffered_frames) % self.capacity_frames
            if start + buffered_frames <= self.capacity_frames:
                return self._buffer[start : start + buffered_frames].copy()
            first = self.capacity_frames - start
            return np.concatenate((self._buffer[start:], self._buffer[: buffered_frames - first]), axis=0)

    def clear(self) -> None:
        with self._lock:
            _write_pos, _buffered_frames, total_written_frames, dropped_frames = self._read_positions()
            self._write_header(
                write_pos=0,
                buffered_frames=0,
                total_written_frames=total_written_frames,
                dropped_frames=dropped_frames,
            )

    def stats(self) -> RingBufferStats:
        with self._lock:
            _write_pos, buffered_frames, total_written_frames, dropped_frames = self._read_positions()
            return RingBufferStats(
                capacity_frames=self.capacity_frames,
                buffered_frames=buffered_frames,
                total_written_frames=total_written_frames,
                dropped_frames=dropped_frames,
                shared_memory_name=self.memory.name,
            )

    def close(self) -> None:
        self._buffer = None  # type: ignore[assignment]
        self.memory.close()

    def unlink(self) -> None:
        if not self.owner:
            return
        try:
            self.memory.unlink()
        except FileNotFoundError:
            pass

    def close_and_unlink(self) -> None:
        self.close()
        self.unlink()

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        payload = np.asarray(audio, dtype=np.float32)
        if payload.ndim == 0:
            payload = payload.reshape((1, 1))
        elif payload.ndim == 1:
            payload = payload.reshape((-1, 1))
        if payload.shape[1] == self.channels:
            return np.ascontiguousarray(payload)
        if payload.shape[1] > self.channels:
            return np.ascontiguousarray(payload[:, : self.channels])
        repeats = self.channels - payload.shape[1]
        padding = np.repeat(payload[:, -1:], repeats, axis=1)
        return np.ascontiguousarray(np.concatenate((payload, padding), axis=1))

    def _read_positions(self) -> tuple[int, int, int, int]:
        capacity, channels, write_pos, buffered_frames, total_written_frames, dropped_frames = struct.unpack_from(
            _HEADER_FORMAT,
            self.memory.buf,
            0,
        )
        if capacity != self.capacity_frames or channels != self.channels:
            raise ValueError("shared memory ring buffer header does not match expected shape")
        return int(write_pos), int(buffered_frames), int(total_written_frames), int(dropped_frames)

    def _write_header(
        self,
        *,
        write_pos: int,
        buffered_frames: int,
        total_written_frames: int,
        dropped_frames: int,
    ) -> None:
        struct.pack_into(
            _HEADER_FORMAT,
            self.memory.buf,
            0,
            self.capacity_frames,
            self.channels,
            int(write_pos),
            int(buffered_frames),
            int(total_written_frames),
            int(dropped_frames),
        )


__all__ = ["MockPcmRingBuffer", "RingBufferStats", "SharedMemoryPcmRingBuffer"]
