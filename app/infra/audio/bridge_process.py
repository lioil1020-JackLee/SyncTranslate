from __future__ import annotations

import argparse
import ctypes
import json
from multiprocessing.connection import Listener
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Thread
from typing import Any

import numpy as np

_numpy_fromstring = np.fromstring


def _fromstring_compat(data, dtype=float, count=-1, *, sep="", like=None):
    if sep == "" and not isinstance(data, str):
        try:
            return np.frombuffer(data, dtype=dtype, count=count)
        except TypeError:
            pass
    if like is not None:
        return _numpy_fromstring(data, dtype=dtype, count=count, sep=sep, like=like)
    return _numpy_fromstring(data, dtype=dtype, count=count, sep=sep)


np.fromstring = _fromstring_compat

try:
    import soundcard as sc
except Exception:
    sc = None

from app.infra.audio.bridge_ring_buffer import SharedMemoryPcmRingBuffer
from app.infra.audio.bridge_protocol import (
    BRIDGE_PROTOCOL_VERSION,
    audio_frame_count,
    audio_peak,
    decode_audio_packet,
    encode_audio_packet,
)
from app.infra.audio.windows_named_event import WindowsNamedEvent


@dataclass(slots=True)
class BridgeState:
    started_at: float = field(default_factory=time.time)
    configured: dict[str, Any] = field(default_factory=dict)
    remote_input_running: bool = False
    remote_input_sample_rate: int = 0
    remote_input_frames: int = 0
    virtual_microphone_frames: int = 0
    virtual_microphone_packets: int = 0
    last_virtual_microphone_peak: float = 0.0
    last_error: str = ""


class BridgeCommandHandler:
    REMOTE_INPUT_SHARED_MEMORY_NAME = r"Local\SyncTranslateRemoteInput"
    VIRTUAL_MIC_SHARED_MEMORY_NAME = r"Local\SyncTranslateVirtualMicrophoneBuffer"

    def __init__(self) -> None:
        self.state = BridgeState()
        self.remote_input_event = WindowsNamedEvent(r"Local\SyncTranslateRemoteInputReady")
        self.remote_input_buffer = self._create_or_attach_ring_buffer(
            name=self.REMOTE_INPUT_SHARED_MEMORY_NAME,
            capacity_frames=48000 * 10,
            channels=2,
        )
        self.remote_input_stop = Event()
        self.remote_input_thread: Thread | None = None
        self.virtual_microphone_buffer = self._create_or_attach_ring_buffer(
            name=self.VIRTUAL_MIC_SHARED_MEMORY_NAME,
            capacity_frames=48000 * 10,
            channels=1,
        )
        self.virtual_microphone_event = WindowsNamedEvent(r"Local\SyncTranslateVirtualMicrophoneReady")

    @staticmethod
    def _create_or_attach_ring_buffer(*, name: str, capacity_frames: int, channels: int) -> SharedMemoryPcmRingBuffer:
        try:
            return SharedMemoryPcmRingBuffer.create(
                name=name,
                capacity_frames=int(capacity_frames),
                channels=int(channels),
            )
        except FileExistsError:
            buffer = SharedMemoryPcmRingBuffer.attach(
                name=name,
                capacity_frames=int(capacity_frames),
                channels=int(channels),
            )
            buffer.clear()
            return buffer

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        cmd = str(request.get("cmd") or "").strip()
        try:
            if cmd == "hello":
                return self._ok(
                    protocol_version=BRIDGE_PROTOCOL_VERSION,
                    worker="python_bridge",
                    transport=str(request.get("transport") or ""),
                    virtual_microphone_shared_memory_name=self.virtual_microphone_buffer.memory.name,
                    virtual_microphone_event_name=self.virtual_microphone_event.name,
                    uptime_ms=self._uptime_ms(),
                )
            if cmd == "start":
                self.state.configured.update(request)
                return self._ok(started=True)
            if cmd == "configure":
                self.state.configured.update(request)
                return self._ok(configured=True)
            if cmd == "start_remote_input":
                self.state.remote_input_running = True
                self.state.remote_input_sample_rate = int(request.get("sample_rate", 48000) or 48000)
                self._start_remote_input_capture(
                    device_name=str(request.get("device_name") or ""),
                    sample_rate=self.state.remote_input_sample_rate,
                    chunk_ms=int(request.get("chunk_ms", 10) or 10),
                )
                return self._ok(remote_input_running=True)
            if cmd == "stop_remote_input":
                self.state.remote_input_running = False
                self._stop_remote_input_capture()
                self.remote_input_buffer.clear()
                self.remote_input_event.reset()
                return self._ok(remote_input_running=False)
            if cmd == "inject_remote_input":
                packet = request.get("packet")
                if not isinstance(packet, dict):
                    raise ValueError("packet is required")
                audio, sample_rate = decode_audio_packet(packet)
                written_frames = self.remote_input_buffer.write(audio)
                self.state.remote_input_sample_rate = int(sample_rate or self.state.remote_input_sample_rate or 48000)
                self.state.remote_input_frames += written_frames
                return self._ok(written_frames=written_frames)
            if cmd == "read_remote_input":
                # Driver path uses named event to signal fresh PCM; keep polling-compatible behavior.
                if self.state.remote_input_running:
                    self.remote_input_event.wait(1)
                audio = self.remote_input_buffer.snapshot()
                self.remote_input_buffer.clear()
                self.remote_input_event.reset()
                sample_rate = int(self.state.remote_input_sample_rate or request.get("sample_rate", 48000) or 48000)
                return self._ok(packet=encode_audio_packet(audio, sample_rate=sample_rate))
            if cmd == "write_virtual_microphone":
                packet = request.get("packet")
                if not isinstance(packet, dict):
                    raise ValueError("packet is required")
                audio, _sample_rate = decode_audio_packet(packet)
                written_frames = self.virtual_microphone_buffer.write(audio)
                self.virtual_microphone_event.set()
                self.state.virtual_microphone_frames += written_frames
                self.state.virtual_microphone_packets += 1
                self.state.last_virtual_microphone_peak = audio_peak(audio)
                return self._ok(written_frames=written_frames)
            if cmd == "flush_virtual_microphone":
                self.virtual_microphone_buffer.clear()
                self.virtual_microphone_event.reset()
                self.state.virtual_microphone_packets = 0
                self.state.last_virtual_microphone_peak = 0.0
                return self._ok(flushed=True)
            if cmd == "dump_sine_wav":
                path = self._dump_sine_wav(request)
                return self._ok(path=str(path))
            if cmd == "get_stats":
                return self._ok(**self._stats())
            if cmd == "ping":
                client_sent_at_ms = int(request.get("client_sent_at_ms", 0) or 0)
                return self._ok(pong=True, uptime_ms=self._uptime_ms(), client_sent_at_ms=client_sent_at_ms)
            if cmd == "shutdown":
                return self._ok(shutdown=True)
            raise ValueError(f"unknown bridge command: {cmd}")
        except Exception as exc:
            self.state.last_error = str(exc)
            return {"ok": False, "error": str(exc)}

    def _stats(self) -> dict[str, Any]:
        remote_input_buffer = self.remote_input_buffer.stats()
        virtual_microphone_buffer = self.virtual_microphone_buffer.stats()
        return {
            "connected": True,
            "protocol_version": BRIDGE_PROTOCOL_VERSION,
            "worker": "python_bridge",
            "uptime_ms": self._uptime_ms(),
            "remote_input_running": self.state.remote_input_running,
            "remote_input_sample_rate": self.state.remote_input_sample_rate,
            "remote_input_frames": int(remote_input_buffer.total_written_frames),
            "remote_input_buffered_frames": int(remote_input_buffer.buffered_frames),
            "remote_input_buffer_capacity_frames": int(remote_input_buffer.capacity_frames),
            "virtual_microphone_frames": self.state.virtual_microphone_frames,
            "virtual_microphone_packets": self.state.virtual_microphone_packets,
            "virtual_microphone_buffered_frames": virtual_microphone_buffer.buffered_frames,
            "virtual_microphone_dropped_frames": virtual_microphone_buffer.dropped_frames,
            "virtual_microphone_buffer_capacity_frames": virtual_microphone_buffer.capacity_frames,
            "virtual_microphone_shared_memory_name": virtual_microphone_buffer.shared_memory_name,
            "virtual_microphone_event_name": self.virtual_microphone_event.name,
            "last_virtual_microphone_peak": self.state.last_virtual_microphone_peak,
            "last_error": self.state.last_error,
        }

    def _start_remote_input_capture(self, *, device_name: str, sample_rate: int, chunk_ms: int) -> None:
        self._stop_remote_input_capture()
        self.remote_input_stop.clear()
        thread = Thread(
            target=self._remote_input_capture_worker,
            kwargs={
                "device_name": str(device_name or ""),
                "sample_rate": int(sample_rate or 48000),
                "chunk_ms": int(chunk_ms or 10),
            },
            daemon=True,
        )
        self.remote_input_thread = thread
        thread.start()

    def _stop_remote_input_capture(self) -> None:
        self.remote_input_stop.set()
        thread = self.remote_input_thread
        if thread and thread.is_alive():
            thread.join(timeout=1.5)
        self.remote_input_thread = None

    def _remote_input_capture_worker(self, *, device_name: str, sample_rate: int, chunk_ms: int) -> None:
        if sc is None:
            self.state.last_error = "soundcard_unavailable"
            return
        com_initialized = self._coinitialize_for_thread()
        try:
            microphone = self._resolve_loopback_microphone(device_name)
            block_frames = max(128, int(int(sample_rate) * max(5, int(chunk_ms)) / 1000))
            with microphone.recorder(samplerate=int(sample_rate), channels=2, blocksize=block_frames) as recorder:
                while not self.remote_input_stop.is_set():
                    audio = recorder.record(numframes=block_frames)
                    if audio is None:
                        continue
                    payload = np.asarray(audio, dtype=np.float32)
                    if payload.size == 0:
                        continue
                    if payload.ndim == 1:
                        payload = payload.reshape((-1, 1))
                    if payload.shape[1] == 1:
                        payload = np.repeat(payload, 2, axis=1)
                    elif payload.shape[1] > 2:
                        payload = payload[:, :2]
                    written_frames = self.remote_input_buffer.write(payload)
                    self.state.remote_input_frames += int(written_frames)
                    self.remote_input_event.set()
                    self.state.last_error = ""
        except Exception as exc:
            if not self.remote_input_stop.is_set():
                self.state.last_error = str(exc)
        finally:
            if com_initialized:
                ctypes.OleDLL("ole32").CoUninitialize()

    @staticmethod
    def _coinitialize_for_thread() -> bool:
        rpc_e_changed_mode = 0x80010106
        ole32 = ctypes.OleDLL("ole32")
        hr = ole32.CoInitializeEx(None, 0)
        if hr in (0, 1):
            return True
        if hr == rpc_e_changed_mode:
            return False
        raise OSError(hr, "CoInitializeEx failed")

    @staticmethod
    def _resolve_loopback_microphone(device_name: str):
        if sc is None:
            raise RuntimeError("soundcard_unavailable")

        requested = str(device_name or "").strip()
        errors: list[str] = []
        if requested:
            try:
                return sc.get_microphone(requested, include_loopback=True)
            except Exception as exc:
                errors.append(str(exc))

        microphones = list(sc.all_microphones(include_loopback=True))
        if requested:
            normalized = requested.lower()
            for microphone in microphones:
                name = str(getattr(microphone, "name", "") or "")
                if normalized == name.lower() or normalized in name.lower() or name.lower() in normalized:
                    return microphone

        sync_loopbacks = [
            microphone
            for microphone in microphones
            if "synctranslate" in str(getattr(microphone, "name", "") or "").lower()
            and "microphone" not in str(getattr(microphone, "name", "") or "").lower()
        ]
        if sync_loopbacks:
            return sync_loopbacks[0]

        if microphones:
            return microphones[0]

        detail = f": {' | '.join(errors[:3])}" if errors else ""
        raise RuntimeError(f"loopback_microphone_not_found{detail}")

    def _dump_sine_wav(self, request: dict[str, Any]) -> Path:
        path = Path(str(request.get("path") or "bridge_sine.wav"))
        sample_rate = int(request.get("sample_rate", 48000) or 48000)
        duration_ms = max(10, int(request.get("duration_ms", 500) or 500))
        frequency = float(request.get("frequency", 440.0) or 440.0)
        frames = max(1, int(sample_rate * duration_ms / 1000))
        t = np.arange(frames, dtype=np.float32) / float(sample_rate)
        audio = np.sin(2.0 * np.pi * frequency * t) * 0.2
        pcm = np.clip(audio, -1.0, 1.0)
        pcm16 = (pcm * 32767.0).astype("<i2")
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as fp:
            fp.setnchannels(1)
            fp.setsampwidth(2)
            fp.setframerate(sample_rate)
            fp.writeframes(pcm16.tobytes())
        return path

    def _uptime_ms(self) -> int:
        return int((time.time() - self.state.started_at) * 1000)

    @staticmethod
    def _ok(**payload: Any) -> dict[str, Any]:
        return {"ok": True, **payload}

    def close(self) -> None:
        self._stop_remote_input_capture()
        self.remote_input_event.close()
        close_buffer = getattr(self.remote_input_buffer, "close", None)
        if callable(close_buffer):
            close_buffer()
        self.virtual_microphone_event.close()
        self.virtual_microphone_buffer.close_and_unlink()


def run_stdio(handler: BridgeCommandHandler) -> int:
    for line in sys.stdin:
        text = line.strip()
        if not text:
            continue
        request: dict[str, Any] = {}
        try:
            request = json.loads(text)
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            response = handler.handle(request)
        except Exception as exc:
            response = {"ok": False, "error": str(exc)}
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()
        if isinstance(request, dict) and request.get("cmd") == "shutdown":
            break
    return 0


def run_pipe(handler: BridgeCommandHandler, pipe_name: str) -> int:
    listener = Listener(pipe_name, family="AF_PIPE")
    try:
        conn = listener.accept()
        try:
            while True:
                request = conn.recv()
                if not isinstance(request, dict):
                    response = {"ok": False, "error": "request must be a dict"}
                else:
                    response = handler.handle(request)
                conn.send(response)
                if isinstance(request, dict) and request.get("cmd") == "shutdown":
                    break
        finally:
            conn.close()
    finally:
        listener.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="SyncTranslate audio bridge")
    parser.add_argument("--stdio", action="store_true", help="Use JSON line stdin/stdout transport")
    parser.add_argument("--pipe", default="", help="Use Windows Named Pipe transport at the given pipe name")
    args = parser.parse_args()

    handler = BridgeCommandHandler()
    try:
        if args.pipe:
            return run_pipe(handler, str(args.pipe))
        return run_stdio(handler)
    finally:
        handler.close()


if __name__ == "__main__":
    raise SystemExit(main())
