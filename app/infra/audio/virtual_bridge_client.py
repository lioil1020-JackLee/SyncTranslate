from __future__ import annotations

from multiprocessing.connection import Client, Connection
from pathlib import Path
import subprocess
import sys
import time
from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Callable

import numpy as np

from app.infra.audio.bridge_protocol import (
    VIRTUAL_AUDIO_V2_CHANNELS,
    decode_audio_packet,
    decode_pcm16_stereo_packet,
    encode_pcm16_stereo_packet,
)
from app.infra.audio.bridge_ring_buffer import SharedMemoryPcmRingBuffer
from app.infra.audio.synctranslate_driver_client import (
    SyncTranslateDriverAudioClient,
    SyncTranslateDriverUnavailable,
)
from app.infra.audio.windows_named_event import WindowsNamedEvent
from app.infra.subprocess_utils import hidden_subprocess_kwargs, safe_subprocess_env


RemoteInputConsumer = Callable[[np.ndarray, float], None]


class VirtualBridgeUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class VirtualBridgeStats:
    connected: bool
    remote_input_running: bool
    remote_input_frames: int
    virtual_microphone_frames: int
    remote_input_buffered_frames: int = 0
    remote_input_buffer_capacity_frames: int = 0
    virtual_microphone_buffered_frames: int = 0
    virtual_microphone_dropped_frames: int = 0
    virtual_microphone_buffer_capacity_frames: int = 0
    virtual_microphone_shared_memory_name: str = ""
    virtual_microphone_event_name: str = ""
    last_error: str = ""


@dataclass(frozen=True, slots=True)
class VirtualBridgePcmVerificationResult:
    ok: bool
    frames_written: int
    stats_delta_frames: int
    shared_memory_frames: int
    event_signaled: bool
    latency_ms: float
    error: str = ""


@dataclass(frozen=True, slots=True)
class VirtualBridgeHeartbeatResult:
    ok: bool
    roundtrip_ms: float
    error: str = ""


class VirtualAudioBridgeClient:
    def __init__(self, *, bridge_path: str, pipe_name: str = r"\\.\pipe\SyncTranslateAudioBridge") -> None:
        self.bridge_path = bridge_path
        self.pipe_name = pipe_name
        self._lock = Lock()
        self._consumer_lock = Lock()
        self._process: subprocess.Popen[str] | None = None
        self._connection: Connection | None = None
        self._last_error = ""
        self._consumers: list[RemoteInputConsumer] = []
        self._remote_input_sample_rate = 48000
        self._remote_input_stop = Event()
        self._remote_input_thread: Thread | None = None
        self._embedded_handler = None
        self._driver_client = SyncTranslateDriverAudioClient()

    def start_remote_input(self, *, sample_rate: int, device_name: str = "", chunk_ms: int = 10) -> None:
        import logging
        logger = logging.getLogger(__name__)
        
        if not device_name:
            logger.error("start_remote_input called with empty device_name")
            raise ValueError("device_name cannot be empty")
        
        logger.info(
            f"VirtualAudioBridgeClient.start_remote_input: "
            f"device_name={device_name!r}, sample_rate={sample_rate}, chunk_ms={chunk_ms}"
        )
        self._remote_input_sample_rate = int(sample_rate)
        self._request(
            {
                "cmd": "start_remote_input",
                "sample_rate": int(sample_rate),
                "device_name": str(device_name or ""),
                "chunk_ms": int(chunk_ms),
            }
        )
        self._ensure_remote_input_thread()

    def stop_remote_input(self) -> None:
        self._stop_remote_input_thread()
        if not self._has_live_transport():
            return
        try:
            self._request({"cmd": "stop_remote_input"})
        except VirtualBridgeUnavailable:
            return

    def add_remote_input_consumer(self, consumer: RemoteInputConsumer) -> None:
        with self._consumer_lock:
            if consumer not in self._consumers:
                self._consumers.append(consumer)
        self._ensure_remote_input_thread()

    def remove_remote_input_consumer(self, consumer: RemoteInputConsumer) -> None:
        with self._consumer_lock:
            if consumer in self._consumers:
                self._consumers.remove(consumer)
            has_consumers = bool(self._consumers)
        if not has_consumers:
            self._stop_remote_input_thread()

    def write_virtual_microphone(self, audio: np.ndarray, *, sample_rate: int) -> None:
        if self._try_write_driver_virtual_microphone(audio, sample_rate=int(sample_rate)):
            return
        self._request(
            {
                "cmd": "write_virtual_microphone",
                "packet": encode_pcm16_stereo_packet(audio, sample_rate=int(sample_rate)),
            }
        )

    def flush_virtual_microphone(self) -> None:
        try:
            if self._driver_client.is_available():
                self._driver_client.flush_virtual_microphone()
        except SyncTranslateDriverUnavailable:
            pass
        if not self._has_live_transport():
            return
        try:
            self._request({"cmd": "flush_virtual_microphone"})
        except VirtualBridgeUnavailable:
            return

    def debug_inject_remote_input(self, audio: np.ndarray, *, sample_rate: int) -> None:
        self._request(
            {
                "cmd": "inject_remote_input",
                "packet": encode_pcm16_stereo_packet(audio, sample_rate=int(sample_rate)),
            }
        )

    def stats(self) -> VirtualBridgeStats:
        try:
            payload = self._request({"cmd": "get_stats"})
            return VirtualBridgeStats(
                connected=bool(payload.get("connected", True)),
                remote_input_running=bool(payload.get("remote_input_running", False)),
                remote_input_frames=int(payload.get("remote_input_frames", 0) or 0),
                remote_input_buffered_frames=int(payload.get("remote_input_buffered_frames", 0) or 0),
                remote_input_buffer_capacity_frames=int(payload.get("remote_input_buffer_capacity_frames", 0) or 0),
                virtual_microphone_frames=int(payload.get("virtual_microphone_frames", 0) or 0),
                virtual_microphone_buffered_frames=int(payload.get("virtual_microphone_buffered_frames", 0) or 0),
                virtual_microphone_dropped_frames=int(payload.get("virtual_microphone_dropped_frames", 0) or 0),
                virtual_microphone_buffer_capacity_frames=int(
                    payload.get("virtual_microphone_buffer_capacity_frames", 0) or 0
                ),
                virtual_microphone_shared_memory_name=str(payload.get("virtual_microphone_shared_memory_name", "") or ""),
                virtual_microphone_event_name=str(payload.get("virtual_microphone_event_name", "") or ""),
                last_error=str(payload.get("last_error", "") or ""),
            )
        except VirtualBridgeUnavailable:
            return VirtualBridgeStats(
                connected=False,
                remote_input_running=False,
                remote_input_frames=0,
                remote_input_buffered_frames=0,
                remote_input_buffer_capacity_frames=0,
                virtual_microphone_frames=0,
                virtual_microphone_buffered_frames=0,
                virtual_microphone_dropped_frames=0,
                virtual_microphone_buffer_capacity_frames=0,
                virtual_microphone_shared_memory_name="",
                virtual_microphone_event_name="",
                last_error=self._last_error or "bridge_unavailable",
            )

    def verify_pcm_loopback(
        self,
        *,
        sample_rate: int = 48000,
        duration_ms: int = 20,
        tone_hz: float = 440.0,
        timeout_ms: int = 200,
    ) -> VirtualBridgePcmVerificationResult:
        started = time.perf_counter()
        try:
            before = self.stats()
            if not before.connected:
                return VirtualBridgePcmVerificationResult(
                    ok=False,
                    frames_written=0,
                    stats_delta_frames=0,
                    shared_memory_frames=0,
                    event_signaled=False,
                    latency_ms=0.0,
                    error="bridge_unavailable",
                )

            frames = max(1, int(int(sample_rate) * max(10, int(duration_ms)) / 1000))
            t = np.arange(frames, dtype=np.float32) / float(sample_rate)
            tone = (np.sin(2.0 * np.pi * float(tone_hz) * t) * 0.2).reshape((-1, 1))
            response = self._request(
                {
                    "cmd": "write_virtual_microphone",
                    "packet": encode_pcm16_stereo_packet(tone, sample_rate=int(sample_rate)),
                }
            )
            written_frames = int(response.get("written_frames", 0) or 0)

            event_signaled = False
            if before.virtual_microphone_event_name:
                event = WindowsNamedEvent(before.virtual_microphone_event_name)
                try:
                    event_signaled = bool(event.wait(max(1, int(timeout_ms))))
                finally:
                    event.close()

            # 給 bridge worker 一個短暫窗口更新統計與 shared memory header。
            time.sleep(0.01)
            after = self.stats()
            stats_delta = max(0, int(after.virtual_microphone_frames) - int(before.virtual_microphone_frames))

            shared_frames = 0
            shm_name = str(after.virtual_microphone_shared_memory_name or "")
            capacity = int(after.virtual_microphone_buffer_capacity_frames or 0)
            if shm_name and capacity > 0:
                shared_buffer = SharedMemoryPcmRingBuffer.attach(
                    name=shm_name,
                    capacity_frames=capacity,
                    channels=VIRTUAL_AUDIO_V2_CHANNELS,
                )
                try:
                    shared_frames = int(shared_buffer.stats().buffered_frames)
                finally:
                    shared_buffer.close()

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            frames_written = written_frames if written_frames > 0 else frames
            bridge_accepted_audio = frames_written >= frames or stats_delta >= frames
            bridge_observed_audio = stats_delta >= frames or shared_frames > 0 or event_signaled
            ok = bridge_accepted_audio and bridge_observed_audio
            return VirtualBridgePcmVerificationResult(
                ok=bool(ok),
                frames_written=frames_written,
                stats_delta_frames=stats_delta,
                shared_memory_frames=shared_frames,
                event_signaled=event_signaled,
                latency_ms=float(elapsed_ms),
                error="" if ok else "pcm_loopback_check_failed",
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return VirtualBridgePcmVerificationResult(
                ok=False,
                frames_written=0,
                stats_delta_frames=0,
                shared_memory_frames=0,
                event_signaled=False,
                latency_ms=float(elapsed_ms),
                error=str(exc),
            )

    def heartbeat(self, *, timeout_ms: int = 200) -> VirtualBridgeHeartbeatResult:
        started = time.perf_counter()
        try:
            payload = self._request(
                {
                    "cmd": "ping",
                    "client_sent_at_ms": int(time.time() * 1000),
                    "timeout_ms": int(timeout_ms),
                }
            )
            if not bool(payload.get("pong", False)):
                raise VirtualBridgeUnavailable("bridge_ping_failed")
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return VirtualBridgeHeartbeatResult(ok=True, roundtrip_ms=float(elapsed_ms), error="")
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return VirtualBridgeHeartbeatResult(ok=False, roundtrip_ms=float(elapsed_ms), error=str(exc))

    def dump_sine_wav(
        self,
        path: str,
        *,
        sample_rate: int = 48000,
        duration_ms: int = 500,
        frequency: float = 440.0,
    ) -> str:
        payload = self._request(
            {
                "cmd": "dump_sine_wav",
                "path": path,
                "sample_rate": int(sample_rate),
                "duration_ms": int(duration_ms),
                "frequency": float(frequency),
            }
        )
        return str(payload.get("path") or path)

    def close(self) -> None:
        self._stop_remote_input_thread()
        with self._lock:
            process = self._process
            connection = self._connection
            embedded_handler = self._embedded_handler
        if process is not None or embedded_handler is not None:
            try:
                self._request({"cmd": "shutdown"})
            except Exception:
                pass
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
        if process is not None:
            try:
                process.wait(timeout=2.0)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
        with self._lock:
            self._process = None
            self._connection = None
            self._embedded_handler = None
        if embedded_handler is not None:
            try:
                embedded_handler.close()
            except Exception:
                pass
        self._driver_client.close()

    def _try_write_driver_virtual_microphone(self, audio: np.ndarray, *, sample_rate: int) -> bool:
        try:
            if not self._driver_client.is_available():
                return False
            self._driver_client.write_virtual_microphone(audio, sample_rate=int(sample_rate))
            self._last_error = ""
            return True
        except SyncTranslateDriverUnavailable as exc:
            self._last_error = str(exc)
            return False

    def _ensure_remote_input_thread(self) -> None:
        with self._consumer_lock:
            has_consumers = bool(self._consumers)
        if not has_consumers:
            return
        thread = self._remote_input_thread
        if thread is not None and thread.is_alive():
            return
        self._remote_input_stop.clear()
        thread = Thread(target=self._poll_remote_input, daemon=True)
        self._remote_input_thread = thread
        thread.start()

    def _stop_remote_input_thread(self) -> None:
        self._remote_input_stop.set()
        thread = self._remote_input_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._remote_input_thread = None

    def _poll_remote_input(self) -> None:
        while not self._remote_input_stop.wait(0.02):
            with self._consumer_lock:
                consumers = list(self._consumers)
            if not consumers:
                return
            try:
                payload = self._request(
                    {
                        "cmd": "read_remote_input",
                        "sample_rate": int(self._remote_input_sample_rate),
                    }
                )
                packet = payload.get("packet")
                if not isinstance(packet, dict):
                    continue
                audio, sample_rate = _decode_bridge_audio_packet(packet)
                if audio.size == 0:
                    continue
                sample_rate_float = float(sample_rate or self._remote_input_sample_rate or 48000)
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(
                    f"[CLIENT] _poll_remote_input: "
                    f"packet_sample_rate={sample_rate}, "
                    f"stored_sample_rate={self._remote_input_sample_rate}, "
                    f"final_rate={sample_rate_float}, "
                    f"audio_frames={audio.shape[0] if audio.ndim > 0 else 0}, "
                    f"consumers={len(consumers)}"
                )
                for consumer in consumers:
                    consumer(audio.astype(np.float32, copy=False), sample_rate_float)
            except Exception as exc:
                self._last_error = str(exc)
                time.sleep(0.1)

    def _request(self, request: dict[str, object]) -> dict[str, object]:
            with self._lock:
                if self._embedded_handler is not None:
                    return self._request_embedded_locked(request)
            try:
                connection = self._ensure_connection_locked()
            except VirtualBridgeUnavailable:
                self._ensure_embedded_handler_locked()
                return self._request_embedded_locked(request)
            try:
                connection.send(request)
                cmd = str(request.get("cmd") or "")
                timeout_sec = self._response_timeout_for_cmd(cmd)
                if not connection.poll(timeout_sec):
                    raise TimeoutError(f"bridge response timeout cmd={cmd or 'unknown'}")
                response = connection.recv()
            except Exception as exc:
                self._connection = None
                self._last_error = str(exc)
                raise VirtualBridgeUnavailable(str(exc)) from exc
            if not isinstance(response, dict):
                self._last_error = "bridge response must be a JSON object"
                raise VirtualBridgeUnavailable(self._last_error)
            if not bool(response.get("ok", False)):
                self._last_error = str(response.get("error") or "bridge command failed")
                raise VirtualBridgeUnavailable(self._last_error)
            self._last_error = ""
            return response

    def _has_live_transport(self) -> bool:
        with self._lock:
            if self._embedded_handler is not None:
                return True
            if self._connection is not None:
                return True
            return self._process is not None and self._process.poll() is None

    def _ensure_embedded_handler_locked(self) -> None:
        if self._embedded_handler is not None:
            return
        from app.infra.audio.bridge_process import BridgeCommandHandler

        self._embedded_handler = BridgeCommandHandler()

    def _request_embedded_locked(self, request: dict[str, object]) -> dict[str, object]:
        if self._embedded_handler is None:
            raise VirtualBridgeUnavailable("embedded_handler_unavailable")
        response = self._embedded_handler.handle(request)
        if not isinstance(response, dict):
            self._last_error = "embedded bridge response must be a JSON object"
            raise VirtualBridgeUnavailable(self._last_error)
        if not bool(response.get("ok", False)):
            self._last_error = str(response.get("error") or "embedded bridge command failed")
            raise VirtualBridgeUnavailable(self._last_error)
        self._last_error = ""
        return response

    def _ensure_connection_locked(self) -> Connection:
        if self._connection is not None:
            return self._connection
        self._ensure_process_locked()
        # Packaged PyInstaller bridge workers can take a few seconds to cold-start.
        deadline = time.monotonic() + 8.0
        last_error = ""
        while time.monotonic() < deadline:
            try:
                self._connection = Client(self.pipe_name, family="AF_PIPE")
                self._request_hello_locked()
                return self._connection
            except Exception as exc:
                last_error = str(exc)
                time.sleep(0.05)
        self._last_error = last_error or "bridge named pipe unavailable"
        raise VirtualBridgeUnavailable(self._last_error)

    def _request_hello_locked(self) -> None:
        assert self._connection is not None
        self._connection.send({"cmd": "hello", "transport": "named_pipe"})
        if not self._connection.poll(0.5):
            raise VirtualBridgeUnavailable("bridge hello timeout")
        response = self._connection.recv()
        if not isinstance(response, dict) or not bool(response.get("ok", False)):
            raise VirtualBridgeUnavailable(str(response))

    @staticmethod
    def _response_timeout_for_cmd(cmd: str) -> float:
        command = str(cmd or "").strip().lower()
        if command == "get_stats":
            return 0.35
        if command == "read_remote_input":
            return 0.8
        if command == "ping":
            return 0.5
        return 1.0

    def _ensure_process_locked(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        command = self._bridge_command()
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                env=safe_subprocess_env(),
                **hidden_subprocess_kwargs(),
            )
        except Exception as exc:
            self._last_error = str(exc)
            raise VirtualBridgeUnavailable(str(exc)) from exc
        return self._process

    def _bridge_command(self) -> list[str]:
        configured = str(self.bridge_path or "").strip()
        if configured and Path(configured).exists():
            return [configured, "--pipe", self.pipe_name]
        return [sys.executable, "-m", "app.infra.audio.bridge_process", "--pipe", self.pipe_name]


class InMemoryVirtualAudioBridgeClient:
    def __init__(self) -> None:
        self._lock = Lock()
        self._consumers: list[RemoteInputConsumer] = []
        self._remote_input_running = False
        self._remote_input_sample_rate = 0
        self._remote_input_frames = 0
        self._virtual_microphone_frames = 0
        self.virtual_microphone_packets: list[tuple[np.ndarray, int]] = []

    def start_remote_input(self, *, sample_rate: int, device_name: str = "", chunk_ms: int = 10) -> None:
        del device_name, chunk_ms
        with self._lock:
            self._remote_input_running = True
            self._remote_input_sample_rate = int(sample_rate)

    def stop_remote_input(self) -> None:
        with self._lock:
            self._remote_input_running = False

    def add_remote_input_consumer(self, consumer: RemoteInputConsumer) -> None:
        with self._lock:
            if consumer not in self._consumers:
                self._consumers.append(consumer)

    def remove_remote_input_consumer(self, consumer: RemoteInputConsumer) -> None:
        with self._lock:
            if consumer in self._consumers:
                self._consumers.remove(consumer)

    def emit_remote_input(self, audio: np.ndarray, *, sample_rate: int | None = None) -> None:
        with self._lock:
            if not self._remote_input_running:
                return
            consumers = list(self._consumers)
            rate = float(sample_rate or self._remote_input_sample_rate or 48000)
        packet = encode_pcm16_stereo_packet(audio, sample_rate=int(rate))
        payload, packet_sample_rate = decode_pcm16_stereo_packet(packet)
        with self._lock:
            self._remote_input_frames += int(payload.shape[0]) if payload.ndim else int(payload.size)
        for consumer in consumers:
            consumer(payload, float(packet_sample_rate))

    def write_virtual_microphone(self, audio: np.ndarray, *, sample_rate: int) -> None:
        packet = encode_pcm16_stereo_packet(audio, sample_rate=int(sample_rate))
        payload, packet_sample_rate = decode_pcm16_stereo_packet(packet)
        with self._lock:
            self.virtual_microphone_packets.append((payload, int(packet_sample_rate)))
            self._virtual_microphone_frames += int(payload.shape[0]) if payload.ndim else int(payload.size)

    def flush_virtual_microphone(self) -> None:
        with self._lock:
            self.virtual_microphone_packets.clear()

    def stats(self) -> VirtualBridgeStats:
        with self._lock:
            return VirtualBridgeStats(
                connected=True,
                remote_input_running=self._remote_input_running,
                remote_input_frames=self._remote_input_frames,
                virtual_microphone_frames=self._virtual_microphone_frames,
                virtual_microphone_buffered_frames=self._virtual_microphone_frames,
                virtual_microphone_dropped_frames=0,
                virtual_microphone_buffer_capacity_frames=self._virtual_microphone_frames,
                virtual_microphone_shared_memory_name="",
                virtual_microphone_event_name="",
            )

    def verify_pcm_loopback(
        self,
        *,
        sample_rate: int = 48000,
        duration_ms: int = 20,
        tone_hz: float = 440.0,
        timeout_ms: int = 200,
    ) -> VirtualBridgePcmVerificationResult:
        del tone_hz, timeout_ms
        frames = max(1, int(int(sample_rate) * max(10, int(duration_ms)) / 1000))
        t = np.arange(frames, dtype=np.float32) / float(sample_rate)
        tone = (np.sin(2.0 * np.pi * 440.0 * t) * 0.2).reshape((-1, 1))
        before = self.stats()
        started = time.perf_counter()
        self.write_virtual_microphone(tone, sample_rate=int(sample_rate))
        after = self.stats()
        stats_delta = max(0, int(after.virtual_microphone_frames) - int(before.virtual_microphone_frames))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        ok = stats_delta >= frames
        return VirtualBridgePcmVerificationResult(
            ok=bool(ok),
            frames_written=frames,
            stats_delta_frames=stats_delta,
            shared_memory_frames=int(after.virtual_microphone_buffered_frames),
            event_signaled=True,
            latency_ms=float(elapsed_ms),
            error="" if ok else "pcm_loopback_check_failed",
        )

    def heartbeat(self, *, timeout_ms: int = 200) -> VirtualBridgeHeartbeatResult:
        del timeout_ms
        return VirtualBridgeHeartbeatResult(ok=True, roundtrip_ms=0.0, error="")


def _decode_bridge_audio_packet(packet: dict[str, object]) -> tuple[np.ndarray, int]:
    if int(packet.get("protocol_version", 1) or 1) == 2:
        return decode_pcm16_stereo_packet(packet)  # type: ignore[arg-type]
    return decode_audio_packet(packet)  # type: ignore[arg-type]


__all__ = [
    "InMemoryVirtualAudioBridgeClient",
    "RemoteInputConsumer",
    "VirtualBridgeHeartbeatResult",
    "VirtualBridgePcmVerificationResult",
    "VirtualAudioBridgeClient",
    "VirtualBridgeStats",
    "VirtualBridgeUnavailable",
]
