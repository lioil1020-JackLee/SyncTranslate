from __future__ import annotations

import time

import numpy as np

from app.infra.audio.sources import VirtualSpeakerSource
from app.infra.audio.sinks import VirtualMicrophoneSink
from app.infra.audio.virtual_bridge_client import InMemoryVirtualAudioBridgeClient, VirtualAudioBridgeClient


def test_bridge_client_starts_bridge_worker_and_uses_named_pipe_transport() -> None:
    client = VirtualAudioBridgeClient(bridge_path="")
    try:
        client.start_remote_input(sample_rate=48000)
        client.write_virtual_microphone(np.zeros((10, 1), dtype=np.float32), sample_rate=48000)
        stats = client.stats()
    finally:
        client.close()

    assert stats.connected is True
    assert stats.remote_input_running is True
    assert stats.virtual_microphone_frames == 10
    assert stats.virtual_microphone_buffered_frames == 10
    assert stats.virtual_microphone_dropped_frames == 0
    assert stats.virtual_microphone_buffer_capacity_frames > 0
    assert stats.virtual_microphone_shared_memory_name
    assert stats.virtual_microphone_event_name == r"Local\SyncTranslateVirtualMicrophoneReady"


def test_bridge_client_can_dump_sine_wav(tmp_path) -> None:
    client = VirtualAudioBridgeClient(bridge_path="")
    wav_path = tmp_path / "sine.wav"
    try:
        produced = client.dump_sine_wav(str(wav_path), duration_ms=50)
    finally:
        client.close()

    assert produced == str(wav_path)
    assert wav_path.exists()
    assert wav_path.stat().st_size > 44


def test_bridge_client_flushes_virtual_microphone_buffer() -> None:
    client = VirtualAudioBridgeClient(bridge_path="")
    try:
        client.write_virtual_microphone(np.ones((10, 1), dtype=np.float32), sample_rate=48000)
        before = client.stats()
        client.flush_virtual_microphone()
        after = client.stats()
    finally:
        client.close()

    assert before.virtual_microphone_buffered_frames == 10
    assert after.virtual_microphone_buffered_frames == 0
    assert after.virtual_microphone_frames == 10


def test_bridge_client_dispatches_remote_input_from_pipe_worker() -> None:
    client = VirtualAudioBridgeClient(bridge_path="")
    received: list[tuple[np.ndarray, float]] = []
    try:
        client.add_remote_input_consumer(lambda audio, sample_rate: received.append((audio, sample_rate)))
        client.start_remote_input(sample_rate=48000)
        client.debug_inject_remote_input(np.ones((8, 2), dtype=np.float32), sample_rate=48000)
        deadline = time.monotonic() + 2.0
        while not received and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        client.close()

    assert received
    assert received[0][0].shape == (8, 2)
    assert received[0][1] == 48000.0


def test_bridge_client_stop_and_flush_are_noops_before_transport_starts() -> None:
    client = VirtualAudioBridgeClient(bridge_path="")

    client.stop_remote_input()
    client.flush_virtual_microphone()

    assert client._process is None


def test_bridge_client_can_verify_pcm_loopback() -> None:
    client = VirtualAudioBridgeClient(bridge_path="")
    try:
        result = client.verify_pcm_loopback(duration_ms=20, timeout_ms=200)
    finally:
        client.close()

    assert result.ok is True
    assert result.frames_written > 0
    assert result.stats_delta_frames >= result.frames_written
    assert result.shared_memory_frames > 0


def test_bridge_client_heartbeat_is_ok() -> None:
    client = VirtualAudioBridgeClient(bridge_path="")
    try:
        heartbeat = client.heartbeat(timeout_ms=200)
    finally:
        client.close()

    assert heartbeat.ok is True
    assert heartbeat.roundtrip_ms >= 0.0


def test_bridge_client_heartbeat_fails_when_ping_is_unsupported(monkeypatch) -> None:
    client = VirtualAudioBridgeClient(bridge_path="")

    def _fake_request(request: dict[str, object]) -> dict[str, object]:
        cmd = str(request.get("cmd") or "")
        if cmd == "ping":
            raise RuntimeError("unknown bridge command: ping")
        raise AssertionError(f"unexpected cmd: {cmd}")

    monkeypatch.setattr(client, "_request", _fake_request)
    heartbeat = client.heartbeat(timeout_ms=200)

    assert heartbeat.ok is False
    assert "unknown bridge command: ping" in heartbeat.error


def test_in_memory_bridge_delivers_virtual_speaker_audio_to_source_consumer() -> None:
    bridge = InMemoryVirtualAudioBridgeClient()
    source = VirtualSpeakerSource(bridge)
    received: list[tuple[np.ndarray, float]] = []

    source.add_consumer(lambda audio, sample_rate: received.append((audio, sample_rate)))
    source.start("SyncTranslate Virtual Speaker", sample_rate=48000, chunk_ms=10)
    bridge.emit_remote_input(np.ones((4, 1), dtype=np.float32), sample_rate=48000)

    assert len(received) == 1
    assert received[0][0].shape == (4, 1)
    assert received[0][1] == 48000.0
    assert source.stats().frame_count == 4


def test_virtual_microphone_sink_writes_to_bridge() -> None:
    bridge = InMemoryVirtualAudioBridgeClient()
    sink = VirtualMicrophoneSink(bridge)

    sink.play(np.full((6, 1), 0.25, dtype=np.float32), sample_rate=48000)

    assert len(bridge.virtual_microphone_packets) == 1
    packet, sample_rate = bridge.virtual_microphone_packets[0]
    assert packet.shape == (6, 1)
    assert sample_rate == 48000
    assert bridge.stats().virtual_microphone_frames == 6
    assert bridge.stats().virtual_microphone_buffered_frames == 6


def test_virtual_microphone_sink_fallback_to_silence_on_bridge_error() -> None:
    class _BrokenBridge:
        def __init__(self) -> None:
            self.calls = 0

        def write_virtual_microphone(self, audio: np.ndarray, *, sample_rate: int) -> None:
            del audio, sample_rate
            self.calls += 1
            raise RuntimeError("bridge write failed")

        def flush_virtual_microphone(self) -> None:
            return

    sink = VirtualMicrophoneSink(_BrokenBridge())
    sink.play(np.ones((12, 1), dtype=np.float32), sample_rate=48000)
    stats = sink.diagnostic_stats()

    assert int(stats["write_failures"]) >= 1
    assert int(stats["dropped_frames"]) >= 12


def test_virtual_microphone_sink_flushes_on_backpressure() -> None:
    class _Stats:
        virtual_microphone_buffered_frames = 480000
        virtual_microphone_buffer_capacity_frames = 480000

    class _BackpressureBridge:
        def __init__(self) -> None:
            self.flush_calls = 0
            self.write_calls = 0

        def stats(self) -> _Stats:
            return _Stats()

        def flush_virtual_microphone(self) -> None:
            self.flush_calls += 1

        def write_virtual_microphone(self, audio: np.ndarray, *, sample_rate: int) -> None:
            del audio, sample_rate
            self.write_calls += 1

    bridge = _BackpressureBridge()
    sink = VirtualMicrophoneSink(bridge)

    sink.play(np.ones((4800, 1), dtype=np.float32), sample_rate=48000)
    stats = sink.diagnostic_stats()

    assert bridge.flush_calls == 1
    assert bridge.write_calls == 1
    assert int(stats["backpressure_flushes"]) == 1


def test_in_memory_bridge_verifies_pcm_loopback() -> None:
    bridge = InMemoryVirtualAudioBridgeClient()

    result = bridge.verify_pcm_loopback(duration_ms=15)

    assert result.ok is True
    assert result.stats_delta_frames >= result.frames_written
