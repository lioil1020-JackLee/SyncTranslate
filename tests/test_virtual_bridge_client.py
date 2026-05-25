from __future__ import annotations

import time
import os

import numpy as np
import pytest

from app.infra.audio.bridge_protocol import decode_pcm16_stereo_packet
from app.infra.audio.synctranslate_driver_client import SyncTranslateDriverAudioClient
from app.infra.audio.sources import VirtualSpeakerSource
from app.infra.audio.sinks import VirtualMicrophoneSink
from app.infra.audio.virtual_bridge_client import InMemoryVirtualAudioBridgeClient, VirtualAudioBridgeClient


RUN_BRIDGE_INTEGRATION = os.environ.get("SYNCTRANSLATE_RUN_BRIDGE_INTEGRATION") == "1"


@pytest.mark.integration
@pytest.mark.skipif(not RUN_BRIDGE_INTEGRATION, reason="Requires real SyncTranslate bridge/driver transport.")
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


@pytest.mark.integration
@pytest.mark.skipif(not RUN_BRIDGE_INTEGRATION, reason="Requires real SyncTranslate bridge/driver transport.")
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


@pytest.mark.integration
@pytest.mark.skipif(not RUN_BRIDGE_INTEGRATION, reason="Requires real SyncTranslate bridge/driver transport.")
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


@pytest.mark.integration
@pytest.mark.skipif(not RUN_BRIDGE_INTEGRATION, reason="Requires real SyncTranslate bridge/driver transport.")
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
    assert received[0][0].shape == (4, 2)
    assert received[0][1] == 48000.0
    assert source.stats().frame_count == 4


def test_virtual_microphone_sink_writes_to_bridge() -> None:
    bridge = InMemoryVirtualAudioBridgeClient()
    sink = VirtualMicrophoneSink(bridge)

    sink.play(np.full((6, 1), 0.25, dtype=np.float32), sample_rate=48000)

    assert len(bridge.virtual_microphone_packets) == 1
    packet, sample_rate = bridge.virtual_microphone_packets[0]
    assert packet.shape == (6, 2)
    assert sample_rate == 48000
    assert bridge.stats().virtual_microphone_frames == 6
    assert bridge.stats().virtual_microphone_buffered_frames == 6


def test_bridge_client_virtual_mic_write_uses_protocol_v2_packet(monkeypatch) -> None:
    client = VirtualAudioBridgeClient(bridge_path="")
    captured: dict[str, object] = {}

    monkeypatch.setattr(client, "_try_write_driver_virtual_microphone", lambda *_args, **_kwargs: False)

    def _fake_request(request: dict[str, object]) -> dict[str, object]:
        captured.update(request)
        return {"ok": True}

    monkeypatch.setattr(client, "_request", _fake_request)

    client.write_virtual_microphone(np.ones((5, 1), dtype=np.float32) * 0.25, sample_rate=16000)

    assert captured["cmd"] == "write_virtual_microphone"
    packet = captured["packet"]
    assert isinstance(packet, dict)
    assert packet["protocol_version"] == 2
    decoded, sample_rate = decode_pcm16_stereo_packet(packet)
    assert sample_rate == 48000
    assert decoded.shape == (15, 2)
    np.testing.assert_allclose(decoded[:, 0], decoded[:, 1], atol=1.0 / 32767.0)


def test_bridge_pcm_loopback_verification_uses_bridge_path_not_driver_fast_path(monkeypatch) -> None:
    client = VirtualAudioBridgeClient(bridge_path="")
    state = {"frames": 0, "write_seen": False}

    def _driver_write_must_not_run(*_args, **_kwargs) -> bool:
        raise AssertionError("bridge loopback verification must not use driver fast-path")

    def _fake_request(request: dict[str, object]) -> dict[str, object]:
        cmd = str(request.get("cmd") or "")
        if cmd == "get_stats":
            return {
                "ok": True,
                "connected": True,
                "remote_input_running": False,
                "remote_input_frames": 0,
                "virtual_microphone_frames": state["frames"],
                "virtual_microphone_buffered_frames": state["frames"],
                "virtual_microphone_dropped_frames": 0,
                "virtual_microphone_buffer_capacity_frames": 0,
                "virtual_microphone_shared_memory_name": "",
                "virtual_microphone_event_name": "",
                "last_error": "",
            }
        if cmd == "write_virtual_microphone":
            packet = request.get("packet")
            assert isinstance(packet, dict)
            decoded, sample_rate = decode_pcm16_stereo_packet(packet)
            assert sample_rate == 48000
            assert decoded.dtype == np.float32
            assert decoded.shape[1] == 2
            state["frames"] += int(decoded.shape[0])
            state["write_seen"] = True
            return {"ok": True, "written_frames": int(decoded.shape[0])}
        raise AssertionError(f"unexpected bridge command: {cmd}")

    monkeypatch.setattr(client, "_try_write_driver_virtual_microphone", _driver_write_must_not_run)
    monkeypatch.setattr(client, "_request", _fake_request)

    result = client.verify_pcm_loopback(duration_ms=10, timeout_ms=1)

    assert state["write_seen"] is True
    assert result.ok is True
    assert result.frames_written > 0
    assert result.stats_delta_frames >= result.frames_written


def test_bridge_client_remote_poll_decodes_protocol_v2_packet(monkeypatch) -> None:
    client = VirtualAudioBridgeClient(bridge_path="")
    received: list[tuple[np.ndarray, float]] = []
    packet = client_payload = None
    del client_payload

    from app.infra.audio.bridge_protocol import encode_pcm16_stereo_packet

    packet = encode_pcm16_stereo_packet(
        np.column_stack((np.ones(4, dtype=np.float32) * 0.1, np.ones(4, dtype=np.float32) * -0.2)),
        sample_rate=48000,
    )
    calls = {"count": 0}

    def _fake_wait(_timeout: float) -> bool:
        calls["count"] += 1
        return calls["count"] > 1

    monkeypatch.setattr(client._remote_input_stop, "wait", _fake_wait)
    monkeypatch.setattr(client, "_request", lambda _request: {"packet": packet})
    with client._consumer_lock:
        client._consumers.append(lambda audio, sample_rate: received.append((audio, sample_rate)))

    client._poll_remote_input()

    assert received
    assert received[0][0].dtype == np.float32
    assert received[0][0].shape == (4, 2)
    assert received[0][1] == 48000.0


def test_driver_client_virtual_mic_boundary_is_pcm16_stereo_48k() -> None:
    audio = np.ones((4, 1), dtype=np.float32) * 0.5

    payload, frames = SyncTranslateDriverAudioClient._to_pcm16_stereo_bytes(audio, sample_rate=16000)
    pcm = np.frombuffer(payload, dtype="<i2").reshape((-1, 2))

    assert frames == 12
    assert pcm.shape == (12, 2)
    np.testing.assert_array_equal(pcm[:, 0], pcm[:, 1])


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
