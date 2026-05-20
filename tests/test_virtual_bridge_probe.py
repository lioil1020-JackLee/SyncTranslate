from __future__ import annotations

from app.infra.audio.virtual_bridge_probe import probe_virtual_audio_bridge
from app.infra.audio.virtual_bridge_client import VirtualBridgeStats


def test_probe_reports_not_ready_on_bridge_exception(monkeypatch) -> None:
    class _BrokenClient:
        def __init__(self, *, bridge_path: str) -> None:
            self.bridge_path = bridge_path

        def stats(self):
            raise RuntimeError("bridge offline")

        def heartbeat(self, *, timeout_ms: int = 200):
            del timeout_ms
            raise RuntimeError("bridge offline")

        def verify_pcm_loopback(self, *, duration_ms: int = 20, timeout_ms: int = 200):
            del duration_ms, timeout_ms
            raise RuntimeError("bridge offline")

        def close(self) -> None:
            return

    monkeypatch.setattr("app.infra.audio.virtual_bridge_probe.VirtualAudioBridgeClient", _BrokenClient)

    result = probe_virtual_audio_bridge("missing.exe")

    assert result.ready is False
    assert "bridge offline" in result.error
    assert result.connected is False


def test_probe_reports_ready_when_bridge_connected(monkeypatch) -> None:
    class _ReadyClient:
        def __init__(self, *, bridge_path: str) -> None:
            self.bridge_path = bridge_path

        def stats(self):
            return VirtualBridgeStats(
                connected=True,
                remote_input_running=False,
                remote_input_frames=0,
                virtual_microphone_frames=0,
                virtual_microphone_buffered_frames=0,
                virtual_microphone_dropped_frames=0,
                virtual_microphone_shared_memory_name="SyncTranslateVirtualMicShared",
                virtual_microphone_event_name=r"Local\SyncTranslateVirtualMicrophoneReady",
                last_error="",
            )

        def heartbeat(self, *, timeout_ms: int = 200):
            del timeout_ms
            from app.infra.audio.virtual_bridge_client import VirtualBridgeHeartbeatResult

            return VirtualBridgeHeartbeatResult(ok=True, roundtrip_ms=3.0, error="")

        def verify_pcm_loopback(self, *, duration_ms: int = 20, timeout_ms: int = 200):
            del duration_ms, timeout_ms
            from app.infra.audio.virtual_bridge_client import VirtualBridgePcmVerificationResult

            return VirtualBridgePcmVerificationResult(
                ok=True,
                frames_written=960,
                stats_delta_frames=960,
                shared_memory_frames=960,
                event_signaled=True,
                latency_ms=4.0,
                error="",
            )

        def close(self) -> None:
            return

    monkeypatch.setattr("app.infra.audio.virtual_bridge_probe.VirtualAudioBridgeClient", _ReadyClient)

    result = probe_virtual_audio_bridge("bridge.exe")

    assert result.ready is True
    assert result.connected is True
    assert result.error == ""
    assert result.shared_memory_name == "SyncTranslateVirtualMicShared"
    assert result.heartbeat_ok is True
    assert result.loopback_ok is True
