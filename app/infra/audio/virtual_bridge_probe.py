from __future__ import annotations

from dataclasses import dataclass

from app.infra.audio.virtual_bridge_client import VirtualAudioBridgeClient


@dataclass(frozen=True, slots=True)
class VirtualBridgeProbeResult:
    ready: bool
    error: str
    connected: bool
    remote_input_running: bool
    virtual_microphone_buffered_frames: int
    shared_memory_name: str
    event_name: str
    remote_input_frames: int = 0
    remote_input_buffered_frames: int = 0
    remote_input_buffer_capacity_frames: int = 0
    virtual_microphone_frames: int = 0
    virtual_microphone_dropped_frames: int = 0
    virtual_microphone_buffer_capacity_frames: int = 0
    heartbeat_ok: bool = False
    heartbeat_roundtrip_ms: float = 0.0
    loopback_ok: bool = False
    loopback_latency_ms: float = 0.0


def probe_virtual_audio_bridge(bridge_path: str) -> VirtualBridgeProbeResult:
    client = VirtualAudioBridgeClient(bridge_path=bridge_path)
    try:
        stats = client.stats()
        error = str(stats.last_error or "")
        heartbeat = client.heartbeat(timeout_ms=250)
        loopback = client.verify_pcm_loopback(duration_ms=20, timeout_ms=250)
        if not heartbeat.ok and not error:
            error = str(heartbeat.error or "bridge_heartbeat_failed")
        if not loopback.ok and not error:
            error = str(loopback.error or "bridge_loopback_failed")
        return VirtualBridgeProbeResult(
            ready=bool(stats.connected and heartbeat.ok and loopback.ok and not error),
            error=error,
            connected=bool(stats.connected),
            remote_input_running=bool(stats.remote_input_running),
            remote_input_frames=int(stats.remote_input_frames),
            remote_input_buffered_frames=int(stats.remote_input_buffered_frames),
            remote_input_buffer_capacity_frames=int(stats.remote_input_buffer_capacity_frames),
            virtual_microphone_buffered_frames=int(stats.virtual_microphone_buffered_frames),
            virtual_microphone_frames=int(stats.virtual_microphone_frames),
            virtual_microphone_dropped_frames=int(stats.virtual_microphone_dropped_frames),
            virtual_microphone_buffer_capacity_frames=int(stats.virtual_microphone_buffer_capacity_frames),
            shared_memory_name=str(stats.virtual_microphone_shared_memory_name or ""),
            event_name=str(stats.virtual_microphone_event_name or ""),
            heartbeat_ok=bool(heartbeat.ok),
            heartbeat_roundtrip_ms=float(heartbeat.roundtrip_ms),
            loopback_ok=bool(loopback.ok),
            loopback_latency_ms=float(loopback.latency_ms),
        )
    except Exception as exc:
        return VirtualBridgeProbeResult(
            ready=False,
            error=str(exc),
            connected=False,
            remote_input_running=False,
            remote_input_frames=0,
            remote_input_buffered_frames=0,
            remote_input_buffer_capacity_frames=0,
            virtual_microphone_buffered_frames=0,
            virtual_microphone_frames=0,
            virtual_microphone_dropped_frames=0,
            virtual_microphone_buffer_capacity_frames=0,
            shared_memory_name="",
            event_name="",
            heartbeat_ok=False,
            heartbeat_roundtrip_ms=0.0,
            loopback_ok=False,
            loopback_latency_ms=0.0,
        )
    finally:
        client.close()


__all__ = ["VirtualBridgeProbeResult", "probe_virtual_audio_bridge"]
