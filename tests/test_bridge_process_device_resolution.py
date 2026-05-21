from __future__ import annotations

from app.infra.audio.bridge_process import BridgeCommandHandler


def test_resolve_remote_input_capture_device_localized_speaker_name(monkeypatch) -> None:
    requested = "喇叭 (SyncTranslate Virtual Audio Device)"
    expected_loopback = "喇叭 (SyncTranslate Virtual Audio Device) (Loopback)"

    indexed_devices = [
        (
            14,
            {
                "name": requested,
                "hostapi": 2,
                "max_input_channels": 0,
                "max_output_channels": 2,
            },
        ),
        (
            21,
            {
                "name": expected_loopback,
                "hostapi": 2,
                "max_input_channels": 2,
                "max_output_channels": 0,
            },
        ),
        (
            22,
            {
                "name": "SyncTranslate Virtual Microphone",
                "hostapi": 2,
                "max_input_channels": 1,
                "max_output_channels": 0,
            },
        ),
    ]

    monkeypatch.setattr("app.infra.audio.bridge_process.list_indexed_devices", lambda: indexed_devices)

    resolved = BridgeCommandHandler._resolve_remote_input_capture_device(requested)
    assert resolved == expected_loopback


def test_resolve_remote_input_capture_device_keeps_input_capable_requested(monkeypatch) -> None:
    requested = "SyncTranslate Virtual Speaker (Loopback)"
    indexed_devices = [
        (
            14,
            {
                "name": requested,
                "hostapi": 0,
                "max_input_channels": 2,
                "max_output_channels": 0,
            },
        ),
    ]

    monkeypatch.setattr("app.infra.audio.bridge_process.list_indexed_devices", lambda: indexed_devices)

    resolved = BridgeCommandHandler._resolve_remote_input_capture_device(requested)
    assert resolved == requested


def test_resolve_remote_input_capture_device_non_sync_name_unchanged(monkeypatch) -> None:
    requested = "Speakers (Realtek High Definition Audio)"
    indexed_devices = [
        (
            1,
            {
                "name": requested,
                "hostapi": 0,
                "max_input_channels": 0,
                "max_output_channels": 2,
            },
        ),
    ]

    monkeypatch.setattr("app.infra.audio.bridge_process.list_indexed_devices", lambda: indexed_devices)

    resolved = BridgeCommandHandler._resolve_remote_input_capture_device(requested)
    assert resolved == requested
