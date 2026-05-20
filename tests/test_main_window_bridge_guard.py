from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.infra.config.schema import AppConfig
from app.infra.audio.virtual_bridge_probe import VirtualBridgeProbeResult
from app.ui.main_window import MainWindow


class _DummyWindow:
    _probe_virtual_bridge_runtime = MainWindow._probe_virtual_bridge_runtime
    _describe_bridge_validation_message = MainWindow._describe_bridge_validation_message
    _validate_virtual_bridge_runtime_or_raise = MainWindow._validate_virtual_bridge_runtime_or_raise

    def __init__(self) -> None:
        self.config = AppConfig()
        self._bridge_probe_cache = None
        self._bridge_probe_cache_ts = 0.0


def _ready_probe() -> VirtualBridgeProbeResult:
    return VirtualBridgeProbeResult(
        ready=True,
        error="",
        connected=True,
        remote_input_running=False,
        virtual_microphone_buffered_frames=0,
        shared_memory_name="shm",
        event_name=r"Local\\SyncTranslateVirtualMicrophoneReady",
        heartbeat_ok=True,
        heartbeat_roundtrip_ms=2.5,
        loopback_ok=True,
        loopback_latency_ms=4.1,
    )


def _not_ready_probe(error: str = "bridge_unavailable") -> VirtualBridgeProbeResult:
    return VirtualBridgeProbeResult(
        ready=False,
        error=error,
        connected=False,
        remote_input_running=False,
        virtual_microphone_buffered_frames=0,
        shared_memory_name="",
        event_name="",
        heartbeat_ok=False,
        heartbeat_roundtrip_ms=0.0,
        loopback_ok=False,
        loopback_latency_ms=0.0,
    )


def test_bridge_guard_message_when_probe_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    window = _DummyWindow()

    monkeypatch.setattr("app.ui.main_window.probe_virtual_audio_bridge", lambda _path: _ready_probe())

    message, is_error = MainWindow._describe_bridge_validation_message(window)  # type: ignore[arg-type]

    assert is_error is False
    assert "Bridge 心跳" in message
    assert "Loopback" in message


def test_bridge_guard_message_when_probe_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    window = _DummyWindow()

    monkeypatch.setattr("app.ui.main_window.probe_virtual_audio_bridge", lambda _path: _not_ready_probe("offline"))

    message, is_error = MainWindow._describe_bridge_validation_message(window)  # type: ignore[arg-type]

    assert is_error is True
    assert "Bridge 未連線" in message


def test_validate_virtual_bridge_runtime_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    window = _DummyWindow()

    monkeypatch.setattr("app.ui.main_window.probe_virtual_audio_bridge", lambda _path: _not_ready_probe("offline"))

    with pytest.raises(ValueError):
        MainWindow._validate_virtual_bridge_runtime_or_raise(window)  # type: ignore[arg-type]


def test_validate_virtual_bridge_runtime_passes_when_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    window = _DummyWindow()

    monkeypatch.setattr("app.ui.main_window.probe_virtual_audio_bridge", lambda _path: _ready_probe())

    MainWindow._validate_virtual_bridge_runtime_or_raise(window)  # type: ignore[arg-type]


def test_validate_virtual_bridge_runtime_skips_when_bridge_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    window = _DummyWindow()
    window.config.audio.virtual_audio.bridge_enabled = False

    monkeypatch.setattr(
        "app.ui.main_window.probe_virtual_audio_bridge",
        lambda _path: (_ for _ in ()).throw(RuntimeError("should not be called")),
    )

    MainWindow._validate_virtual_bridge_runtime_or_raise(window)  # type: ignore[arg-type]
