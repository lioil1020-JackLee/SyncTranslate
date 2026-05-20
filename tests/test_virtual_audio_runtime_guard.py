from __future__ import annotations

from app.application.virtual_audio_runtime_guard import ensure_virtual_audio_runtime_ready
from app.infra.audio.virtual_devices import VirtualAudioInstallStatus
from app.infra.config.schema import AppConfig


def _status(*, speaker: bool, microphone: bool) -> VirtualAudioInstallStatus:
    return VirtualAudioInstallStatus(
        installed=speaker and microphone,
        speaker_available=speaker,
        microphone_available=microphone,
        speaker_index=10 if speaker else -1,
        microphone_index=11 if microphone else -1,
        speaker_name="SyncTranslate Virtual Speaker" if speaker else "",
        microphone_name="SyncTranslate Virtual Microphone" if microphone else "",
        render_endpoints=(),
        capture_endpoints=(),
    )


def test_guard_normalizes_manual_mode_into_virtual_mode_and_checks_driver(monkeypatch) -> None:
    cfg = AppConfig()
    cfg.audio.routing_mode = "advanced_manual"

    monkeypatch.setattr(
        "app.application.virtual_audio_runtime_guard.detect_virtual_audio_install",
        lambda: _status(speaker=True, microphone=True),
    )

    result = ensure_virtual_audio_runtime_ready(cfg)

    assert result.effective_routing_mode == "synctranslate_virtual_audio"
    assert result.blocked is False
    assert cfg.audio.routing_mode == "advanced_manual"


def test_guard_blocks_when_virtual_driver_missing(monkeypatch) -> None:
    cfg = AppConfig()
    cfg.audio.routing_mode = "synctranslate_virtual_audio"
    cfg.audio.virtual_audio.require_driver = True

    monkeypatch.setattr(
        "app.application.virtual_audio_runtime_guard.detect_virtual_audio_install",
        lambda: _status(speaker=False, microphone=True),
    )

    result = ensure_virtual_audio_runtime_ready(cfg)

    assert result.blocked is True
    assert result.effective_routing_mode == "synctranslate_virtual_audio"
    assert "virtual_audio_driver_unavailable" in result.reason
    assert cfg.audio.routing_mode == "synctranslate_virtual_audio"


def test_guard_skips_driver_requirement_when_disabled(monkeypatch) -> None:
    cfg = AppConfig()
    cfg.audio.routing_mode = "synctranslate_virtual_audio"
    cfg.audio.virtual_audio.require_driver = False

    def _should_not_run() -> VirtualAudioInstallStatus:
        raise AssertionError("driver detection should not run when require_driver is false")

    monkeypatch.setattr(
        "app.application.virtual_audio_runtime_guard.detect_virtual_audio_install",
        _should_not_run,
    )

    result = ensure_virtual_audio_runtime_ready(cfg)

    assert result.blocked is False
    assert result.effective_routing_mode == "synctranslate_virtual_audio"
    assert cfg.audio.routing_mode == "synctranslate_virtual_audio"


def test_guard_reports_bridge_path_warning(monkeypatch, tmp_path) -> None:
    cfg = AppConfig()
    cfg.audio.routing_mode = "synctranslate_virtual_audio"
    cfg.audio.virtual_audio.bridge_path = str(tmp_path / "missing_sync_audio_bridge.exe")

    monkeypatch.setattr(
        "app.application.virtual_audio_runtime_guard.detect_virtual_audio_install",
        lambda: _status(speaker=True, microphone=True),
    )

    result = ensure_virtual_audio_runtime_ready(cfg)

    assert result.blocked is False
    assert any(item.startswith("bridge_path_missing:") for item in result.warnings)