from __future__ import annotations

from app.application.config_apply_service import ConfigApplyService
from app.infra.config.schema import AppConfig
from app.infra.config.schema import AudioRouteConfig


class _CaptureStub:
    pass


class _PlaybackStub:
    pass


class _PageStub:
    def selected_audio_routes(self):
        return AudioRouteConfig()

    def update_config(self, _config) -> None:
        return


class _VolumeControllerStub:
    def __init__(self) -> None:
        self.calls = 0

    def apply_audio_route_config(self, _audio: AudioRouteConfig) -> None:
        self.calls += 1


def _service(volume_controller: _VolumeControllerStub):
    return ConfigApplyService(
        meeting_capture=_CaptureStub(),
        local_capture=_CaptureStub(),
        speaker_playback=_PlaybackStub(),
        meeting_playback=_PlaybackStub(),
        audio_routing_page=_PageStub(),
        live_caption_page=_PageStub(),
        local_ai_page=_PageStub(),
        device_volume_controller=volume_controller,
    )


def test_sync_ui_to_config_updates_audio_without_level_mutation() -> None:
    volume_controller = _VolumeControllerStub()
    service = _service(volume_controller)
    config = AppConfig()

    service.sync_ui_to_config(config)

    assert isinstance(config.audio, AudioRouteConfig)


def test_virtual_audio_mode_does_not_apply_system_device_volume() -> None:
    volume_controller = _VolumeControllerStub()
    service = _service(volume_controller)
    audio = AudioRouteConfig(routing_mode="synctranslate_virtual_audio")

    service.apply_audio_route_levels(audio)

    assert volume_controller.calls == 0


def test_legacy_manual_mode_value_is_still_treated_as_virtual_and_skips_system_volume_apply() -> None:
    volume_controller = _VolumeControllerStub()
    service = _service(volume_controller)
    audio = AudioRouteConfig(routing_mode="advanced_manual")

    service.apply_audio_route_levels(audio)

    assert volume_controller.calls == 0
