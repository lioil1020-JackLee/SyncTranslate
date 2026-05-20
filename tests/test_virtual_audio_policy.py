from __future__ import annotations

from app.application.call_translation_policy import resolve_call_translation_policy
from app.infra.config.schema import AppConfig


def test_virtual_audio_policy_maps_acer_style_toggles_to_output_modes() -> None:
    cfg = AppConfig()
    cfg.audio.routing_mode = "synctranslate_virtual_audio"
    cfg.audio.call_translation.listen_remote_original = True
    cfg.audio.call_translation.listen_remote_translation = True
    cfg.audio.call_translation.output_local_original = False
    cfg.audio.call_translation.output_local_translation = True

    policy = resolve_call_translation_policy(cfg)

    assert policy.uses_virtual_audio
    assert policy.local_channel_output_mode == "tts"
    assert policy.remote_channel_output_mode == "tts"


def test_call_translation_policy_can_disable_local_output_to_meeting() -> None:
    cfg = AppConfig()
    cfg.audio.routing_mode = "synctranslate_virtual_audio"
    cfg.audio.call_translation.output_local_original = False
    cfg.audio.call_translation.output_local_translation = False

    policy = resolve_call_translation_policy(cfg)

    assert policy.remote_channel_output_mode == "subtitle_only"

