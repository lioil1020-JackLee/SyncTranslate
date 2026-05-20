from __future__ import annotations

from app.infra.config.schema import AppConfig


def test_virtual_audio_config_round_trips_from_yaml_shape() -> None:
    cfg = AppConfig.from_dict(
        {
            "audio": {
                "routing_mode": "synctranslate_virtual_audio",
                "system_devices": {
                    "capture_role": "communications",
                    "render_role": "communications",
                    "exclude_virtual_devices": True,
                },
                "virtual_audio": {
                    "speaker_name": "SyncTranslate Virtual Speaker",
                    "microphone_name": "SyncTranslate Virtual Microphone",
                    "bridge_path": r"runtimes\audio\sync_audio_bridge.exe",
                    "sample_rate": 48000,
                    "frame_ms": 10,
                    "target_latency_ms": 120,
                    "require_driver": True,
                },
                "call_translation": {
                    "listen_remote_original": True,
                    "listen_remote_translation": True,
                    "output_local_original": False,
                    "output_local_translation": True,
                },
            }
        }
    )

    assert cfg.audio.routing_mode == "synctranslate_virtual_audio"
    assert cfg.audio.virtual_audio.speaker_name == "SyncTranslate Virtual Speaker"
    assert cfg.audio.virtual_audio.microphone_name == "SyncTranslate Virtual Microphone"
    assert cfg.audio.call_translation.output_local_translation is True
    assert cfg.audio.call_translation.output_local_original is False


def test_invalid_virtual_audio_values_are_clamped_or_normalized() -> None:
    cfg = AppConfig.from_dict(
        {
            "audio": {
                "routing_mode": "bad-mode",
                "virtual_audio": {
                    "sample_rate": 1,
                    "frame_ms": 1,
                    "target_latency_ms": 5,
                },
            }
        }
    )

    assert cfg.audio.routing_mode == "synctranslate_virtual_audio"
    assert cfg.audio.virtual_audio.sample_rate == 8000
    assert cfg.audio.virtual_audio.frame_ms == 5
    assert cfg.audio.virtual_audio.target_latency_ms == 20
