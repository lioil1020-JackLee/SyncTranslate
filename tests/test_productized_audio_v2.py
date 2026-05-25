from __future__ import annotations

import numpy as np

from app.infra.audio.format_adapter import (
    ensure_float32_frame,
    to_asr_mono_16k,
    to_output_float32_stereo_48k,
    to_pcm16_stereo_48k,
)
from app.infra.config.schema import AppConfig


def test_asr_branch_converts_48k_stereo_to_16k_mono_float32() -> None:
    audio = np.column_stack(
        (
            np.linspace(-0.5, 0.5, 4800, dtype=np.float32),
            np.linspace(0.5, -0.5, 4800, dtype=np.float32),
        )
    )
    frame = ensure_float32_frame(audio, 48000, "test", "dialogue_remote")

    converted = to_asr_mono_16k(frame)

    assert converted.dtype == np.float32
    assert converted.ndim == 1
    assert abs(converted.shape[0] - 1600) <= 2


def test_direct_passthrough_stereo_remains_stereo_until_boundary() -> None:
    audio = np.zeros((480, 2), dtype=np.float32)
    audio[:, 0] = 0.25
    audio[:, 1] = -0.25
    frame = ensure_float32_frame(audio, 48000, "test", "dialogue_local")

    converted = to_output_float32_stereo_48k(frame)

    assert converted.shape == (480, 2)
    assert np.allclose(converted[:, 0], 0.25)
    assert np.allclose(converted[:, 1], -0.25)


def test_mono_passthrough_duplicates_to_stereo_boundary() -> None:
    mono = np.linspace(-0.2, 0.2, 480, dtype=np.float32).reshape(-1, 1)
    frame = ensure_float32_frame(mono, 48000, "test", "dialogue_local")

    converted = to_output_float32_stereo_48k(frame)

    assert converted.shape == (480, 2)
    assert np.allclose(converted[:, 0], converted[:, 1])


def test_boundary_conversion_outputs_pcm16_stereo_bytes() -> None:
    mono = np.ones((480, 1), dtype=np.float32) * 0.5
    frame = ensure_float32_frame(mono, 48000, "test", "boundary")

    payload = to_pcm16_stereo_48k(frame)

    assert isinstance(payload, bytes)
    assert len(payload) == 480 * 2 * 2


def test_config_defaults_are_productized_meeting_mode() -> None:
    cfg = AppConfig.from_dict({})

    assert cfg.runtime.session_mode == "meeting"
    assert cfg.runtime.asr_language_mode == "fixed"
    assert cfg.meeting.tts_enabled is False
    assert cfg.audio.virtual_audio.sample_rate == 48000
    assert cfg.audio.virtual_audio.channels == 2
    assert cfg.audio.virtual_audio.bit_depth == 16
    assert cfg.audio.virtual_audio.dtype == "int16"


def test_config_auto_languages_migrate_to_fixed_values() -> None:
    cfg = AppConfig.from_dict(
        {
            "runtime": {
                "session_mode": "dialogue",
                "remote_asr_language": "auto",
                "local_asr_language": "auto",
            }
        }
    )

    assert cfg.runtime.session_mode == "dialogue"
    assert cfg.runtime.remote_asr_language == "en"
    assert cfg.runtime.local_asr_language == "zh-TW"
    assert cfg.dialogue.remote_asr_language == "en"
    assert cfg.dialogue.local_asr_language == "zh-TW"

