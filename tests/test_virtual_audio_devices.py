from __future__ import annotations

from unittest.mock import patch

from app.infra.audio.virtual_devices import detect_virtual_audio_install


HOSTAPIS = [
    {"name": "MME"},
    {"name": "Windows WASAPI"},
    {"name": "Windows WDM-KS"},
]


def test_detect_virtual_audio_install_accepts_sysvad_poc_endpoint_names() -> None:
    devices = [
        (
            10,
            {
                "name": "喇叭 (SyncTranslate Virtual Audio Device)",
                "hostapi": 1,
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 44100.0,
            },
        ),
        (
            11,
            {
                "name": "SyncTranslate Virtual Microphone (SyncTranslate Virtual Audio Device)",
                "hostapi": 1,
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 48000.0,
            },
        ),
    ]

    with (
        patch("app.infra.audio.device_registry.sd.query_hostapis", return_value=HOSTAPIS),
        patch("app.infra.audio.virtual_devices.list_indexed_devices", return_value=devices),
    ):
        status = detect_virtual_audio_install()

    assert status.installed is True
    assert status.speaker_available is True
    assert status.microphone_available is True
    assert status.speaker_name == "喇叭 (SyncTranslate Virtual Audio Device)"
    assert status.microphone_name == "SyncTranslate Virtual Microphone (SyncTranslate Virtual Audio Device)"


def test_detect_virtual_audio_install_prefers_speaker_semantics_over_aux_render_names() -> None:
    devices = [
        (
            9,
            {
                "name": "SinkDescription Sample (SyncTranslate Virtual Audio Device)",
                "hostapi": 1,
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000.0,
            },
        ),
        (
            10,
            {
                "name": "喇叭 (SyncTranslate Virtual Audio Device)",
                "hostapi": 1,
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000.0,
            },
        ),
        (
            11,
            {
                "name": "SyncTranslate Virtual Microphone (SyncTranslate Virtual Audio Device)",
                "hostapi": 1,
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 48000.0,
            },
        ),
    ]

    with (
        patch("app.infra.audio.device_registry.sd.query_hostapis", return_value=HOSTAPIS),
        patch("app.infra.audio.virtual_devices.list_indexed_devices", return_value=devices),
    ):
        status = detect_virtual_audio_install()

    assert status.speaker_name == "喇叭 (SyncTranslate Virtual Audio Device)"


def test_detect_virtual_audio_install_prefers_explicit_virtual_speaker_name() -> None:
    devices = [
        (
            5,
            {
                "name": "喇叭 (SyncTranslate Virtual Audio Device)",
                "hostapi": 1,
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000.0,
            },
        ),
        (
            6,
            {
                "name": "Output 1 (SyncTranslate Virtual Speaker)",
                "hostapi": 2,
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 44100.0,
            },
        ),
        (
            7,
            {
                "name": "SyncTranslate Virtual Microphone 2 (SyncTranslate Virtual Microphone)",
                "hostapi": 2,
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 16000.0,
            },
        ),
    ]

    with (
        patch("app.infra.audio.device_registry.sd.query_hostapis", return_value=HOSTAPIS),
        patch("app.infra.audio.virtual_devices.list_indexed_devices", return_value=devices),
    ):
        status = detect_virtual_audio_install()

    assert status.speaker_name == "喇叭 (SyncTranslate Virtual Audio Device)"
    assert status.microphone_name == "SyncTranslate Virtual Microphone 2 (SyncTranslate Virtual Microphone)"


def test_detect_virtual_audio_install_prefers_explicit_name_within_same_hostapi() -> None:
    devices = [
        (
            5,
            {
                "name": "External Microphone Headphone (SyncTranslate Virtual Audio Device)",
                "hostapi": 1,
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 48000.0,
            },
        ),
        (
            6,
            {
                "name": "SyncTranslate Virtual Microphone (SyncTranslate Virtual Audio Device)",
                "hostapi": 1,
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 48000.0,
            },
        ),
    ]

    with (
        patch("app.infra.audio.device_registry.sd.query_hostapis", return_value=HOSTAPIS),
        patch("app.infra.audio.virtual_devices.list_indexed_devices", return_value=devices),
    ):
        status = detect_virtual_audio_install()

    assert status.microphone_name == "SyncTranslate Virtual Microphone (SyncTranslate Virtual Audio Device)"


def test_detect_virtual_audio_install_requires_capture_and_render() -> None:
    devices = [
        (
            10,
            {
                "name": "喇叭 (SyncTranslate Virtual Audio Device)",
                "hostapi": 1,
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 44100.0,
            },
        ),
    ]

    with (
        patch("app.infra.audio.device_registry.sd.query_hostapis", return_value=HOSTAPIS),
        patch("app.infra.audio.virtual_devices.list_indexed_devices", return_value=devices),
    ):
        status = detect_virtual_audio_install()

    assert status.installed is False
    assert status.speaker_available is True
    assert status.microphone_available is False
