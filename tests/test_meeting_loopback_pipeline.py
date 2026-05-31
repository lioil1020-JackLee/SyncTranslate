from __future__ import annotations

from app.application.transcript_service import TranscriptBuffer
from app.bootstrap import dependency_container
from app.infra.audio.capture import AudioCapture
from app.infra.audio.loopback_capture import WasapiLoopbackCaptureSource
from app.infra.audio.playback import AudioPlayback
from app.infra.config.schema import AppConfig


class _AsrManager:
    def configure_pipeline(self, config, revision) -> None:
        pass


class _TranslatorManager:
    def __init__(self, *args, **kwargs) -> None:
        pass


class _TtsManager:
    def __init__(self, *args, **kwargs) -> None:
        self.modes: dict[str, str] = {}

    def set_output_mode(self, channel: str, mode: str) -> None:
        self.modes[channel] = mode

    def set_callbacks(self, **kwargs) -> None:
        pass


def test_meeting_output_loopback_uses_wasapi_loopback_capture_source(monkeypatch) -> None:
    config = AppConfig()
    config.runtime.session_mode = "meeting"
    config.meeting.audio_source = "system_output_loopback"
    config.meeting.output_loopback_device = "Speakers"

    monkeypatch.setattr(dependency_container, "create_asr_manager", lambda *args, **kwargs: _AsrManager())
    monkeypatch.setattr(dependency_container, "TranslatorManager", _TranslatorManager)
    monkeypatch.setattr(dependency_container, "TTSManager", _TtsManager)

    bundle = dependency_container.build_pipeline_bundle(
        config=config,
        pipeline_revision=1,
        transcript_buffer=TranscriptBuffer(),
        local_capture=AudioCapture(),
        speaker_playback=AudioPlayback(),
        get_local_output_device=lambda: "",
    )

    input_manager = getattr(bundle.audio_router, "_input_manager")
    captures = getattr(input_manager, "_captures")

    assert isinstance(captures["remote"], WasapiLoopbackCaptureSource)
