from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.application.session_service import SessionController
from app.application.audio_router import AudioRouter
from app.domain.events import ErrorEvent
from app.infra.audio.capture import AudioCapture
from app.infra.audio.playback import AudioPlayback
from app.infra.audio.routing import AudioInputManager
from app.infra.asr.streaming_pipeline import ASRManager
from app.infra.translation.engine import TranslatorManager
from app.infra.tts.playback_queue import TTSManager
from app.infra.config.schema import AppConfig
from app.domain.runtime_state import StateManager
from app.application.transcript_service import TranscriptBuffer


@dataclass(slots=True)
class PipelineBundle:
    audio_router: AudioRouter
    session_controller: SessionController


def build_pipeline_bundle(
    *,
    config: AppConfig,
    pipeline_revision: int,
    transcript_buffer: TranscriptBuffer,
    local_capture: AudioCapture,
    meeting_capture: AudioCapture,
    speaker_playback: AudioPlayback,
    meeting_playback: AudioPlayback,
    get_local_output_device: Callable[[], str],
    get_remote_output_device: Callable[[], str],
    on_error: Callable[[str | ErrorEvent], None] | None = None,
    on_diagnostic_event: Callable[[str], None] | None = None,
    on_asr_event: Callable[[object], None] | None = None,
    on_translation_event: Callable[[object], None] | None = None,
) -> PipelineBundle:
    input_manager = AudioInputManager(local_capture=local_capture, remote_capture=meeting_capture)
    asr_manager = ASRManager(config, on_error=on_error, pipeline_revision=pipeline_revision)
    translator_manager = TranslatorManager(config, on_error=on_error)
    state_manager = StateManager(
        local_echo_guard_enabled=config.runtime.local_echo_guard_enabled,
        local_resume_delay_ms=config.runtime.local_echo_guard_resume_delay_ms,
        remote_resume_delay_ms=config.runtime.remote_echo_guard_resume_delay_ms,
    )
    tts_manager = TTSManager(
        config=config,
        local_playback=speaker_playback,
        remote_playback=meeting_playback,
        get_local_output_device=get_local_output_device,
        get_remote_output_device=get_remote_output_device,
        on_error=on_error,
    )
    audio_router = AudioRouter(
        transcript_buffer=transcript_buffer,
        input_manager=input_manager,
        asr_manager=asr_manager,
        translator_manager=translator_manager,
        tts_manager=tts_manager,
        state_manager=state_manager,
        on_error=on_error,
        on_asr_event=on_asr_event,
        on_translation_event=on_translation_event,
        on_diagnostic_event=on_diagnostic_event,
        async_translation=True,
    )
    tts_manager.set_callbacks(
        on_play_start=audio_router.handle_tts_play_start,
        on_play_end=audio_router.handle_tts_play_end,
    )
    return PipelineBundle(
        audio_router=audio_router,
        session_controller=SessionController(audio_router),
    )


__all__ = ["PipelineBundle", "build_pipeline_bundle"]
