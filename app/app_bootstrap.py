from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.audio_capture import AudioCapture
from app.audio_input_manager import AudioInputManager
from app.audio_playback import AudioPlayback
from app.audio_router import AudioRouter
from app.asr_manager import ASRManager
from app.events import ErrorEvent
from app.schemas import AppConfig
from app.session_controller import SessionController
from app.state_manager import StateManager
from app.transcript_buffer import TranscriptBuffer
from app.translator_manager import TranslatorManager
from app.tts_manager import TTSManager


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
        on_diagnostic_event=on_diagnostic_event,
    )
    tts_manager.set_callbacks(
        on_play_start=audio_router.handle_tts_play_start,
        on_play_end=audio_router.handle_tts_play_end,
    )
    return PipelineBundle(
        audio_router=audio_router,
        session_controller=SessionController(audio_router),
    )
