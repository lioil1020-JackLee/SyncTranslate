from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.application.session_service import SessionController
from app.application.audio_router import AudioRouter
from app.application.call_translation_policy import SYNC_VIRTUAL_AUDIO, resolve_call_translation_policy
from app.application.virtual_audio_runtime_guard import ensure_virtual_audio_runtime_ready
from app.domain.models import ErrorEvent
from app.infra.audio.capture import AudioCapture
from app.infra.audio.default_devices import SystemDefaultDeviceResolver
from app.infra.audio.playback import AudioPlayback
from app.infra.audio.routing import AudioInputManager
from app.infra.audio.sinks import SoundDevicePlaybackSink, VirtualMicrophoneSink
from app.infra.audio.loopback_capture import WasapiLoopbackCaptureSource
from app.infra.audio.sources import SoundDeviceCaptureSource, VirtualSpeakerSource
from app.infra.audio.virtual_bridge_client import VirtualAudioBridgeClient
from app.infra.asr.manager_v2 import create_asr_manager
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
    speaker_playback: AudioPlayback,
    get_local_output_device: Callable[[], str],
    on_error: Callable[[str | ErrorEvent], None] | None = None,
    on_diagnostic_event: Callable[[str], None] | None = None,
    on_asr_event: Callable[[object], None] | None = None,
    on_translation_event: Callable[[object], None] | None = None,
) -> PipelineBundle:
    session_mode = str(getattr(config.runtime, "session_mode", "meeting") or "meeting").strip().lower()
    if session_mode == "dialogue":
        guard_result = ensure_virtual_audio_runtime_ready(config)
        if on_diagnostic_event:
            if guard_result.blocked:
                on_diagnostic_event(
                    "virtual_audio_runtime_blocked "
                    f"reason={guard_result.reason} effective_mode={guard_result.effective_routing_mode}"
                )
            for warning in guard_result.warnings:
                on_diagnostic_event(f"virtual_audio_runtime_warning {warning}")
    elif on_diagnostic_event:
        on_diagnostic_event("virtual_audio_runtime_skipped session_mode=meeting")

    policy = resolve_call_translation_policy(config)
    resolver = SystemDefaultDeviceResolver(
        exclude_virtual_devices=bool(config.audio.system_devices.exclude_virtual_devices)
    )

    def _local_output_device() -> str:
        return get_local_output_device() or resolver.default_render_name()

    local_source = SoundDeviceCaptureSource(local_capture)
    bridge_client = VirtualAudioBridgeClient(bridge_path=config.audio.virtual_audio.bridge_path) if session_mode == "dialogue" else None
    if session_mode == "meeting":
        remote_source = (
            WasapiLoopbackCaptureSource()
            if config.meeting.audio_source == "system_output_loopback"
            else SoundDeviceCaptureSource(AudioCapture())
        )
        remote_sink = SoundDevicePlaybackSink(speaker_playback, lambda: "")
    else:
        remote_source = VirtualSpeakerSource(bridge_client)
        remote_sink = VirtualMicrophoneSink(bridge_client)
    input_manager = AudioInputManager(local_source=local_source, remote_source=remote_source)
    asr_manager = create_asr_manager(config, on_error=on_error, pipeline_revision=pipeline_revision)
    translator_manager = TranslatorManager(config, on_error=on_error)
    state_manager = StateManager(
        local_echo_guard_enabled=config.runtime.local_echo_guard_enabled,
        local_resume_delay_ms=config.runtime.local_echo_guard_resume_delay_ms,
        remote_resume_delay_ms=config.runtime.remote_echo_guard_resume_delay_ms,
    )
    tts_manager = TTSManager(
        config=config,
        local_sink=SoundDevicePlaybackSink(speaker_playback, _local_output_device),
        remote_sink=remote_sink,
        on_error=on_error,
    )
    if session_mode == "meeting":
        tts_manager.set_output_mode("local", "subtitle_only")
        tts_manager.set_output_mode("remote", "subtitle_only")
    elif policy.routing_mode == SYNC_VIRTUAL_AUDIO:
        tts_manager.set_output_mode("local", policy.local_channel_output_mode)
        tts_manager.set_output_mode("remote", policy.remote_channel_output_mode)
        tts_manager.set_output_mode("local", config.dialogue.remote_to_local.output_policy.replace("translated_tts", "tts").replace("direct_passthrough", "passthrough"))
        tts_manager.set_output_mode("remote", config.dialogue.local_to_remote.output_policy.replace("translated_tts", "tts").replace("direct_passthrough", "passthrough"))
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
    audio_router.refresh_runtime_config(config)
    return PipelineBundle(
        audio_router=audio_router,
        session_controller=SessionController(audio_router),
    )


__all__ = ["PipelineBundle", "build_pipeline_bundle"]
