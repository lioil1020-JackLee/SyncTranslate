from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import numpy as np

from app.asr_manager import ASRManager, ASREventWithSource
from app.audio_input_manager import AudioInputManager
from app.events import ErrorEvent
from app.schemas import AudioRouteConfig
from app.state_manager import StateManager
from app.transcript_buffer import TranscriptBuffer
from app.translator_manager import TranslatorManager
from app.tts_manager import TTSManager


@dataclass(slots=True)
class RouterStats:
    running: bool
    active_sources: list[str]
    state: dict[str, object]
    capture: dict[str, dict[str, object]]
    asr: dict[str, dict[str, object]]
    tts: dict[str, object]


class AudioRouter:
    def __init__(
        self,
        *,
        transcript_buffer: TranscriptBuffer,
        input_manager: AudioInputManager,
        asr_manager: ASRManager,
        translator_manager: TranslatorManager,
        tts_manager: TTSManager,
        state_manager: StateManager,
        on_error: Callable[[str | ErrorEvent], None] | None = None,
        on_asr_event: Callable[[ASREventWithSource], None] | None = None,
        on_translation_event: Callable[[object], None] | None = None,
        on_tts_request: Callable[[str, str], None] | None = None,
        on_diagnostic_event: Callable[[str], None] | None = None,
    ) -> None:
        self._transcript_buffer = transcript_buffer
        self._input_manager = input_manager
        self._asr_manager = asr_manager
        self._translator_manager = translator_manager
        self._tts_manager = tts_manager
        self._state = state_manager
        self._on_error = on_error
        self._on_asr_stage_event = on_asr_event
        self._on_translation_event = on_translation_event
        self._on_tts_request = on_tts_request
        self._on_diagnostic_event = on_diagnostic_event
        self._active_sources: set[str] = set()

    @property
    def running(self) -> bool:
        return self._state.snapshot().running

    def start(self, mode: str, routes: AudioRouteConfig, sample_rate: int, chunk_ms: int = 100) -> None:
        self.stop()
        self._state.start_session()
        self._tts_manager.start()
        self._active_sources = set()

        if mode in ("meeting_to_local", "bidirectional"):
            self._active_sources.add("remote")
            self._asr_manager.set_enabled("remote", True)
            self._asr_manager.start("remote", self._on_asr_event)
            self._input_manager.add_consumer("remote", self._on_remote_audio_chunk)
            self._input_manager.start("remote", routes.meeting_in, sample_rate=sample_rate, chunk_ms=chunk_ms)

        if mode in ("local_to_meeting", "bidirectional"):
            self._active_sources.add("local")
            self._asr_manager.set_enabled("local", True)
            self._asr_manager.start("local", self._on_asr_event)
            self._input_manager.add_consumer("local", self._on_local_audio_chunk)
            self._input_manager.start("local", routes.microphone_in, sample_rate=sample_rate, chunk_ms=chunk_ms)

        if not self._active_sources:
            raise ValueError(f"Unsupported mode: {mode}")

    def stop(self) -> None:
        for source, consumer in (("remote", self._on_remote_audio_chunk), ("local", self._on_local_audio_chunk)):
            try:
                self._input_manager.remove_consumer(source, consumer)
            except Exception:
                pass
        self._input_manager.stop_all()
        self._asr_manager.stop_all()
        self._tts_manager.stop()
        self._active_sources.clear()
        self._state.stop_session()

    def stats(self) -> RouterStats:
        capture_stats = self._input_manager.stats()
        capture = {
            source: {
                "running": item.running,
                "sample_rate": item.sample_rate,
                "frame_count": item.frame_count,
                "level": item.level,
                "last_error": item.last_error,
            }
            for source, item in capture_stats.items()
        }
        state = self._state.snapshot()
        return RouterStats(
            running=state.running,
            active_sources=sorted(self._active_sources),
            state={
                "local_asr_enabled": state.local_asr_enabled,
                "remote_asr_enabled": state.remote_asr_enabled,
                "local_tts_busy": state.local_tts_busy,
                "remote_tts_busy": state.remote_tts_busy,
                "local_resume_in_ms": state.local_resume_in_ms,
                "remote_resume_in_ms": state.remote_resume_in_ms,
            },
            capture=capture,
            asr=self._asr_manager.stats(),
            tts=self._tts_manager.stats(),
        )

    def _on_local_audio_chunk(self, chunk: np.ndarray, sample_rate: float) -> None:
        self._state.tick()
        if not self._state.can_accept_asr("local"):
            return
        self._asr_manager.submit("local", chunk, sample_rate)

    def _on_remote_audio_chunk(self, chunk: np.ndarray, sample_rate: float) -> None:
        self._state.tick()
        if not self._state.can_accept_asr("remote"):
            return
        self._asr_manager.submit("remote", chunk, sample_rate)

    def _on_asr_event(self, event: ASREventWithSource) -> None:
        try:
            max_latency_ms = max(500, int(getattr(self._asr_manager._config.runtime, "max_pipeline_latency_ms", 3000)))
            if int(event.latency_ms) > max_latency_ms:
                self._emit_diagnostic_event(
                    f"drop_over_latency source={event.source} utterance_id={event.utterance_id} latency_ms={event.latency_ms}"
                )
                return
            self._emit_asr_event(event)
            original_channel = self._translator_manager.original_channel_of(event.source)
            self._transcript_buffer.upsert_event(
                source=original_channel,
                channel=original_channel,
                kind="original",
                text=event.text,
                is_final=event.is_final,
                utterance_id=event.utterance_id,
                revision=event.revision,
                latency_ms=event.latency_ms,
                created_at=datetime.fromtimestamp(event.created_at),
            )
            translated = self._translator_manager.process(event)
            if not translated:
                self._emit_diagnostic_event(
                    f"translation_skipped source={event.source} utterance_id={event.utterance_id} revision={event.revision}"
                )
                return
            self._emit_translation_event(translated)
            self._transcript_buffer.upsert_event(
                source=translated.translated_channel,
                channel=translated.translated_channel,
                kind="translated",
                text=translated.text,
                is_final=translated.is_final,
                utterance_id=translated.utterance_id,
                revision=translated.revision,
                latency_ms=event.latency_ms,
                created_at=datetime.fromtimestamp(translated.created_at),
            )
            if translated.should_speak:
                self._emit_tts_request(translated.tts_channel, translated.speak_text)
                self._tts_manager.enqueue(
                    translated.tts_channel,
                    translated.speak_text,
                    utterance_id=translated.utterance_id,
                    revision=translated.revision,
                    is_final=translated.is_final,
                )
        except Exception as exc:
            if self._on_error:
                self._on_error(
                    ErrorEvent(
                        level="error",
                        module="audio_router",
                        source=event.source,
                        code="asr_event_failed",
                        message="Failed to process ASR event",
                        detail=str(exc),
                    )
                )

    def _emit_asr_event(self, event: ASREventWithSource) -> None:
        if self._on_asr_stage_event:
            self._on_asr_stage_event(event)

    def _emit_translation_event(self, event: object) -> None:
        if self._on_translation_event:
            self._on_translation_event(event)

    def _emit_tts_request(self, channel: str, text: str) -> None:
        if self._on_tts_request:
            self._on_tts_request(channel, text)

    def _emit_diagnostic_event(self, message: str) -> None:
        if self._on_diagnostic_event:
            self._on_diagnostic_event(message)

    def handle_tts_play_start(self, channel: str) -> None:
        self._state.on_tts_start(channel)

    def handle_tts_play_end(self, channel: str) -> None:
        self._state.on_tts_end(channel)
