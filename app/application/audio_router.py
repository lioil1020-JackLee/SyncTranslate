from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Callable

import numpy as np

from app.domain.events import ErrorEvent
from app.infra.audio.routing import AudioInputManager
from app.infra.asr.streaming_pipeline import ASREventWithSource, ASRManager
from app.infra.config.schema import AppConfig, AudioRouteConfig
from app.domain.runtime_state import StateManager
from app.domain.constants import (
    DISPLAY_PARTIAL_ALL,
    DISPLAY_PARTIAL_NONE,
    DISPLAY_PARTIAL_STABLE_ONLY,
    OUTPUT_MODE_TTS,
)
from app.application.transcript_service import TranscriptBuffer
from app.infra.translation.engine import TranslatorManager
from app.infra.tts.playback_queue import TTSManager


@dataclass(slots=True)
class RouterStats:
    running: bool
    active_sources: list[str]
    state: dict[str, object]
    capture: dict[str, dict[str, object]]
    asr: dict[str, dict[str, object]]
    tts: dict[str, object]
    latency: list[dict[str, object]]
    translation_overflow: dict[str, int]


@dataclass(slots=True)
class _PartialDisplayState:
    utterance_id: str = ""
    last_text: str = ""
    repeat_count: int = 0
    last_displayed_text: str = ""


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
        async_translation: bool = False,
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
        self._mode: str = ""
        self._routes: AudioRouteConfig | None = None
        self._sample_rate: int = 0
        self._chunk_ms: int = 100
        self._capture_running: dict[str, bool] = {"local": False, "remote": False}
        self._asr_running: dict[str, bool] = {"local": False, "remote": False}
        self._async_translation = bool(async_translation)
        self._translation_workers: dict[str, Thread] = {}
        self._translation_stop_event = Event()
        self._runtime_config: AppConfig | None = None
        self._translation_overflow_counters: dict[str, int] = {"local": 0, "remote": 0}
        self._translation_queues = {
            "local": Queue(maxsize=self._translation_queue_maxsize_for_source("local")),
            "remote": Queue(maxsize=self._translation_queue_maxsize_for_source("remote")),
        }
        self._partial_display_state: dict[str, _PartialDisplayState] = {}
        self._latency_by_utterance: dict[tuple[str, str], dict[str, object]] = {}
        self._recent_latency: deque[dict[str, object]] = deque(maxlen=32)

    @property
    def running(self) -> bool:
        return self._state.snapshot().running

    def start(self, mode: str, routes: AudioRouteConfig, sample_rate: int, chunk_ms: int = 100) -> None:
        self.stop()
        self._state.start_session()
        self._tts_manager.start()
        self._mode = "bidirectional"
        self._routes = routes
        self._sample_rate = int(sample_rate)
        self._chunk_ms = int(chunk_ms)
        desired = self._desired_source_state()
        self._active_sources = {source for source, state in desired.items() if state["capture"] or state["asr"]}

        if not self._active_sources and not self._has_passthrough_enabled():
            raise ValueError("No active sources configured")

        self._start_translation_workers()
        self._reconcile_runtime_sources()

    def stop(self) -> None:
        for source, consumer in (("remote", self._on_remote_audio_chunk), ("local", self._on_local_audio_chunk)):
            try:
                self._input_manager.remove_consumer(source, consumer)
            except Exception:
                pass
        self._input_manager.stop_all()
        self._asr_manager.stop_all()
        self._stop_translation_workers()
        self._tts_manager.stop()
        self._active_sources.clear()
        self._mode = ""
        self._routes = None
        self._sample_rate = 0
        self._chunk_ms = 100
        self._capture_running = {"local": False, "remote": False}
        self._asr_running = {"local": False, "remote": False}
        self._partial_display_state.clear()
        self._latency_by_utterance.clear()
        self._recent_latency.clear()
        self._translation_overflow_counters = {"local": 0, "remote": 0}
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
            latency=list(self._recent_latency),
            translation_overflow=dict(self._translation_overflow_counters),
        )

    def _on_local_audio_chunk(self, chunk: np.ndarray, sample_rate: float) -> None:
        self._handle_source_audio_chunk(
            source="local",
            passthrough_channel="remote",
            chunk=chunk,
            sample_rate=sample_rate,
        )

    def _on_remote_audio_chunk(self, chunk: np.ndarray, sample_rate: float) -> None:
        self._handle_source_audio_chunk(
            source="remote",
            passthrough_channel="local",
            chunk=chunk,
            sample_rate=sample_rate,
        )

    def _handle_source_audio_chunk(
        self,
        *,
        source: str,
        passthrough_channel: str,
        chunk: np.ndarray,
        sample_rate: float,
    ) -> None:
        self._state.tick()
        # Submit audio to ASR before any optional passthrough/output work so the
        # recognition path is not delayed by playback-device overhead.
        if self._state.can_accept_asr(source):
            self._asr_manager.submit(source, chunk, sample_rate)
        # Keep local/remote passthrough handling fully symmetric: the source-specific
        # difference is only which output channel receives the live audio.
        self._tts_manager.submit_passthrough(passthrough_channel, chunk, sample_rate)

    def _on_asr_event(self, event: ASREventWithSource) -> None:
        try:
            if self._should_drop_over_latency(event):
                return
            self._emit_asr_event(event)
            self._record_asr_latency(event)
            original_channel, translated_channel, tts_channel = self._channels_of(event.source)
            self._maybe_store_transcript(
                source=original_channel,
                channel=original_channel,
                kind="original",
                text=event.text,
                is_final=event.is_final,
                is_stable_partial=not event.is_final,
                utterance_id=event.utterance_id,
                revision=event.revision,
                latency_ms=event.latency_ms,
                created_at=datetime.fromtimestamp(event.created_at),
                speaker_label=getattr(event, "speaker_label", ""),
            )
            translation_checker = getattr(self._translator_manager, "translation_enabled", lambda *_args: True)
            try:
                translation_enabled = bool(translation_checker(event.source))
            except TypeError:
                translation_enabled = bool(translation_checker())
            if not translation_enabled:
                correct_event = getattr(self._translator_manager, "correct_asr_event", lambda value: value)
                corrected_event = correct_event(event) if event.is_final else event
                self._maybe_store_transcript(
                    source=translated_channel,
                    channel=translated_channel,
                    kind="translated",
                    text=corrected_event.text,
                    is_final=corrected_event.is_final,
                    is_stable_partial=not corrected_event.is_final,
                    utterance_id=corrected_event.utterance_id,
                    revision=corrected_event.revision,
                    latency_ms=corrected_event.latency_ms,
                    created_at=datetime.fromtimestamp(corrected_event.created_at),
                    speaker_label=getattr(corrected_event, "speaker_label", ""),
                )
                should_speak = corrected_event.is_final or bool(getattr(corrected_event, "is_early_final", False))
                if should_speak and self._tts_manager.output_mode(tts_channel) == OUTPUT_MODE_TTS:
                    speak_text = corrected_event.text.strip()
                    if speak_text:
                        self._emit_tts_request(tts_channel, speak_text)
                        self._tts_manager.enqueue(
                            tts_channel,
                            speak_text,
                            utterance_id=corrected_event.utterance_id,
                            revision=corrected_event.revision,
                            is_final=corrected_event.is_final,
                            is_stable_partial=not corrected_event.is_final,
                            is_early_final=bool(getattr(corrected_event, "is_early_final", False)),
                        )
                return
            if self._async_translation:
                self._enqueue_translation_event(event)
                return
            self._process_translation_event(event)
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

    def _process_translation_event(self, event: ASREventWithSource) -> None:
        correct_event = getattr(self._translator_manager, "correct_asr_event", lambda value: value)
        corrected_event = correct_event(event)
        translated = self._translator_manager.process(corrected_event)
        if not translated:
            skip_reason = ""
            try:
                skip_reason = str(getattr(self._translator_manager, "last_skip_reason", lambda _source: "")(event.source) or "")
            except Exception:
                skip_reason = ""
            self._emit_diagnostic_event(
                "translation_skipped "
                f"source={corrected_event.source} utterance_id={corrected_event.utterance_id} revision={corrected_event.revision}"
                + (f" reason={skip_reason}" if skip_reason else "")
            )
            return
        self._emit_translation_event(translated)
        self._record_translation_latency(corrected_event, translated)
        self._maybe_store_transcript(
            source=translated.translated_channel,
            channel=translated.translated_channel,
            kind="translated",
            text=translated.text,
            is_final=translated.is_final,
            is_stable_partial=translated.is_stable_partial,
            utterance_id=translated.utterance_id,
            revision=translated.revision,
            latency_ms=event.latency_ms,
            created_at=datetime.fromtimestamp(translated.created_at),
            speaker_label=getattr(translated, "speaker_label", ""),
        )
        if translated.should_speak and self._tts_manager.output_mode(translated.tts_channel) == OUTPUT_MODE_TTS:
            self._emit_tts_request(translated.tts_channel, translated.speak_text)
            self._record_tts_enqueue_latency(
                channel=translated.tts_channel,
                source=translated.source,
                utterance_id=translated.utterance_id,
                revision=translated.revision,
                is_final=translated.is_final,
                is_stable_partial=translated.is_stable_partial,
                is_early_final=translated.is_early_final,
            )
            self._tts_manager.enqueue(
                translated.tts_channel,
                translated.speak_text,
                utterance_id=translated.utterance_id,
                revision=translated.revision,
                is_final=translated.is_final,
                is_stable_partial=translated.is_stable_partial,
                is_early_final=translated.is_early_final,
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
        self._record_playback_start_latency(channel)
        self._state.on_tts_start(channel)

    def handle_tts_play_end(self, channel: str) -> None:
        self._state.on_tts_end(channel)

    def set_tts_muted(self, channel: str, muted: bool) -> None:
        self._tts_manager.set_muted(channel, muted)

    def set_passthrough_enabled(self, channel: str, enabled: bool) -> None:
        self._tts_manager.set_passthrough_enabled(channel, enabled)
        self._reconcile_runtime_sources()

    def set_output_mode(self, channel: str, mode: str) -> None:
        self._tts_manager.set_output_mode(channel, mode)
        self._reconcile_runtime_sources()

    def refresh_runtime_config(self, config: AppConfig) -> None:
        self._runtime_config = config
        refresh_runtime = getattr(self._asr_manager, "refresh_runtime", None)
        if callable(refresh_runtime):
            refresh_runtime()
        else:
            self._asr_manager.configure_pipeline(
                config,
                getattr(self._asr_manager, "_pipeline_revision", 1),
            )

    def _reconcile_runtime_sources(self) -> None:
        if not self.running or self._routes is None or self._sample_rate <= 0:
            return
        desired = self._desired_source_state()
        for source in ("local", "remote"):
            capture_needed = bool(desired[source]["capture"])
            asr_needed = bool(desired[source]["asr"])

            self._asr_manager.set_enabled(source, asr_needed)
            asr_running = self._asr_running.get(source, False)
            if asr_needed and not asr_running:
                self._asr_manager.start(source, self._on_asr_event)
                self._asr_running[source] = True
            elif (not asr_needed) and asr_running:
                self._asr_manager.stop(source)
                self._asr_running[source] = False

            running = self._capture_running.get(source, False)
            if capture_needed and not running:
                self._input_manager.add_consumer(source, self._consumer_of(source))
                self._input_manager.start(
                    source,
                    self._device_of(source),
                    sample_rate=self._sample_rate,
                    chunk_ms=self._chunk_ms,
                )
                self._capture_running[source] = True
            elif not capture_needed and running:
                self._input_manager.remove_consumer(source, self._consumer_of(source))
                self._input_manager.stop(source)
                self._capture_running[source] = False

    def _desired_source_state(self) -> dict[str, dict[str, bool]]:
        return {
            "local": {
                "asr": self._asr_needed_for_source("local"),
                "capture": self._asr_needed_for_source("local") or self._tts_manager.is_passthrough_enabled("remote"),
            },
            "remote": {
                "asr": self._asr_needed_for_source("remote"),
                "capture": self._asr_needed_for_source("remote") or self._tts_manager.is_passthrough_enabled("local"),
            },
        }

    def _asr_needed_for_source(self, source: str) -> bool:
        runtime = getattr(self._runtime_config, "runtime", None)
        if runtime is None:
            return True
        attr = "remote_asr_language" if source == "remote" else "local_asr_language"
        return str(getattr(runtime, attr, "auto") or "auto").strip().lower() != "none"

    def _has_passthrough_enabled(self) -> bool:
        return self._tts_manager.is_passthrough_enabled("local") or self._tts_manager.is_passthrough_enabled("remote")

    @staticmethod
    def _channels_of(source: str) -> tuple[str, str, str]:
        if source == "remote":
            return ("meeting_original", "meeting_translated", "local")
        return ("local_original", "local_translated", "remote")

    def _device_of(self, source: str) -> str:
        if self._routes is None:
            return ""
        if source == "remote":
            return self._routes.meeting_in
        return self._routes.microphone_in

    def _consumer_of(self, source: str):
        if source == "remote":
            return self._on_remote_audio_chunk
        return self._on_local_audio_chunk

    def _translation_queue_maxsize_for_source(self, source: str) -> int:
        runtime = getattr(self._runtime_config, "runtime", None)
        if source == "remote":
            value = int(getattr(runtime, "llm_queue_maxsize_remote", 32)) if runtime is not None else 32
        else:
            value = int(getattr(runtime, "llm_queue_maxsize_local", 32)) if runtime is not None else 32
        return max(4, value)

    def _maybe_store_transcript(
        self,
        *,
        source: str,
        channel: str,
        kind: str,
        text: str,
        is_final: bool,
        is_stable_partial: bool,
        utterance_id: str | None,
        revision: int,
        latency_ms: int | None,
        created_at: datetime,
        speaker_label: str = "",
    ) -> None:
        should_display, is_stable_partial = self._should_display_partial(
            channel=channel,
            utterance_id=utterance_id,
            text=text,
            is_final=is_final,
        )
        if not should_display:
            return
        self._transcript_buffer.upsert_event(
            source=source,
            channel=channel,
            kind=kind,
            text=text,
            is_final=is_final,
            utterance_id=utterance_id,
            revision=revision,
            is_stable_partial=is_stable_partial and not is_final,
            latency_ms=latency_ms,
            created_at=created_at,
            speaker_label=speaker_label,
        )

    def _should_display_partial(
        self,
        *,
        channel: str,
        utterance_id: str | None,
        text: str,
        is_final: bool,
    ) -> tuple[bool, bool]:
        if is_final:
            self._partial_display_state.pop(channel, None)
            return True, False
        strategy = self._display_partial_strategy()
        if strategy == DISPLAY_PARTIAL_ALL:
            return True, False
        if strategy == DISPLAY_PARTIAL_NONE:
            return False, False
        normalized = text.strip()
        if not normalized or not utterance_id:
            return False, False
        state = self._partial_display_state.get(channel)
        if state is None or state.utterance_id != utterance_id:
            self._partial_display_state[channel] = _PartialDisplayState(
                utterance_id=utterance_id,
                last_text=normalized,
                repeat_count=1,
            )
            return False, False
        if self._is_stable_partial_progression(state.last_text, normalized):
            state.repeat_count += 1
        else:
            state.repeat_count = 1
        state.last_text = normalized
        is_stable = state.repeat_count >= self._stable_partial_min_repeats()
        if not is_stable:
            return False, False
        if normalized == state.last_displayed_text:
            return False, True
        state.last_displayed_text = normalized
        return True, True

    def _display_partial_strategy(self) -> str:
        runtime = getattr(self._runtime_config, "runtime", None)
        value = str(getattr(runtime, "display_partial_strategy", DISPLAY_PARTIAL_STABLE_ONLY) or DISPLAY_PARTIAL_STABLE_ONLY).strip().lower()
        if value in {DISPLAY_PARTIAL_ALL, DISPLAY_PARTIAL_NONE, DISPLAY_PARTIAL_STABLE_ONLY}:
            return value
        return DISPLAY_PARTIAL_STABLE_ONLY

    def _stable_partial_min_repeats(self) -> int:
        runtime = getattr(self._runtime_config, "runtime", None)
        value = int(getattr(runtime, "stable_partial_min_repeats", 2)) if runtime is not None else 2
        return max(1, value)

    def _partial_stability_max_delta_chars(self) -> int:
        runtime = getattr(self._runtime_config, "runtime", None)
        value = int(getattr(runtime, "partial_stability_max_delta_chars", 8)) if runtime is not None else 8
        return max(1, value)

    def _is_stable_partial_progression(self, previous: str, current: str) -> bool:
        if previous == current:
            return True
        delta_limit = self._partial_stability_max_delta_chars()
        if current.startswith(previous):
            return (len(current) - len(previous)) <= delta_limit
        if previous.startswith(current):
            return (len(previous) - len(current)) <= delta_limit
        common_prefix_len = 0
        for prev_char, curr_char in zip(previous, current):
            if prev_char != curr_char:
                break
            common_prefix_len += 1
        if common_prefix_len <= 0:
            return False
        previous_tail = len(previous) - common_prefix_len
        current_tail = len(current) - common_prefix_len
        return previous_tail <= delta_limit and current_tail <= delta_limit

    def _start_translation_workers(self) -> None:
        if not self._async_translation:
            return
        self._translation_stop_event.clear()
        for source in ("local", "remote"):
            worker = self._translation_workers.get(source)
            if worker and worker.is_alive():
                continue
            thread = Thread(target=self._run_translation_worker, args=(source,), daemon=True)
            self._translation_workers[source] = thread
            thread.start()

    def _stop_translation_workers(self) -> None:
        self._translation_stop_event.set()
        for source, queue in self._translation_queues.items():
            while not queue.empty():
                try:
                    queue.get_nowait()
                except Empty:
                    break
            worker = self._translation_workers.get(source)
            if worker and worker.is_alive():
                worker.join(timeout=2.0)
        self._translation_workers.clear()
        self._translation_stop_event = Event()

    def _enqueue_translation_event(self, event: ASREventWithSource) -> None:
        key = event.source if event.source in ("local", "remote") else "local"
        queue = self._translation_queues[key]
        dropped = self._put_drop_oldest(queue, event)
        if dropped:
            self._translation_overflow_counters[key] = self._translation_overflow_counters.get(key, 0) + 1
            self._emit_diagnostic_event(
                f"translation_queue_overflow source={key} utterance_id={event.utterance_id} revision={event.revision}"
                f" total_overflow={self._translation_overflow_counters[key]}"
            )

    def _put_drop_oldest(self, queue: Queue, item: object) -> bool:
        """Put item into queue, dropping the oldest entry if full. Returns True if a drop occurred."""
        try:
            queue.put_nowait(item)
            return False
        except Full:
            pass
        dropped = False
        try:
            queue.get_nowait()
            dropped = True
        except Empty:
            pass
        try:
            queue.put_nowait(item)
        except Full:
            pass
        return dropped

    def _should_drop_over_latency(self, event: ASREventWithSource) -> bool:
        """Return True (and emit diagnostic) if event latency exceeds max_pipeline_latency_ms."""
        runtime_config = getattr(self._runtime_config, "runtime", None)
        max_latency_ms = max(500, int(getattr(runtime_config, "max_pipeline_latency_ms", 3000)))
        if int(event.latency_ms) > max_latency_ms:
            self._emit_diagnostic_event(
                f"drop_over_latency source={event.source} utterance_id={event.utterance_id}"
                f" latency_ms={event.latency_ms} max_ms={max_latency_ms}"
            )
            return True
        return False

    def _run_translation_worker(self, source: str) -> None:
        queue = self._translation_queues[source]
        while not self._translation_stop_event.is_set():
            try:
                event = queue.get(timeout=0.2)
            except Empty:
                continue
            try:
                self._process_translation_event(event)
            except Exception as exc:
                if self._on_error:
                    self._on_error(
                        ErrorEvent(
                            level="error",
                            module="audio_router",
                            source=source,
                            code="translation_worker_failed",
                            message="Failed to process translation event",
                            detail=str(exc),
                        )
                    )

    def _record_asr_latency(self, event: ASREventWithSource) -> None:
        key = (event.source, event.utterance_id)
        entry = self._latency_by_utterance.setdefault(
            key,
            {
                "source": event.source,
                "utterance_id": event.utterance_id,
                "revision": event.revision,
                "speech_start_ms": event.start_ms,
            },
        )
        entry["revision"] = event.revision
        entry["speech_start_ms"] = min(int(entry.get("speech_start_ms", event.start_ms)), int(event.start_ms))
        if not event.is_final and "first_asr_partial_ms" not in entry:
            entry["first_asr_partial_ms"] = max(0, int(event.end_ms) - int(event.start_ms))
        if event.is_final:
            entry["speech_end_to_asr_final_ms"] = max(0, int(event.latency_ms))
            entry["asr_final_kind"] = "early_final" if event.is_early_final else "final"

    def _record_translation_latency(self, event: ASREventWithSource, translated: object) -> None:
        key = (event.source, event.utterance_id)
        entry = self._latency_by_utterance.setdefault(key, {"source": event.source, "utterance_id": event.utterance_id})
        now_ms = int(datetime.now().timestamp() * 1000)
        entry["translation_ready_at_ms"] = now_ms
        if not event.is_final and "first_display_partial_ms" not in entry:
            entry["first_display_partial_ms"] = max(0, int(event.end_ms) - int(event.start_ms))
        if getattr(translated, "is_final", False):
            entry["asr_final_to_llm_final_ms"] = max(0, now_ms - int(event.created_at * 1000))

    def _record_tts_enqueue_latency(
        self,
        *,
        channel: str,
        source: str,
        utterance_id: str,
        revision: int,
        is_final: bool,
        is_stable_partial: bool,
        is_early_final: bool,
    ) -> None:
        key = (source, utterance_id)
        entry = self._latency_by_utterance.setdefault(key, {"source": source, "utterance_id": utterance_id})
        now_ms = int(datetime.now().timestamp() * 1000)
        entry["tts_channel"] = channel
        entry["tts_enqueue_at_ms"] = now_ms
        entry["tts_enqueue_revision"] = revision
        entry["tts_enqueue_kind"] = (
            "final" if is_final else "early_final" if is_early_final else "stable_partial" if is_stable_partial else "partial"
        )

    def _record_playback_start_latency(self, channel: str) -> None:
        current = getattr(self._tts_manager, "current_task", lambda _channel: None)(channel)
        if not current:
            return
        source = "remote" if channel == "local" else "local"
        utterance_id = str(current.get("utterance_id") or "")
        if not utterance_id:
            return
        key = (source, utterance_id)
        entry = self._latency_by_utterance.setdefault(key, {"source": source, "utterance_id": utterance_id})
        now_ms = int(datetime.now().timestamp() * 1000)
        entry["playback_start_at_ms"] = now_ms
        enqueue_at = int(entry.get("tts_enqueue_at_ms", now_ms))
        entry["tts_enqueue_to_playback_start_ms"] = max(0, now_ms - enqueue_at)
        self._recent_latency.appendleft(
            {
                "source": entry.get("source", source),
                "utterance_id": entry.get("utterance_id", utterance_id),
                "first_asr_partial_ms": entry.get("first_asr_partial_ms"),
                "first_display_partial_ms": entry.get("first_display_partial_ms"),
                "speech_end_to_asr_final_ms": entry.get("speech_end_to_asr_final_ms"),
                "asr_final_to_llm_final_ms": entry.get("asr_final_to_llm_final_ms"),
                "tts_enqueue_to_playback_start_ms": entry.get("tts_enqueue_to_playback_start_ms"),
                "tts_enqueue_kind": entry.get("tts_enqueue_kind", "unknown"),
            }
        )
