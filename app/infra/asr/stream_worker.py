"""Legacy segment worker for the Whisper-based ASR path.

This worker remains as the frozen compatibility path while the new ASR v2 stack
is introduced. Avoid adding new product features here unless they are required
to keep the existing pipeline functioning during migration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from collections import deque
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import Callable
import re
import unicodedata

import numpy as np

from app.infra.asr.faster_whisper_adapter import FasterWhisperEngine
from app.infra.asr.language_policy import VadSegmenter
from app.infra.asr.speaker_diarizer import OnlineSpeakerDiarizer


@dataclass(slots=True)
class AsrEvent:
    text: str
    is_final: bool
    is_early_final: bool
    start_ms: int
    end_ms: int
    latency_ms: int
    detected_language: str = ""
    speaker_label: str = ""


@dataclass(slots=True)
class StreamingAsrStats:
    queue_size: int
    dropped_chunks: int
    partial_count: int
    final_count: int
    last_debug: str
    vad_rms: float
    vad_threshold: float
    adaptive_mode: str
    adaptive_partial_interval_ms: int
    adaptive_min_silence_duration_ms: int
    adaptive_soft_final_audio_ms: int


class StreamingAsr:
    def __init__(
        self,
        *,
        engine: FasterWhisperEngine,
        vad: VadSegmenter,
        partial_interval_ms: int = 800,
        partial_history_seconds: int = 3,
        final_history_seconds: int = 6,
        soft_final_audio_ms: int = 4200,
        pre_roll_ms: int = 500,
        min_partial_audio_ms: int = 280,
        partial_interval_floor_ms: int = 280,
        early_final_enabled: bool = True,
        adaptive_enabled: bool = True,
        queue_maxsize: int = 128,
        speaker_diarizer: OnlineSpeakerDiarizer | None = None,
        on_event: Callable[[AsrEvent], None] | None = None,
        on_debug: Callable[[str], None] | None = None,
    ) -> None:
        self._engine = engine
        self._vad = vad
        # Keep partial decode conservative to avoid native decoder overload.
        self._partial_interval_floor_ms = max(120, int(partial_interval_floor_ms))
        self._base_partial_interval_ms = max(self._partial_interval_floor_ms, int(partial_interval_ms))
        self._partial_interval_ms = self._base_partial_interval_ms
        self._partial_history_seconds = max(1, int(partial_history_seconds))
        self._final_history_seconds = max(1, int(final_history_seconds))
        self._pre_roll_ms = max(0, int(pre_roll_ms))
        self._queue: Queue[tuple[np.ndarray, float]] = Queue(maxsize=max(4, queue_maxsize))
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._on_event = on_event
        self._on_debug = on_debug
        self._segment_chunks: list[np.ndarray] = []
        self._segment_start_ms = 0
        self._segment_end_ms = 0
        self._last_partial_ms = 0
        self._segment_sample_rate = 16000
        self._min_partial_audio_ms = max(120, int(min_partial_audio_ms))
        self._force_final_queue_size = max(8, self._queue.maxsize // 4)
        self._force_final_audio_ms = 1800
        self._base_soft_final_audio_ms = max(self._force_final_audio_ms, int(soft_final_audio_ms))
        self._soft_final_audio_ms = self._base_soft_final_audio_ms
        self._early_final_enabled = bool(early_final_enabled)
        self._adaptive_enabled = bool(adaptive_enabled)
        self._speaker_diarizer = speaker_diarizer
        self._base_min_silence_duration_ms = int(
            max(120.0, float(getattr(vad, "effective_min_silence_duration_ms", 240.0)))
        )
        self._adaptive_mode = "baseline"
        self._adaptive_recent_audio_ms: deque[int] = deque(maxlen=10)
        self._adaptive_partial_latencies: deque[int] = deque(maxlen=10)
        self._adaptive_final_latencies: deque[int] = deque(maxlen=10)
        self._adaptive_load_backoff_until_ms = 0
        self._final_error_cooldown_until_ms = 0
        self._last_final_error_text = ""
        self._overflow_count = 0
        self._last_overflow_report_ms = 0
        self._drop_partial_until_final = False
        self._partial_cooldown_until_ms = 0
        self._pre_roll_chunks: deque[np.ndarray] = deque()
        self._pre_roll_sample_count = 0
        self._pre_roll_sample_rate = 16000
        self._in_speech_segment = False
        self._soft_split_pause_ms = 220.0
        self._partial_count = 0
        self._final_count = 0
        self._last_debug = ""
        self._stats_lock = Lock()
        self._apply_adaptive_tuning(
            mode="baseline",
            partial_interval_ms=self._base_partial_interval_ms,
            min_silence_duration_ms=self._base_min_silence_duration_ms,
            soft_final_audio_ms=self._base_soft_final_audio_ms,
            force=True,
        )

    def start(self, on_event: Callable[[AsrEvent], None]) -> None:
        self.stop()
        self._on_event = on_event
        self._stop_event.clear()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def request_stop(self) -> None:
        self._stop_event.set()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def cleanup_if_stopped(self) -> bool:
        if self._thread and self._thread.is_alive():
            return False
        self._thread = None
        self._vad.reset()
        self._segment_chunks = []
        self._segment_start_ms = 0
        self._segment_end_ms = 0
        self._last_partial_ms = 0
        self._segment_sample_rate = 16000
        self._overflow_count = 0
        self._last_overflow_report_ms = 0
        self._drop_partial_until_final = False
        self._partial_cooldown_until_ms = 0
        self._pre_roll_chunks.clear()
        self._pre_roll_sample_count = 0
        self._pre_roll_sample_rate = 16000
        self._in_speech_segment = False
        self._partial_count = 0
        self._final_count = 0
        self._last_debug = ""
        self._adaptive_recent_audio_ms.clear()
        self._adaptive_partial_latencies.clear()
        self._adaptive_final_latencies.clear()
        self._adaptive_load_backoff_until_ms = 0
        self._final_error_cooldown_until_ms = 0
        self._last_final_error_text = ""
        if self._speaker_diarizer is not None:
            self._speaker_diarizer.reset()
        self._apply_adaptive_tuning(
            mode="baseline",
            partial_interval_ms=self._base_partial_interval_ms,
            min_silence_duration_ms=self._base_min_silence_duration_ms,
            soft_final_audio_ms=self._base_soft_final_audio_ms,
            force=True,
        )
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Empty:
                break
        return True

    def stop(self) -> None:
        self.request_stop()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=8.0)
            if self._thread.is_alive():
                # Do not clear/reset the shared stop event if old worker is still alive,
                # otherwise a subsequent start() can revive the previous worker thread.
                self._debug("asr worker stop timeout; skip reset to avoid concurrent workers")
                raise RuntimeError("ASR worker did not stop in time")
        self.cleanup_if_stopped()

    def stats(self) -> StreamingAsrStats:
        with self._stats_lock:
            return StreamingAsrStats(
                queue_size=self._queue.qsize(),
                dropped_chunks=self._overflow_count,
                partial_count=self._partial_count,
                final_count=self._final_count,
                last_debug=self._last_debug,
                vad_rms=self._vad.last_rms,
                vad_threshold=self._vad.effective_rms_threshold,
                adaptive_mode=self._adaptive_mode,
                adaptive_partial_interval_ms=self._partial_interval_ms,
                adaptive_min_silence_duration_ms=int(
                    getattr(self._vad, "effective_min_silence_duration_ms", self._base_min_silence_duration_ms)
                ),
                adaptive_soft_final_audio_ms=self._soft_final_audio_ms,
            )

    def submit_chunk(self, chunk: np.ndarray, sample_rate: float) -> None:
        try:
            self._queue.put_nowait((chunk.copy(), sample_rate))
        except Full:
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            try:
                self._queue.put_nowait((chunk.copy(), sample_rate))
            except Full:
                pass
            self._overflow_count += 1
            now_ms = int(time.monotonic() * 1000)
            if now_ms - self._last_overflow_report_ms >= 1000:
                self._debug(f"asr queue overflow: dropped oldest chunk x{self._overflow_count}")
                self._last_overflow_report_ms = now_ms
            self._drop_partial_until_final = True
            # Cool down partial decoding after overflow to reduce native ASR load spikes.
            self._partial_cooldown_until_ms = max(self._partial_cooldown_until_ms, now_ms + 8000)
            self._adaptive_load_backoff_until_ms = max(self._adaptive_load_backoff_until_ms, now_ms + 12000)
            self._recompute_adaptive_tuning(now_ms=now_ms)
            self._debug("asr safety mode: partial decode cooling down after overflow")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                chunk, sample_rate = self._queue.get(timeout=0.2)
            except Empty:
                continue

            drained = 0
            pending_parts = [chunk]
            while drained < 3:
                try:
                    extra_chunk, extra_rate = self._queue.get_nowait()
                except Empty:
                    break
                if extra_rate > 0 and int(extra_rate) == int(sample_rate):
                    pending_parts.append(extra_chunk)
                    drained += 1
                    continue
                self._process_chunk(chunk=np.concatenate(pending_parts, axis=0), sample_rate=sample_rate)
                pending_parts = [extra_chunk]
                sample_rate = extra_rate
                drained = 0

            self._process_chunk(chunk=np.concatenate(pending_parts, axis=0), sample_rate=sample_rate)

    def _process_chunk(self, *, chunk: np.ndarray, sample_rate: float) -> None:
        if sample_rate <= 0:
            return
        now_ms = int(time.monotonic() * 1000)
        if not self._in_speech_segment:
            self._append_pre_roll_chunk(chunk=chunk, sample_rate=sample_rate)
        decision = self._vad.update(chunk, sample_rate)
        chunk_ms = int(decision.chunk_ms)
        self._segment_sample_rate = int(sample_rate)
        if self._segment_start_ms == 0:
            self._segment_start_ms = now_ms - chunk_ms
        self._segment_end_ms = now_ms

        if decision.speech_active:
            just_started = False
            if not self._in_speech_segment:
                just_started = self._prime_segment_from_pre_roll(now_ms=now_ms, sample_rate=sample_rate)
                self._in_speech_segment = True
            if not just_started:
                self._segment_chunks.append(chunk)
            segment_audio_ms = self._segment_audio_ms()
            should_force_final = (
                self._queue.qsize() >= self._force_final_queue_size
                and segment_audio_ms >= self._force_final_audio_ms
            )
            should_soft_split = self._should_emit_soft_split(decision)
            if should_force_final:
                self._debug(
                    "asr backlog: force final "
                    f"queue={self._queue.qsize()} audio_ms={segment_audio_ms}"
                )
                self._emit_final(now_ms=now_ms, is_early_final=False)
                self._reset_segment()
                return
            if should_soft_split:
                pause_ms = int(round(float(getattr(decision, "pause_ms", 0.0))))
                self._debug(f"asr long speech: soft final audio_ms={segment_audio_ms} pause_ms={pause_ms}")
                self._emit_final(now_ms=now_ms, is_early_final=False)
                self._reset_segment()
                return
            should_emit_partial = (
                (not self._drop_partial_until_final)
                and now_ms >= self._partial_cooldown_until_ms
                and self._queue.qsize() == 0
                and now_ms - self._last_partial_ms >= self._partial_interval_ms
            )
            if should_emit_partial:
                self._emit_partial()
                self._last_partial_ms = now_ms

        if decision.finalize:
            # Include the transition chunk (speech→silence boundary) so the very last
            # words are not silently dropped before final transcription.
            if (not decision.speech_active) and chunk.size > 0:
                self._segment_chunks.append(chunk)
            self._emit_final(now_ms=now_ms, is_early_final=False)
            self._reset_segment()
            return

        if self._should_emit_early_final(decision):
            self._debug(
                "asr early final "
                f"speech_ms={int(decision.speech_ms)} silence_ms={int(decision.silence_ms)}"
            )
            self._emit_final(now_ms=now_ms, is_early_final=True)
            self._reset_segment()

    def _emit_partial(self) -> None:
        if not self._on_event or not self._segment_chunks:
            return
        audio = np.concatenate(self._segment_chunks, axis=0)
        audio = self._limited_audio(audio, self._partial_history_seconds)
        audio_ms = int(len(audio) * 1000 / max(1, self._segment_sample_rate))
        if audio_ms < self._min_partial_audio_ms:
            return
        start = time.perf_counter()
        try:
            result = self._engine.transcribe_partial_result(audio=audio, sample_rate=self._segment_sample_rate)
            text = result.text or ""
        except Exception as exc:
            self._debug(f"partial asr failed: {exc}")
            now_ms = int(time.monotonic() * 1000)
            self._partial_cooldown_until_ms = max(self._partial_cooldown_until_ms, now_ms + 12000)
            self._debug("asr safety mode: partial decode cooling down after partial failure")
            return
        if not text.strip():
            return
        if self._stop_event.is_set():
            return
        drop_reason = _transcript_drop_reason(
            text,
            audio_ms=audio_ms,
            vad_rms=self._vad.last_rms,
            expected_language=str(getattr(self._engine, "language", "") or ""),
        )
        if drop_reason:
            self._debug(f"drop {drop_reason} partial text={text.strip()[:80]}")
            return
        latency_ms = int((time.perf_counter() - start) * 1000)
        if latency_ms >= 1500:
            now_ms = int(time.monotonic() * 1000)
            self._partial_cooldown_until_ms = max(self._partial_cooldown_until_ms, now_ms + 15000)
            self._debug("asr safety mode: partial decode cooling down after high partial latency")
        self._record_partial_adaptation(latency_ms=latency_ms)
        self._on_event(
            AsrEvent(
                text=text.strip(),
                is_final=False,
                is_early_final=False,
                start_ms=self._segment_start_ms,
                end_ms=self._segment_end_ms,
                latency_ms=latency_ms,
                detected_language=result.detected_language,
            )
        )
        with self._stats_lock:
            self._partial_count += 1
        self._debug(f"asr partial latency={latency_ms}ms text={text.strip()[:80]}")

    def _emit_final(self, *, now_ms: int, is_early_final: bool) -> None:
        if not self._on_event or not self._segment_chunks:
            return
        audio = np.concatenate(self._segment_chunks, axis=0)
        audio = self._limited_audio(audio, self._final_history_seconds)
        start = time.perf_counter()
        try:
            result = self._engine.transcribe_final_result(audio=audio, sample_rate=self._segment_sample_rate)
            text = result.text or ""
        except Exception as exc:
            if self._stop_event.is_set():
                return
            error_text = _format_asr_exception_message(exc)
            cooldown_ms = 15000
            if "_ssl" in error_text.lower() or "libssl" in error_text.lower() or "libcrypto" in error_text.lower():
                cooldown_ms = 45000
            should_emit = (
                now_ms >= self._final_error_cooldown_until_ms
                or error_text != self._last_final_error_text
            )
            self._final_error_cooldown_until_ms = max(self._final_error_cooldown_until_ms, now_ms + cooldown_ms)
            self._last_final_error_text = error_text
            if not should_emit:
                self._debug("asr final failed repeatedly; suppressing duplicate error during cooldown")
                return
            self._on_event(
                AsrEvent(
                    text=f"[asr-error] {error_text}",
                    is_final=True,
                    is_early_final=is_early_final,
                    start_ms=self._segment_start_ms,
                    end_ms=now_ms,
                    latency_ms=0,
                )
            )
            return
        audio_ms = int(len(audio) * 1000 / max(1, self._segment_sample_rate))
        drop_reason = _transcript_drop_reason(
            text,
            audio_ms=audio_ms,
            vad_rms=self._vad.last_rms,
            expected_language=str(getattr(self._engine, "language", "") or ""),
        )
        if drop_reason:
            self._debug(f"drop {drop_reason} final text={text.strip()[:80]}")
            return
        latency_ms = int((time.perf_counter() - start) * 1000)
        if latency_ms >= 1500:
            now_ms = int(time.monotonic() * 1000)
            self._partial_cooldown_until_ms = max(self._partial_cooldown_until_ms, now_ms + 6000)
            self._debug(f"asr partial backoff: latency={latency_ms}ms")
        self._record_final_adaptation(audio_ms=audio_ms, latency_ms=latency_ms, now_ms=now_ms)
        if not text.strip():
            return
        if self._stop_event.is_set():
            return
        speaker_label = ""
        if self._speaker_diarizer is not None:
            speaker_label = self._speaker_diarizer.assign(
                audio=audio,
                sample_rate=self._segment_sample_rate,
                now_ms=now_ms,
            )
        self._on_event(
            AsrEvent(
                text=text.strip(),
                is_final=True,
                is_early_final=is_early_final,
                start_ms=self._segment_start_ms,
                end_ms=now_ms,
                latency_ms=latency_ms,
                detected_language=result.detected_language,
                speaker_label=speaker_label,
            )
        )
        with self._stats_lock:
            self._final_count += 1
        self._debug(f"asr final latency={latency_ms}ms text={text.strip()[:80]}")

    def _should_emit_early_final(self, decision) -> bool:
        if not self._early_final_enabled:
            return False
        if decision.speech_active:
            return False
        if not self._segment_chunks:
            return False
        speech_ms = float(getattr(decision, "speech_ms", 0.0))
        silence_ms = float(getattr(decision, "silence_ms", 0.0))
        if speech_ms <= 0 or silence_ms <= 0:
            return False
        segment_audio_ms = self._segment_audio_ms()
        if segment_audio_ms < 700 or segment_audio_ms > 2200:
            return False
        if speech_ms < 520:
            return False
        configured_silence_ms = float(getattr(self._vad, "effective_min_silence_duration_ms", 240.0))
        early_silence_ms = max(320.0, min(420.0, configured_silence_ms * 0.75))
        return silence_ms >= early_silence_ms

    def _should_emit_soft_split(self, decision) -> bool:
        if self._queue.qsize() != 0 or not self._segment_chunks:
            return False
        segment_audio_ms = self._segment_audio_ms()
        if segment_audio_ms < self._soft_final_audio_ms:
            return False
        pause_ms = float(getattr(decision, "pause_ms", 0.0))
        if pause_ms >= self._soft_split_pause_ms:
            return True
        absolute_cap_ms = max(
            int(round(self._soft_final_audio_ms * 1.75)),
            self._soft_final_audio_ms + 3200,
        )
        return segment_audio_ms >= absolute_cap_ms

    def _record_partial_adaptation(self, *, latency_ms: int) -> None:
        if not self._adaptive_enabled:
            return
        self._adaptive_partial_latencies.append(max(0, int(latency_ms)))
        self._recompute_adaptive_tuning(now_ms=int(time.monotonic() * 1000))

    def _record_final_adaptation(self, *, audio_ms: int, latency_ms: int, now_ms: int) -> None:
        if not self._adaptive_enabled:
            return
        self._adaptive_recent_audio_ms.append(max(0, int(audio_ms)))
        self._adaptive_final_latencies.append(max(0, int(latency_ms)))
        self._recompute_adaptive_tuning(now_ms=now_ms)

    def _recompute_adaptive_tuning(self, *, now_ms: int) -> None:
        if not self._adaptive_enabled:
            return

        avg_audio_ms = (
            sum(self._adaptive_recent_audio_ms) / len(self._adaptive_recent_audio_ms)
            if self._adaptive_recent_audio_ms
            else 0.0
        )
        avg_partial_latency_ms = (
            sum(self._adaptive_partial_latencies) / len(self._adaptive_partial_latencies)
            if self._adaptive_partial_latencies
            else 0.0
        )
        avg_final_latency_ms = (
            sum(self._adaptive_final_latencies) / len(self._adaptive_final_latencies)
            if self._adaptive_final_latencies
            else 0.0
        )

        partial_interval_ms = self._base_partial_interval_ms
        min_silence_duration_ms = self._base_min_silence_duration_ms
        soft_final_audio_ms = self._base_soft_final_audio_ms
        mode_parts: list[str] = []

        if avg_audio_ms and avg_audio_ms <= 1800:
            partial_interval_ms = max(self._partial_interval_floor_ms, self._base_partial_interval_ms - 180)
            min_silence_duration_ms = max(180, int(round(self._base_min_silence_duration_ms * 0.75)))
            soft_final_audio_ms = max(self._force_final_audio_ms, int(round(self._base_soft_final_audio_ms * 0.85)))
            mode_parts.append("short_turn")
        elif avg_audio_ms >= 4200:
            partial_interval_ms = min(1800, self._base_partial_interval_ms + 260)
            min_silence_duration_ms = min(1100, int(round(self._base_min_silence_duration_ms * 1.15)))
            mode_parts.append("long_form")

        if (
            avg_partial_latency_ms >= 900
            or avg_final_latency_ms >= 1400
            or now_ms < self._adaptive_load_backoff_until_ms
        ):
            partial_interval_ms = min(2000, max(partial_interval_ms, self._base_partial_interval_ms + 360))
            soft_final_audio_ms = max(self._force_final_audio_ms, min(3600, self._base_soft_final_audio_ms - 600))
            mode_parts.append("load_shed")

        mode = "+".join(mode_parts) if mode_parts else "baseline"
        self._apply_adaptive_tuning(
            mode=mode,
            partial_interval_ms=partial_interval_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            soft_final_audio_ms=soft_final_audio_ms,
        )

    def _apply_adaptive_tuning(
        self,
        *,
        mode: str,
        partial_interval_ms: int,
        min_silence_duration_ms: int,
        soft_final_audio_ms: int,
        force: bool = False,
    ) -> None:
        next_partial_interval_ms = max(self._partial_interval_floor_ms, int(partial_interval_ms))
        next_min_silence_duration_ms = max(120, int(min_silence_duration_ms))
        next_soft_final_audio_ms = max(self._force_final_audio_ms, int(soft_final_audio_ms))
        current_min_silence_duration_ms = int(
            getattr(self._vad, "effective_min_silence_duration_ms", self._base_min_silence_duration_ms)
        )
        changed = force or any(
            (
                mode != self._adaptive_mode,
                next_partial_interval_ms != self._partial_interval_ms,
                next_min_silence_duration_ms != current_min_silence_duration_ms,
                next_soft_final_audio_ms != self._soft_final_audio_ms,
            )
        )
        self._adaptive_mode = mode
        self._partial_interval_ms = next_partial_interval_ms
        self._soft_final_audio_ms = next_soft_final_audio_ms
        if hasattr(self._vad, "set_runtime_tuning"):
            self._vad.set_runtime_tuning(min_silence_duration_ms=next_min_silence_duration_ms)
        if changed and not force:
            self._debug(
                "asr adaptive "
                f"mode={mode} partial_ms={next_partial_interval_ms} "
                f"silence_ms={next_min_silence_duration_ms} "
                f"soft_final_ms={next_soft_final_audio_ms}"
            )

    def _debug(self, message: str) -> None:
        with self._stats_lock:
            self._last_debug = message
        if self._on_debug:
            self._on_debug(message)

    def _segment_audio_ms(self) -> int:
        if not self._segment_chunks:
            return 0
        sample_count = sum(int(chunk.shape[0]) for chunk in self._segment_chunks)
        return int(sample_count * 1000 / max(1, self._segment_sample_rate))

    def _limited_audio(self, audio: np.ndarray, history_seconds: int) -> np.ndarray:
        max_frames = int(max(1, history_seconds) * self._segment_sample_rate)
        if audio.shape[0] <= max_frames:
            return audio
        return audio[-max_frames:]

    def _reset_segment(self) -> None:
        self._segment_chunks = []
        self._segment_start_ms = 0
        self._segment_end_ms = 0
        self._last_partial_ms = 0
        self._drop_partial_until_final = False
        self._in_speech_segment = False

    def _append_pre_roll_chunk(self, *, chunk: np.ndarray, sample_rate: float) -> None:
        if self._pre_roll_ms <= 0:
            return
        sr = max(1, int(sample_rate))
        if sr != self._pre_roll_sample_rate:
            self._pre_roll_chunks.clear()
            self._pre_roll_sample_count = 0
            self._pre_roll_sample_rate = sr
        copied = chunk.copy()
        self._pre_roll_chunks.append(copied)
        self._pre_roll_sample_count += int(copied.shape[0])
        max_samples = int(sr * self._pre_roll_ms / 1000)
        while self._pre_roll_chunks and self._pre_roll_sample_count > max_samples:
            dropped = self._pre_roll_chunks.popleft()
            self._pre_roll_sample_count -= int(dropped.shape[0])

    def _prime_segment_from_pre_roll(self, *, now_ms: int, sample_rate: float) -> bool:
        if not self._pre_roll_chunks:
            return False
        sr = max(1, int(sample_rate))
        self._segment_chunks = list(self._pre_roll_chunks)
        sample_count = self._pre_roll_sample_count
        pre_roll_ms = int(sample_count * 1000 / sr)
        self._segment_start_ms = max(0, now_ms - pre_roll_ms)
        self._pre_roll_chunks.clear()
        self._pre_roll_sample_count = 0
        self._pre_roll_sample_rate = sr
        return True


_HALLUCINATION_PATTERNS = (
    re.compile(r"^\s*thank(s| you)( everyone| all)?[.! ]*$", re.IGNORECASE),
    re.compile(r"^\s*thank(s| you)?\s+for\s+watching[.! ]*$", re.IGNORECASE),
    re.compile(r"^\s*thanks\s+for\s+your\s+watching[.! ]*$", re.IGNORECASE),
    re.compile(r"^\s*good night[.! ]*$", re.IGNORECASE),
    re.compile(r"^\s*bye(-| )?bye[.! ]*$", re.IGNORECASE),
    re.compile(r"^\s*y[' ]?all[.! ]*$", re.IGNORECASE),
    re.compile(r"^\s*(感謝|謝謝)(您的|你們的)?收看[。！! ]*$"),
    re.compile(r"^\s*感謝大家收看[。！! ]*$"),
    re.compile(r"^\s*謝謝(大家|各位)?[。！! ]*$"),
    re.compile(r"^\s*晚安[。！! ]*$"),
)

_SHORT_HALLUCINATION_NORMALIZED = {
    "bybwd6",
    "thankyou",
    "thanks",
    "thankyouall",
    "thankyoueveryone",
    "thankseveryone",
    "thanksall",
    "goodnight",
    "byebye",
    "\u8b1d\u8b1d\u5927\u5bb6",
    "\u8c22\u8c22\u5927\u5bb6",
    "\u611f\u8b1d\u5927\u5bb6",
    "\u611f\u8c22\u5927\u5bb6",
    "\u665a\u5b89",
    "\u62dc\u62dc",
    "\u6380\u6380",
    "\u611f\u8b1d\u6536\u770b",
    "\u611f\u8c22\u6536\u770b",
    "\u8b1d\u8b1d\u6536\u770b",
    "\u8c22\u8c22\u6536\u770b",
    "\u611f\u8b1d\u60a8\u7684\u6536\u770b",
    "\u611f\u8c22\u60a8\u7684\u6536\u770b",
    "\u8b1d\u8b1d\u60a8\u7684\u6536\u770b",
    "\u8c22\u8c22\u60a8\u7684\u6536\u770b",
    "\u611f\u8b1d\u89c0\u770b",
    "\u611f\u8c22\u89c2\u770b",
    "\u8b1d\u8b1d\u89c0\u770b",
    "\u8c22\u8c22\u89c2\u770b",
}

_NON_SPEECH_TEXT_PATTERNS = (
    re.compile(r"amara\.org", re.IGNORECASE),
    re.compile(r"yoyo\s+television\s+series\s+exclusive", re.IGNORECASE),
    re.compile(r"\bming\s*pao\b", re.IGNORECASE),
    re.compile(r"字幕.{0,8}(志願者|志愿者|由|提供)"),
    re.compile(r"中文.{0,4}(字幕|字暮).{0,4}(提供|製作|制作)"),
    re.compile(r"(請不吝|请不吝).{0,12}(點贊|点赞|訂閱|订阅|轉發|转发|打賞|打赏)"),
    re.compile(r"(明鏡與點點欄目|明镜与点点栏目)"),
)

_NON_SPEECH_NORMALIZED_SUBSTRINGS = (
    "優優獨播劇場yoyotelevisionseriesexclusive",
    "谢谢观看下次见",
    "謝謝觀看下次見",
    "感谢观看下次见",
    "感謝觀看下次見",
    "感謝觀看",
    "感谢观看",
    "感謝收看",
    "感谢收看",
    "字幕由amaraorg社群提供",
    "中文字幕提供",
    "請不吝點贊訂閱轉發打賞支援明鏡與點點欄目",
)


def _format_asr_exception_message(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    lowered = message.lower()
    if "dll load failed while importing _ssl" in lowered:
        return (
            f"{message}。ASR 執行環境缺少 OpenSSL 相依檔，"
            "請確認發行包中的 _internal/libssl-3-x64.dll、_internal/libcrypto-3-x64.dll 與 _internal/_ssl.pyd 都存在，"
            "並從 dist/SyncTranslate-onedir/SyncTranslate.exe 啟動。"
        )
    return message


def _looks_like_silence_hallucination(text: str, *, audio_ms: int, vad_rms: float) -> bool:
    value = (text or "").strip()
    if not value:
        return False
    if audio_ms > 1800:
        return False
    if vad_rms >= 0.035:
        return False
    compact = re.sub(r"\s+", " ", value)
    normalized = "".join(ch.lower() for ch in value if ch.isalnum())
    if normalized in _SHORT_HALLUCINATION_NORMALIZED:
        return True
    return any(pattern.match(compact) for pattern in _HALLUCINATION_PATTERNS)


def _transcript_drop_reason(
    text: str,
    *,
    audio_ms: int,
    vad_rms: float,
    expected_language: str,
) -> str:
    if _looks_like_known_non_speech_text(text):
        return "non-speech-overlay"
    if _looks_like_silence_hallucination(text, audio_ms=audio_ms, vad_rms=vad_rms):
        return "hallucinated"
    if _looks_like_script_mismatch_junk(text, expected_language=expected_language):
        return "script-mismatch"
    return ""


def _looks_like_known_non_speech_text(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return False
    compact = re.sub(r"\s+", " ", value)
    normalized = "".join(ch.lower() for ch in value if ch.isalnum())
    if any(token in normalized for token in _NON_SPEECH_NORMALIZED_SUBSTRINGS):
        return True
    return any(pattern.search(compact) for pattern in _NON_SPEECH_TEXT_PATTERNS)


def _looks_like_script_mismatch_junk(text: str, *, expected_language: str) -> bool:
    normalized_language = (expected_language or "").strip().lower()
    if not normalized_language:
        return False
    if "-" in normalized_language:
        normalized_language = normalized_language.split("-", 1)[0]

    compact = [ch for ch in (text or "").strip() if ch.isalpha()]
    if not compact or len(compact) > 6:
        return False
    if not _contains_cyrillic_or_greek(compact):
        return False

    if normalized_language == "zh":
        return not any(_is_cjk(ch) or _is_latin(ch) for ch in compact)
    if normalized_language == "en":
        return not all(_is_latin(ch) for ch in compact)
    if normalized_language == "ja":
        return not any(_is_japanese(ch) or _is_latin(ch) for ch in compact)
    if normalized_language == "ko":
        return not any(_is_hangul(ch) or _is_latin(ch) for ch in compact)
    return False


def _contains_cyrillic_or_greek(chars: list[str]) -> bool:
    for ch in chars:
        name = unicodedata.name(ch, "")
        if "CYRILLIC" in name or "GREEK" in name:
            return True
    return False


def _is_latin(ch: str) -> bool:
    codepoint = ord(ch)
    return (
        0x0041 <= codepoint <= 0x005A
        or 0x0061 <= codepoint <= 0x007A
        or 0x00C0 <= codepoint <= 0x024F
    )


def _is_cjk(ch: str) -> bool:
    codepoint = ord(ch)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )


def _is_hangul(ch: str) -> bool:
    codepoint = ord(ch)
    return 0xAC00 <= codepoint <= 0xD7AF


def _is_japanese(ch: str) -> bool:
    codepoint = ord(ch)
    return _is_cjk(ch) or 0x3040 <= codepoint <= 0x30FF
