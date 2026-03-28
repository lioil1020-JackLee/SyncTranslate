from __future__ import annotations

import time
from dataclasses import dataclass
from collections import deque
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import Callable
import re

import numpy as np

from app.infra.asr.faster_whisper_adapter import FasterWhisperEngine
from app.infra.asr.language_policy import VadSegmenter


@dataclass(slots=True)
class AsrEvent:
    text: str
    is_final: bool
    is_early_final: bool
    start_ms: int
    end_ms: int
    latency_ms: int
    detected_language: str = ""


@dataclass(slots=True)
class StreamingAsrStats:
    queue_size: int
    dropped_chunks: int
    partial_count: int
    final_count: int
    last_debug: str
    vad_rms: float
    vad_threshold: float


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
        queue_maxsize: int = 128,
        on_event: Callable[[AsrEvent], None] | None = None,
        on_debug: Callable[[str], None] | None = None,
    ) -> None:
        self._engine = engine
        self._vad = vad
        # Keep partial decode conservative to avoid native decoder overload.
        self._partial_interval_ms = max(int(partial_interval_floor_ms), int(partial_interval_ms))
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
        self._soft_final_audio_ms = max(self._force_final_audio_ms, int(soft_final_audio_ms))
        self._early_final_enabled = bool(early_final_enabled)
        self._overflow_count = 0
        self._last_overflow_report_ms = 0
        self._drop_partial_until_final = False
        self._partial_cooldown_until_ms = 0
        self._pre_roll_chunks: deque[np.ndarray] = deque()
        self._pre_roll_sample_count = 0
        self._pre_roll_sample_rate = 16000
        self._in_speech_segment = False
        self._partial_count = 0
        self._final_count = 0
        self._last_debug = ""
        self._stats_lock = Lock()

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
            should_soft_split = self._queue.qsize() == 0 and segment_audio_ms >= self._soft_final_audio_ms
            if should_force_final:
                self._debug(
                    "asr backlog: force final "
                    f"queue={self._queue.qsize()} audio_ms={segment_audio_ms}"
                )
                self._emit_final(now_ms=now_ms, is_early_final=False)
                self._reset_segment()
                return
            if should_soft_split:
                self._debug(f"asr long speech: soft final audio_ms={segment_audio_ms}")
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
        if _looks_like_silence_hallucination(text, audio_ms=audio_ms, vad_rms=self._vad.last_rms):
            self._debug(f"drop hallucinated partial text={text.strip()[:80]}")
            return
        latency_ms = int((time.perf_counter() - start) * 1000)
        if latency_ms >= 1500:
            now_ms = int(time.monotonic() * 1000)
            self._partial_cooldown_until_ms = max(self._partial_cooldown_until_ms, now_ms + 15000)
            self._debug("asr safety mode: partial decode cooling down after high partial latency")
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
            self._on_event(
                AsrEvent(
                    text=f"[asr-error] {exc}",
                    is_final=True,
                    is_early_final=is_early_final,
                    start_ms=self._segment_start_ms,
                    end_ms=now_ms,
                    latency_ms=0,
                )
            )
            return
        audio_ms = int(len(audio) * 1000 / max(1, self._segment_sample_rate))
        if _looks_like_silence_hallucination(text, audio_ms=audio_ms, vad_rms=self._vad.last_rms):
            self._debug(f"drop hallucinated final text={text.strip()[:80]}")
            return
        latency_ms = int((time.perf_counter() - start) * 1000)
        if latency_ms >= 1500:
            now_ms = int(time.monotonic() * 1000)
            self._partial_cooldown_until_ms = max(self._partial_cooldown_until_ms, now_ms + 6000)
            self._debug(f"asr partial backoff: latency={latency_ms}ms")
        if not text.strip():
            return
        if self._stop_event.is_set():
            return
        self._on_event(
            AsrEvent(
                text=text.strip(),
                is_final=True,
                is_early_final=is_early_final,
                start_ms=self._segment_start_ms,
                end_ms=now_ms,
                latency_ms=latency_ms,
                detected_language=result.detected_language,
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
        if segment_audio_ms < 300 or segment_audio_ms > 2600:
            return False
        configured_silence_ms = float(getattr(getattr(self._vad, "_config", None), "min_silence_duration_ms", 240))
        early_silence_ms = max(120.0, min(220.0, configured_silence_ms * 0.7))
        return silence_ms >= early_silence_ms

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
    re.compile(r"^\s*good night[.! ]*$", re.IGNORECASE),
    re.compile(r"^\s*bye(-| )?bye[.! ]*$", re.IGNORECASE),
    re.compile(r"^\s*y[' ]?all[.! ]*$", re.IGNORECASE),
    re.compile(r"^\s*謝謝(大家|各位)?[。！! ]*$"),
    re.compile(r"^\s*晚安[。！! ]*$"),
)


def _looks_like_silence_hallucination(text: str, *, audio_ms: int, vad_rms: float) -> bool:
    value = (text or "").strip()
    if not value:
        return False
    if audio_ms > 1800:
        return False
    if vad_rms >= 0.035:
        return False
    compact = re.sub(r"\s+", " ", value)
    return any(pattern.match(compact) for pattern in _HALLUCINATION_PATTERNS)
