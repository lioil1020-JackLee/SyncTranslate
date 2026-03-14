from __future__ import annotations

import time
from dataclasses import dataclass
from collections import deque
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import Callable

import numpy as np

from app.local_ai.faster_whisper_engine import FasterWhisperEngine
from app.local_ai.vad_segmenter import VadSegmenter


@dataclass(slots=True)
class AsrEvent:
    text: str
    is_final: bool
    start_ms: int
    end_ms: int
    latency_ms: int


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
        queue_maxsize: int = 128,
        on_event: Callable[[AsrEvent], None] | None = None,
        on_debug: Callable[[str], None] | None = None,
    ) -> None:
        self._engine = engine
        self._vad = vad
        # Keep partial decode conservative to avoid native decoder overload.
        self._partial_interval_ms = max(700, int(partial_interval_ms))
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
        self._min_partial_audio_ms = 600
        self._force_final_queue_size = max(8, self._queue.maxsize // 4)
        self._force_final_audio_ms = 1800
        self._soft_final_audio_ms = max(self._force_final_audio_ms, int(soft_final_audio_ms))
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

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=8.0)
            if self._thread.is_alive():
                # Do not clear/reset the shared stop event if old worker is still alive,
                # otherwise a subsequent start() can revive the previous worker thread.
                self._debug("asr worker stop timeout; skip reset to avoid concurrent workers")
                raise RuntimeError("ASR worker did not stop in time")
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
                self._emit_final(now_ms=now_ms)
                self._reset_segment()
                return
            if should_soft_split:
                self._debug(f"asr long speech: soft final audio_ms={segment_audio_ms}")
                self._emit_final(now_ms=now_ms)
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
            self._emit_final(now_ms=now_ms)
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
            text = self._engine.transcribe_partial(audio=audio, sample_rate=self._segment_sample_rate) or ""
        except Exception as exc:
            self._debug(f"partial asr failed: {exc}")
            now_ms = int(time.monotonic() * 1000)
            self._partial_cooldown_until_ms = max(self._partial_cooldown_until_ms, now_ms + 12000)
            self._debug("asr safety mode: partial decode cooling down after partial failure")
            return
        if not text.strip():
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
                start_ms=self._segment_start_ms,
                end_ms=self._segment_end_ms,
                latency_ms=latency_ms,
            )
        )
        with self._stats_lock:
            self._partial_count += 1
        self._debug(f"asr partial latency={latency_ms}ms text={text.strip()[:80]}")

    def _emit_final(self, *, now_ms: int) -> None:
        if not self._on_event or not self._segment_chunks:
            return
        audio = np.concatenate(self._segment_chunks, axis=0)
        audio = self._limited_audio(audio, self._final_history_seconds)
        start = time.perf_counter()
        try:
            text = self._engine.transcribe_final(audio=audio, sample_rate=self._segment_sample_rate) or ""
        except Exception as exc:
            self._on_event(
                AsrEvent(
                    text=f"[asr-error] {exc}",
                    is_final=True,
                    start_ms=self._segment_start_ms,
                    end_ms=now_ms,
                    latency_ms=0,
                )
            )
            return
        latency_ms = int((time.perf_counter() - start) * 1000)
        if latency_ms >= 1500:
            now_ms = int(time.monotonic() * 1000)
            self._partial_cooldown_until_ms = max(self._partial_cooldown_until_ms, now_ms + 6000)
            self._debug(f"asr partial backoff: latency={latency_ms}ms")
        if not text.strip():
            return
        self._on_event(
            AsrEvent(
                text=text.strip(),
                is_final=True,
                start_ms=self._segment_start_ms,
                end_ms=now_ms,
                latency_ms=latency_ms,
            )
        )
        with self._stats_lock:
            self._final_count += 1
        self._debug(f"asr final latency={latency_ms}ms text={text.strip()[:80]}")

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
