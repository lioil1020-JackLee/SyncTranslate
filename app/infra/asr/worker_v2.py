from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import Callable

import numpy as np

from app.infra.asr.endpointing_v2 import EndpointSignal, EndpointingRuntime
from app.infra.asr.frontend_v2 import AsrAudioFrontendV2


@dataclass(slots=True)
class V2RuntimeEvent:
    text: str
    is_final: bool
    is_early_final: bool
    start_ms: int
    end_ms: int
    latency_ms: int
    detected_language: str = ""


@dataclass(slots=True)
class SourceRuntimeV2Stats:
    queue_size: int
    dropped_chunks: int
    partial_count: int
    final_count: int
    last_debug: str
    partial_backend_name: str
    final_backend_name: str
    backend_runtime: dict[str, object]
    endpointing: dict[str, object]
    last_signal: EndpointSignal
    frontend: dict[str, object]


class SourceRuntimeV2:
    def __init__(
        self,
        *,
        source: str,
        partial_backend: object,
        final_backend: object,
        endpointing: EndpointingRuntime,
        partial_interval_ms: int,
        partial_history_seconds: int,
        final_history_seconds: int,
        soft_final_audio_ms: int,
        pre_roll_ms: int,
        min_partial_audio_ms: int,
        queue_maxsize: int,
        frontend_enabled: bool = True,
        frontend_target_rms: float = 0.05,
        frontend_max_gain: float = 3.0,
        frontend_highpass_alpha: float = 0.96,
        enhancement_enabled: bool = True,
        enhancement_noise_reduce_strength: float = 0.42,
        enhancement_noise_adapt_rate: float = 0.18,
        enhancement_music_suppress_strength: float = 0.2,
        on_event: Callable[[V2RuntimeEvent], None] | None = None,
        on_debug: Callable[[str], None] | None = None,
    ) -> None:
        self._source = source
        self._partial_backend = partial_backend
        self._final_backend = final_backend
        self._endpointing = endpointing
        self._frontend = AsrAudioFrontendV2(
            enabled=frontend_enabled,
            target_rms=frontend_target_rms,
            max_gain=frontend_max_gain,
            highpass_alpha=frontend_highpass_alpha,
            enhancement_enabled=enhancement_enabled,
            enhancement_noise_reduce_strength=enhancement_noise_reduce_strength,
            enhancement_noise_adapt_rate=enhancement_noise_adapt_rate,
            enhancement_music_suppress_strength=enhancement_music_suppress_strength,
        )
        self._partial_interval_ms = max(200, int(partial_interval_ms))
        self._partial_history_seconds = max(1, int(partial_history_seconds))
        self._final_history_seconds = max(2, int(final_history_seconds))
        self._soft_final_audio_ms = max(1200, int(soft_final_audio_ms))
        self._pre_roll_ms = max(0, int(pre_roll_ms))
        self._min_partial_audio_ms = max(240, int(min_partial_audio_ms))
        self._queue: Queue[tuple[np.ndarray, float]] = Queue(maxsize=max(4, queue_maxsize))
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._on_event = on_event
        self._on_debug = on_debug
        self._last_signal = EndpointSignal()
        self._last_partial_emit_ms = 0
        self._last_partial_text = ""
        self._segment_chunks: list[np.ndarray] = []
        self._segment_sample_rate = 16000
        self._segment_start_ms = 0
        self._pre_roll_chunks: deque[np.ndarray] = deque()
        self._pre_roll_sample_count = 0
        self._pre_roll_sample_rate = 16000
        self._in_segment = False
        self._force_final_queue_size = max(8, self._queue.maxsize // 4)
        self._force_final_audio_ms = 1800
        final_backend_name = str(getattr(getattr(self._final_backend, "descriptor", None), "name", "")).lower()
        self._prefer_conservative_finalize = "funasr_v2" in final_backend_name
        self._soft_endpoint_finalize_audio_ms = max(
            1500 if self._prefer_conservative_finalize else 900,
            self._min_partial_audio_ms,
        )
        self._speech_end_finalize_audio_ms = max(
            1300 if self._prefer_conservative_finalize else 900,
            min(self._soft_endpoint_finalize_audio_ms, self._min_partial_audio_ms + 300),
        )
        self._adaptive_length_floor_ms = max(3200, self._soft_final_audio_ms)
        self._adaptive_length_ceiling_ms = max(self._adaptive_length_floor_ms, 12000)
        self._drop_partial_until_final = False
        self._partial_cooldown_until_ms = 0
        self._partial_count = 0
        self._final_count = 0
        self._dropped_chunks = 0
        self._last_debug = ""
        self._stats_lock = Lock()

    def start(self, on_event: Callable[[V2RuntimeEvent], None]) -> None:
        self.stop()
        self._on_event = on_event
        self._stop_event.clear()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=8.0)
        self._thread = None
        self._endpointing.reset()
        self._frontend.reset()
        self._segment_chunks = []
        self._segment_sample_rate = 16000
        self._segment_start_ms = 0
        self._pre_roll_chunks.clear()
        self._pre_roll_sample_count = 0
        self._pre_roll_sample_rate = 16000
        self._in_segment = False
        self._drop_partial_until_final = False
        self._partial_cooldown_until_ms = 0
        self._last_signal = EndpointSignal()
        self._last_partial_emit_ms = 0
        self._last_partial_text = ""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Empty:
                break

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
            self._dropped_chunks += 1
            now_ms = int(time.monotonic() * 1000)
            self._drop_partial_until_final = True
            self._partial_cooldown_until_ms = max(self._partial_cooldown_until_ms, now_ms + 8000)
            self._debug("v2 queue overflow: dropped oldest chunk")

    def stats(self) -> SourceRuntimeV2Stats:
        with self._stats_lock:
            return SourceRuntimeV2Stats(
                queue_size=self._queue.qsize(),
                dropped_chunks=self._dropped_chunks,
                partial_count=self._partial_count,
                final_count=self._final_count,
                last_debug=self._last_debug,
                partial_backend_name=str(getattr(getattr(self._partial_backend, "descriptor", None), "name", "")),
                final_backend_name=str(getattr(getattr(self._final_backend, "descriptor", None), "name", "")),
                backend_runtime=self._backend_runtime_info(),
                endpointing=self._endpointing.snapshot(),
                last_signal=self._last_signal,
                frontend=self._frontend.stats(),
            )

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
                self._process_chunk(
                    chunk=np.concatenate(pending_parts, axis=0).astype(np.float32, copy=False),
                    sample_rate=sample_rate,
                )
                pending_parts = [extra_chunk]
                sample_rate = extra_rate
                drained = 0

            self._process_chunk(
                chunk=np.concatenate(pending_parts, axis=0).astype(np.float32, copy=False),
                sample_rate=sample_rate,
            )

    def _process_chunk(self, *, chunk: np.ndarray, sample_rate: float) -> None:
        if sample_rate <= 0:
            return
        prepared = self._frontend.process(chunk, sample_rate)
        chunk = prepared.audio
        if chunk.size == 0:
            return
        sample_rate_int = prepared.sample_rate
        self._append_pre_roll(chunk, sample_rate_int)
        signal = self._endpointing.update(chunk, sample_rate_int)
        self._last_signal = signal
        now_ms = int(time.monotonic() * 1000)

        if signal.speech_started and not self._in_segment:
            self._start_segment(sample_rate=sample_rate_int, now_ms=now_ms)

        if self._in_segment:
            self._segment_chunks.append(chunk)
            self._segment_sample_rate = sample_rate_int
            segment_audio_ms = self._segment_audio_ms()
            backlog = self._queue.qsize()
            should_force_final = backlog >= self._force_final_queue_size and segment_audio_ms >= self._force_final_audio_ms
            can_emit_partial = (
                signal.speech_active
                and segment_audio_ms >= self._min_partial_audio_ms
                and backlog <= 1
                and not self._drop_partial_until_final
                and now_ms >= self._partial_cooldown_until_ms
            )
            if can_emit_partial and now_ms - self._last_partial_emit_ms >= self._partial_interval_ms:
                self._emit_partial(now_ms=now_ms)
            adaptive_length_limit_ms = self._adaptive_length_limit_ms(signal=signal)
            should_finalize_on_soft_endpoint = (
                signal.soft_endpoint
                and segment_audio_ms >= self._soft_endpoint_finalize_audio_ms
            )
            should_finalize_on_speech_end = (
                signal.speech_ended
                and segment_audio_ms >= self._speech_end_finalize_audio_ms
            )
            should_finalize = (
                signal.hard_endpoint
                or should_finalize_on_speech_end
                or should_finalize_on_soft_endpoint
                or (segment_audio_ms >= adaptive_length_limit_ms and signal.pause_ms >= 180.0)
                or should_force_final
            )
            if should_finalize:
                self._emit_final(
                    now_ms=now_ms,
                    is_early_final=(not signal.hard_endpoint),
                )
                if should_finalize_on_soft_endpoint and not signal.hard_endpoint:
                    # Treat a soft endpoint final as an utterance boundary so the
                    # next speech burst can immediately start a new segment.
                    self._endpointing.reset()
                self._reset_segment()

    def _start_segment(self, *, sample_rate: int, now_ms: int) -> None:
        self._in_segment = True
        self._segment_sample_rate = sample_rate
        self._segment_start_ms = now_ms
        self._segment_chunks = list(self._pre_roll_chunks)
        self._last_partial_emit_ms = 0
        self._last_partial_text = ""

    def _reset_segment(self) -> None:
        self._in_segment = False
        self._segment_chunks = []
        self._segment_start_ms = 0
        self._last_partial_emit_ms = 0
        self._last_partial_text = ""
        self._drop_partial_until_final = False

    def _append_pre_roll(self, chunk: np.ndarray, sample_rate: int) -> None:
        if self._pre_roll_sample_rate != sample_rate:
            self._pre_roll_chunks.clear()
            self._pre_roll_sample_count = 0
            self._pre_roll_sample_rate = sample_rate
        self._pre_roll_chunks.append(chunk.copy())
        self._pre_roll_sample_count += int(chunk.shape[0])
        max_samples = int(sample_rate * (self._pre_roll_ms / 1000.0))
        while self._pre_roll_chunks and self._pre_roll_sample_count > max_samples:
            removed = self._pre_roll_chunks.popleft()
            self._pre_roll_sample_count -= int(removed.shape[0])

    def _emit_partial(self, *, now_ms: int) -> None:
        if self._on_event is None:
            return
        audio = self._limited_audio(self._partial_history_seconds)
        start = time.perf_counter()
        result = self._transcribe_backend(
            self._partial_backend,
            method_name="transcribe_partial",
            audio=audio,
            sample_rate=self._segment_sample_rate,
        )
        text = (result.text or "").strip()
        if not text or text == self._last_partial_text:
            return
        latency_ms = int((time.perf_counter() - start) * 1000)
        self._last_partial_text = text
        self._last_partial_emit_ms = now_ms
        self._on_event(
            V2RuntimeEvent(
                text=text,
                is_final=False,
                is_early_final=False,
                start_ms=self._segment_start_ms,
                end_ms=now_ms,
                latency_ms=latency_ms,
                detected_language=result.detected_language,
            )
        )
        with self._stats_lock:
            self._partial_count += 1

    def _emit_final(self, *, now_ms: int, is_early_final: bool) -> None:
        if self._on_event is None:
            return
        audio = self._limited_audio(self._final_history_seconds)
        start = time.perf_counter()
        result = self._transcribe_backend(
            self._final_backend,
            method_name="transcribe_final",
            audio=audio,
            sample_rate=self._segment_sample_rate,
        )
        text = (result.text or "").strip()
        if not text:
            return
        latency_ms = int((time.perf_counter() - start) * 1000)
        self._on_event(
            V2RuntimeEvent(
                text=text,
                is_final=True,
                is_early_final=bool(is_early_final),
                start_ms=self._segment_start_ms,
                end_ms=now_ms,
                latency_ms=latency_ms,
                detected_language=result.detected_language,
            )
        )
        with self._stats_lock:
            self._final_count += 1

    def _limited_audio(self, history_seconds: int) -> np.ndarray:
        if not self._segment_chunks:
            return np.zeros((0,), dtype=np.float32)
        audio = np.concatenate(self._segment_chunks, axis=0).astype(np.float32, copy=False)
        max_samples = int(self._segment_sample_rate * max(1, history_seconds))
        if audio.shape[0] <= max_samples:
            return audio
        return audio[-max_samples:]

    def _segment_audio_ms(self) -> int:
        if not self._segment_chunks:
            return 0
        total_samples = sum(int(chunk.shape[0]) for chunk in self._segment_chunks)
        return int(total_samples * 1000 / max(1, self._segment_sample_rate))

    def _debug(self, message: str) -> None:
        with self._stats_lock:
            self._last_debug = message
        if self._on_debug:
            self._on_debug(message)

    def _backend_runtime_info(self) -> dict[str, object]:
        info_fn = getattr(self._final_backend, "runtime_info", None)
        if callable(info_fn):
            try:
                info = info_fn()
                if isinstance(info, dict):
                    return info
            except Exception:
                pass
        return {
            "device_effective": "",
            "model_init_mode": "lazy",
            "init_failure": "",
        }

    def _transcribe_backend(
        self,
        backend: object,
        *,
        method_name: str,
        audio: np.ndarray,
        sample_rate: int,
    ):
        method = getattr(backend, method_name)
        frontend_stats = self._frontend.stats()
        try:
            return method(audio, sample_rate, frontend_stats=frontend_stats)
        except TypeError as exc:
            if "frontend_stats" not in str(exc):
                raise
            return method(audio, sample_rate)

    def _adaptive_length_limit_ms(self, *, signal: EndpointSignal) -> int:
        if signal.pause_ms >= 420.0:
            return max(self._adaptive_length_floor_ms, 4200)
        if signal.pause_ms >= 260.0:
            return max(self._adaptive_length_floor_ms, 5600)
        if signal.pause_ms >= 120.0:
            return max(self._adaptive_length_floor_ms, 7200)
        return self._adaptive_length_ceiling_ms


__all__ = ["V2RuntimeEvent", "SourceRuntimeV2", "SourceRuntimeV2Stats"]
