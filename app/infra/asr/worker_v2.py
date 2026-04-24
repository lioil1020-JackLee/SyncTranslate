from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import Callable

import numpy as np

from app.infra.asr.endpointing_v2 import EndpointSignal, EndpointingRuntime
from app.infra.asr.frontend_v2 import AsrAudioFrontendV2
from app.infra.asr.streaming_policy import (
    DEGRADATION_CONGESTED,
    DEGRADATION_DEGRADED,
    DEGRADATION_NORMAL,
    StreamingContext,
    StreamingDecision,
    StreamingPolicy,
)


def _scaled_finalize_thresholds(
    *,
    soft_final_audio_ms: int,
    min_partial_audio_ms: int,
    force_final_audio_ms: int,
) -> tuple[int, int]:
    """Derive conservative finalize thresholds from the configured soft-final window.

    Short fixed thresholds (for example ~900ms) are too aggressive for long-form
    narration and can fragment one spoken sentence into many tiny finals. Scale
    the thresholds from the configured soft-final window instead so the worker
    stays responsive on short turns without over-segmenting longer speech.
    """
    base_soft_final = max(1200, int(soft_final_audio_ms))
    base_partial = max(240, int(min_partial_audio_ms))
    base_force_final = max(1200, int(force_final_audio_ms))

    # Finals should happen promptly once speech has clearly ended.
    # Overly conservative thresholds keep the UI stuck in partial state and
    # make the eventual final decode re-run over too much audio.
    soft_endpoint_finalize_audio_ms = max(
        base_force_final,
        base_partial + 520,
        int(round(base_soft_final * 0.40)),
    )
    speech_end_finalize_audio_ms = max(
        720,
        base_partial + 220,
        int(round(base_soft_final * 0.26)),
    )
    speech_end_finalize_audio_ms = min(speech_end_finalize_audio_ms, soft_endpoint_finalize_audio_ms)
    return soft_endpoint_finalize_audio_ms, speech_end_finalize_audio_ms


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
    adaptive_partial_interval_ms: int
    adaptive_soft_final_audio_ms: int
    last_debug: str
    partial_backend_name: str
    final_backend_name: str
    backend_runtime: dict[str, object]
    endpointing: dict[str, object]
    last_signal: EndpointSignal
    frontend: dict[str, object]
    degradation_level: str = DEGRADATION_NORMAL


@dataclass(slots=True)
class SegmentSignalStats:
    audio_ms: int = 0
    speech_ms: int = 0
    trailing_silence_ms: int = 0
    speech_ratio: float = 0.0
    mean_rms: float = 0.0
    max_rms: float = 0.0
    mean_probability: float = 0.0
    max_probability: float = 0.0
    speech_threshold: float = 0.0


_SHORT_TAIL_HALLUCINATION_NORMALIZED = {
    "you",
    "bye",
    "byebye",
    "thankyou",
    "thanks",
    "thankyouall",
    "thankyoueveryone",
    "thankseveryone",
    "thanksall",
    "thankyouforwatching",
    "thanksforwatching",
    "thanksforyourwatching",
    "goodnight",
}

_TAIL_HALLUCINATION_PATTERNS = (
    re.compile(r"^\s*thank(s| you)( everyone| all)?[.! ]*$", re.IGNORECASE),
    re.compile(r"^\s*thank(s| you)?\s+for\s+watching[.! ]*$", re.IGNORECASE),
    re.compile(r"^\s*thanks\s+for\s+your\s+watching[.! ]*$", re.IGNORECASE),
    re.compile(r"^\s*bye(-| )?bye?[.! ]*$", re.IGNORECASE),
)

_CJK_TAIL_HALLUCINATION_PATTERNS = (
    re.compile(r"^[哈呵嘿哇嗚呃嗯啊]{2,}[!！。,.，\s]*$"),
    re.compile(r"^(記得)?按(下|讚).*(訂閱|小鈴鐺)?.*$"),
    re.compile(r"^.*(安妞|按鈕哦|訂閱按鈕|小鈴鐺|頻道嘍|頻道囉|記得按讚).{0,8}$"),
    re.compile(r"^歡迎收看.*下次見[!！。,.，\s]*$"),
)


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
        early_final_enabled: bool = True,
        partial_interval_floor_ms: int = 280,
        adaptive_enabled: bool = True,
        degradation_enabled: bool = True,
        soft_endpoint_finalize_audio_ms: int | None = None,
        speech_end_finalize_audio_ms: int | None = None,
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
        self._partial_interval_floor_ms = max(120, int(partial_interval_floor_ms))
        self._base_partial_interval_ms = max(self._partial_interval_floor_ms, int(partial_interval_ms))
        self._partial_interval_ms = self._base_partial_interval_ms
        self._partial_history_seconds = max(1, int(partial_history_seconds))
        self._final_history_seconds = max(2, int(final_history_seconds))
        self._base_soft_final_audio_ms = max(1200, int(soft_final_audio_ms))
        self._soft_final_audio_ms = self._base_soft_final_audio_ms
        self._pre_roll_ms = max(0, int(pre_roll_ms))
        self._min_partial_audio_ms = max(240, int(min_partial_audio_ms))
        self._early_final_enabled = bool(early_final_enabled)
        self._adaptive_enabled = bool(adaptive_enabled)
        self._queue: Queue[tuple[np.ndarray, float]] = Queue(maxsize=max(4, queue_maxsize))
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._on_event = on_event
        self._on_debug = on_debug
        self._last_signal = EndpointSignal()
        self._last_partial_emit_ms = 0
        self._last_partial_text = ""
        self._last_partial_latency_ms = 0
        self._deferred_early_final_until_ms = 0
        self._segment_chunks: list[np.ndarray] = []
        self._segment_sample_rate = 16000
        self._segment_start_ms = 0
        self._segment_signal_audio_ms = 0.0
        self._segment_signal_speech_ms = 0.0
        self._segment_signal_trailing_silence_ms = 0.0
        self._segment_signal_rms_weighted_sum = 0.0
        self._segment_signal_probability_weighted_sum = 0.0
        self._segment_signal_max_rms = 0.0
        self._segment_signal_max_probability = 0.0
        self._segment_signal_threshold = 0.0
        self._pre_roll_chunks: deque[np.ndarray] = deque()
        self._pre_roll_sample_count = 0
        self._pre_roll_sample_rate = 16000
        self._in_segment = False
        self._force_final_queue_size = max(8, self._queue.maxsize // 4)
        self._force_final_audio_ms = 1500
        self._prefer_conservative_finalize = False
        scaled_soft_ms, scaled_speech_end_ms = _scaled_finalize_thresholds(
            soft_final_audio_ms=self._base_soft_final_audio_ms,
            min_partial_audio_ms=self._min_partial_audio_ms,
            force_final_audio_ms=self._force_final_audio_ms,
        )
        self._base_soft_endpoint_finalize_audio_ms = max(
            self._force_final_audio_ms,
            int(soft_endpoint_finalize_audio_ms) if soft_endpoint_finalize_audio_ms is not None else scaled_soft_ms,
        )
        self._base_speech_end_finalize_audio_ms = min(
            self._base_soft_endpoint_finalize_audio_ms,
            max(
                900,
                int(speech_end_finalize_audio_ms) if speech_end_finalize_audio_ms is not None else scaled_speech_end_ms,
            ),
        )
        self._soft_endpoint_finalize_audio_ms = self._base_soft_endpoint_finalize_audio_ms
        self._speech_end_finalize_audio_ms = self._base_speech_end_finalize_audio_ms
        self._adaptive_length_floor_ms = max(2200, int(self._soft_final_audio_ms * 0.72))
        self._adaptive_length_ceiling_ms = max(self._adaptive_length_floor_ms, 9000)
        self._drop_partial_until_final = False
        self._partial_cooldown_until_ms = 0
        self._adaptive_recent_audio_ms: deque[int] = deque(maxlen=10)
        self._adaptive_partial_latencies: deque[int] = deque(maxlen=10)
        self._adaptive_final_latencies: deque[int] = deque(maxlen=10)
        self._partial_count = 0
        self._final_count = 0
        self._dropped_chunks = 0
        self._last_debug = ""
        self._last_degradation_level: str = DEGRADATION_NORMAL
        self._stats_lock = Lock()
        self._streaming_policy = StreamingPolicy(degradation_enabled=bool(degradation_enabled))

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
        self._reset_segment_signal_stats()
        self._pre_roll_chunks.clear()
        self._pre_roll_sample_count = 0
        self._pre_roll_sample_rate = 16000
        self._in_segment = False
        self._drop_partial_until_final = False
        self._partial_cooldown_until_ms = 0
        self._last_signal = EndpointSignal()
        self._last_partial_emit_ms = 0
        self._last_partial_text = ""
        self._last_partial_latency_ms = 0
        self._deferred_early_final_until_ms = 0
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
                adaptive_partial_interval_ms=self._partial_interval_ms,
                adaptive_soft_final_audio_ms=self._soft_final_audio_ms,
                last_debug=self._last_debug,
                partial_backend_name=str(getattr(getattr(self._partial_backend, "descriptor", None), "name", "")),
                final_backend_name=str(getattr(getattr(self._final_backend, "descriptor", None), "name", "")),
                backend_runtime=self._backend_runtime_info(),
                endpointing=self._endpointing.snapshot(),
                last_signal=self._last_signal,
                frontend=self._frontend.stats(),
                degradation_level=self._last_degradation_level,
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
        chunk_ms = int(round(chunk.shape[0] * 1000 / max(1, sample_rate_int)))
        self._append_pre_roll(chunk, sample_rate_int)
        signal = self._endpointing.update(chunk, sample_rate_int)
        self._last_signal = signal
        now_ms = int(time.monotonic() * 1000)

        if signal.speech_started and not self._in_segment:
            self._start_segment(sample_rate=sample_rate_int, now_ms=now_ms)

        if self._in_segment:
            self._segment_chunks.append(chunk)
            self._segment_sample_rate = sample_rate_int
            self._record_segment_signal(signal=signal, chunk_ms=chunk_ms)
            segment_audio_ms = self._segment_audio_ms()
            backlog = self._queue.qsize()

            ctx = StreamingContext(
                signal=signal,
                segment_audio_ms=segment_audio_ms,
                now_ms=now_ms,
                last_partial_emit_ms=self._last_partial_emit_ms,
                backlog=backlog,
                drop_partial_until_final=self._drop_partial_until_final,
                partial_cooldown_until_ms=self._partial_cooldown_until_ms,
                dropped_chunks_total=self._dropped_chunks,
                partial_interval_ms=self._partial_interval_ms,
                min_partial_audio_ms=self._min_partial_audio_ms,
                soft_endpoint_finalize_audio_ms=self._soft_endpoint_finalize_audio_ms,
                speech_end_finalize_audio_ms=self._speech_end_finalize_audio_ms,
                adaptive_length_limit_ms=self._adaptive_length_limit_ms(signal=signal),
                adaptive_length_ceiling_ms=self._adaptive_length_ceiling_ms,
                force_final_queue_size=self._force_final_queue_size,
                force_final_audio_ms=self._force_final_audio_ms,
            )
            decision = self._streaming_policy.decide(ctx)
            self._last_degradation_level = decision.degradation_level

            if decision.emit_partial:
                self._emit_partial(now_ms=now_ms)
            if decision.emit_final:
                if decision.is_early_final and not self._early_final_enabled:
                    return
                if self._should_defer_early_final(
                    now_ms=now_ms,
                    decision=decision,
                    signal=signal,
                    segment_audio_ms=segment_audio_ms,
                ):
                    self._recompute_adaptive_tuning(now_ms=now_ms, segment_audio_ms=segment_audio_ms, signal=signal)
                    return
                self._emit_final(
                    now_ms=now_ms,
                    is_early_final=decision.is_early_final,
                )
                if decision.reset_endpointing_after_final:
                    # Treat a soft endpoint final as an utterance boundary so the
                    # next speech burst can immediately start a new segment.
                    self._endpointing.reset()
                self._reset_segment()
                # If speech was still active when we hit a ceiling/force-final
                # (not a speech_ended boundary), immediately start a new segment
                # so continuous speech (e.g. with background music) is not lost.
                if signal.speech_active and not signal.speech_ended:
                    self._start_segment(sample_rate=sample_rate_int, now_ms=now_ms)
            self._recompute_adaptive_tuning(now_ms=now_ms, segment_audio_ms=segment_audio_ms, signal=signal)

    def _start_segment(self, *, sample_rate: int, now_ms: int) -> None:
        self._in_segment = True
        self._segment_sample_rate = sample_rate
        self._segment_start_ms = now_ms
        self._segment_chunks = list(self._pre_roll_chunks)
        self._reset_segment_signal_stats()
        self._last_partial_emit_ms = 0
        self._last_partial_text = ""
        self._last_partial_latency_ms = 0
        self._deferred_early_final_until_ms = 0

    def _reset_segment(self) -> None:
        self._in_segment = False
        self._segment_chunks = []
        self._segment_start_ms = 0
        self._reset_segment_signal_stats()
        self._last_partial_emit_ms = 0
        self._last_partial_text = ""
        self._last_partial_latency_ms = 0
        self._deferred_early_final_until_ms = 0
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
        if not text and getattr(result, "rejection_reason", ""):
            self._debug(f"v2 partial rejected reason={result.rejection_reason}")
            return
        if not text or text == self._last_partial_text:
            return
        latency_ms = int((time.perf_counter() - start) * 1000)
        self._adaptive_partial_latencies.append(max(0, latency_ms))
        self._last_partial_text = text
        self._last_partial_latency_ms = latency_ms
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
        segment_stats = self._segment_signal_stats(audio=audio, sample_rate=self._segment_sample_rate)
        frontend_stats = self._segment_frontend_stats(segment_stats)
        start = time.perf_counter()
        result = self._transcribe_backend(
            self._final_backend,
            method_name="transcribe_final",
            audio=audio,
            sample_rate=self._segment_sample_rate,
            frontend_stats=frontend_stats,
        )
        result = self._retry_empty_final_with_shorter_tail(
            result=result,
            audio=audio,
            sample_rate=self._segment_sample_rate,
        )
        if not (result.text or "").strip() and getattr(result, "rejection_reason", ""):
            self._debug(f"v2 final rejected reason={result.rejection_reason}")
            return
        final_text = (result.text or "").strip()
        merge_reason = self._reason_to_prefer_last_partial(
            final_text=final_text,
            last_partial_text=self._last_partial_text,
        )
        text = final_text
        used_partial_fallback = False
        if merge_reason:
            fallback_partial = self._last_partial_text.strip()
            if merge_reason == "empty-final" and not self._should_promote_empty_final_partial(fallback_partial):
                self._debug(
                    "v2 final suppress_empty_partial_fallback "
                    f"partial={fallback_partial!r}"
                )
            else:
                text = fallback_partial
                used_partial_fallback = True
        if used_partial_fallback:
            self._debug(
                "v2 final fallback_to_partial "
                f"reason={merge_reason} final={final_text!r} partial={self._last_partial_text.strip()!r}"
            )
        if not text:
            return
        latency_ms = int((time.perf_counter() - start) * 1000)
        audio_ms = int(audio.shape[0] * 1000 / max(1, self._segment_sample_rate))
        drop_reason = self._tail_hallucination_drop_reason(
            text,
            audio_ms=audio_ms,
            segment_stats=segment_stats,
        )
        if drop_reason:
            self._debug(f"v2 final dropped reason={drop_reason} text={text!r}")
            return
        self._adaptive_recent_audio_ms.append(max(0, audio_ms))
        self._adaptive_final_latencies.append(max(0, latency_ms))
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

    def _retry_empty_final_with_shorter_tail(self, *, result, audio: np.ndarray, sample_rate: int):
        if (result.text or "").strip():
            return result
        if audio.size == 0 or not self._last_partial_text.strip():
            return result

        retry_seconds = min(
            max(4, self._partial_history_seconds + 2),
            max(4, self._final_history_seconds // 2),
        )
        if retry_seconds >= self._final_history_seconds:
            return result

        retry_audio = self._limited_audio(retry_seconds)
        if retry_audio.size == 0 or retry_audio.shape[0] >= audio.shape[0]:
            return result

        retried = self._transcribe_backend(
            self._final_backend,
            method_name="transcribe_final",
            audio=retry_audio,
            sample_rate=sample_rate,
        )
        if (retried.text or "").strip():
            retry_ms = int(retry_audio.shape[0] * 1000 / max(1, sample_rate))
            self._debug(
                "v2 final recovered_with_short_tail "
                f"retry_seconds={retry_seconds} retry_audio_ms={retry_ms} text={retried.text.strip()!r}"
            )
            return retried
        return result

    @staticmethod
    def _merge_final_with_last_partial(*, final_text: str, last_partial_text: str) -> str:
        reason = SourceRuntimeV2._reason_to_prefer_last_partial(
            final_text=final_text,
            last_partial_text=last_partial_text,
        )
        if reason:
            return (last_partial_text or "").strip()
        return (final_text or "").strip()

    @staticmethod
    def _reason_to_prefer_last_partial(*, final_text: str, last_partial_text: str) -> str:
        final_clean = (final_text or "").strip()
        partial_clean = (last_partial_text or "").strip()
        if not partial_clean:
            return ""
        if not final_clean:
            return "empty-final"
        if final_clean == partial_clean:
            return ""

        final_compact = "".join(final_clean.split())
        partial_compact = "".join(partial_clean.split())
        if not final_compact or not partial_compact:
            return ""

        if SourceRuntimeV2._looks_like_repetitive_loop(final_clean):
            return "looped-final"

        containment_reason = SourceRuntimeV2._reason_partial_clearly_contains_final(
            final_clean=final_clean,
            partial_clean=partial_clean,
            final_compact=final_compact,
            partial_compact=partial_compact,
        )
        if containment_reason:
            return containment_reason

        return ""

    @staticmethod
    def _should_promote_empty_final_partial(partial_text: str) -> bool:
        value = (partial_text or "").strip()
        if not value:
            return False
        if SourceRuntimeV2._looks_like_repetitive_loop(value):
            return False

        compact = "".join(value.split())
        if len(compact) < 4:
            return False
        if SourceRuntimeV2._looks_like_short_cta_tail(value):
            return False

        cjk_count = sum(1 for ch in compact if "\u4e00" <= ch <= "\u9fff")
        if cjk_count:
            # Very short CJK fillers such as "好" / "謝謝" are often noise-driven
            # partials when the final decode returns empty. Do not promote them.
            return cjk_count >= 6

        words = [token for token in value.split() if token]
        if len(words) < 3 and len(compact) < 12:
            return False
        return True

    @staticmethod
    def _looks_like_short_cta_tail(text: str) -> bool:
        value = (text or "").strip()
        if not value:
            return False
        compact = "".join(value.split())
        if len(compact) > 24:
            return False
        return any(pattern.match(value) for pattern in _CJK_TAIL_HALLUCINATION_PATTERNS)

    @staticmethod
    def _reason_partial_clearly_contains_final(
        *,
        final_clean: str,
        partial_clean: str,
        final_compact: str,
        partial_compact: str,
    ) -> str:
        # Only trust the partial over the final when we have strong evidence that
        # the final is a truncated regression of the same utterance.
        if len(final_compact) >= len(partial_compact):
            return ""

        # Final shrank substantially but is still contained within the stable partial.
        if len(final_compact) < max(4, int(len(partial_compact) * 0.7)):
            if partial_compact.startswith(final_compact) or final_compact in partial_compact:
                return "final-substring-regression"

        # Final lost its prefix but still shares the same ending with the partial.
        probe_len = min(18, len(final_compact), len(partial_compact))
        if probe_len >= 6 and partial_compact.endswith(final_compact[-probe_len:]):
            if final_compact[-probe_len:] in partial_compact:
                return "missing-prefix-regression"

        # Final preserves a long common prefix but dropped a significant suffix.
        prefix_chars = 0
        for final_char, partial_char in zip(final_clean, partial_clean):
            if final_char != partial_char:
                break
            prefix_chars += 1
        if prefix_chars >= 6:
            shorter_len = min(len(final_clean), len(partial_clean))
            shared_ratio = prefix_chars / max(1, shorter_len)
            if shared_ratio >= 0.72 and len(final_clean) + 8 < len(partial_clean):
                return "missing-suffix-regression"

        return ""

    def _should_defer_early_final(
        self,
        *,
        now_ms: int,
        decision: StreamingDecision,
        signal: EndpointSignal,
        segment_audio_ms: int,
    ) -> bool:
        if not decision.is_early_final:
            return False
        if signal.hard_endpoint or signal.speech_ended:
            return False
        if self._last_partial_emit_ms <= 0 or not self._last_partial_text:
            return False
        if decision.reason not in {"pause_turn", "adaptive_length", "soft_endpoint"}:
            return False
        if now_ms < self._deferred_early_final_until_ms:
            return False
        pause_ms = float(signal.pause_ms)
        if pause_ms >= 240.0:
            return False
        if segment_audio_ms >= max(1600, int(self._base_soft_final_audio_ms * 0.55)):
            return False
        time_since_partial_ms = now_ms - self._last_partial_emit_ms
        if time_since_partial_ms > max(700, int(self._partial_interval_ms * 1.5)):
            return False
        hold_ms = max(100, min(220, int(round((300.0 - pause_ms) * 0.72))))
        self._deferred_early_final_until_ms = now_ms + hold_ms
        self._debug(
            "v2 defer early final "
            f"reason={decision.reason or 'early'} pause_ms={int(round(pause_ms))} "
            f"segment_ms={segment_audio_ms} hold_ms={hold_ms}"
        )
        return True

    @staticmethod
    def _looks_like_repetitive_loop(text: str) -> bool:
        tokens = [token for token in text.split() if token]
        if len(tokens) >= 8:
            for size in range(3, min(8, len(tokens) // 2 + 1)):
                chunk = tokens[-size:]
                prev = tokens[-size * 2 : -size]
                if prev == chunk:
                    return True
        compact = "".join(text.split())
        if len(compact) < 12:
            return False
        max_span = min(18, len(compact) // 2)
        for span in range(6, max_span + 1):
            suffix = compact[-span:]
            prev = compact[-span * 2 : -span]
            if prev == suffix:
                return True
        return False

    @staticmethod
    def _token_overlap_ratio(left: str, right: str) -> float:
        left_tokens = {token for token in left.split() if token}
        right_tokens = {token for token in right.split() if token}
        if left_tokens and right_tokens:
            intersection = len(left_tokens & right_tokens)
            return intersection / max(1, min(len(left_tokens), len(right_tokens)))
        prefix_chars = 0
        for left_char, right_char in zip(left, right):
            if left_char != right_char:
                break
            prefix_chars += 1
        return prefix_chars / max(1, min(len(left), len(right)))

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

    def _reset_segment_signal_stats(self) -> None:
        self._segment_signal_audio_ms = 0.0
        self._segment_signal_speech_ms = 0.0
        self._segment_signal_trailing_silence_ms = 0.0
        self._segment_signal_rms_weighted_sum = 0.0
        self._segment_signal_probability_weighted_sum = 0.0
        self._segment_signal_max_rms = 0.0
        self._segment_signal_max_probability = 0.0
        self._segment_signal_threshold = 0.0

    def _record_segment_signal(self, *, signal: EndpointSignal, chunk_ms: int) -> None:
        duration_ms = max(0.0, float(chunk_ms))
        if duration_ms <= 0.0:
            return
        probability = max(0.0, float(signal.speech_probability))
        threshold = max(0.0, float(signal.speech_threshold))
        rms = max(0.0, float(signal.rms))
        is_speech_frame = bool(signal.is_speech_frame) or (threshold > 0.0 and probability >= threshold)
        self._segment_signal_audio_ms += duration_ms
        self._segment_signal_rms_weighted_sum += rms * duration_ms
        self._segment_signal_probability_weighted_sum += probability * duration_ms
        self._segment_signal_max_rms = max(self._segment_signal_max_rms, rms)
        self._segment_signal_max_probability = max(self._segment_signal_max_probability, probability)
        self._segment_signal_threshold = max(self._segment_signal_threshold, threshold)
        if is_speech_frame:
            self._segment_signal_speech_ms += duration_ms
            self._segment_signal_trailing_silence_ms = 0.0
        else:
            self._segment_signal_trailing_silence_ms += duration_ms

    def _segment_signal_stats(self, *, audio: np.ndarray, sample_rate: int) -> SegmentSignalStats:
        audio_ms = int(round(self._segment_signal_audio_ms))
        if audio_ms <= 0 and audio.size > 0:
            audio_ms = int(audio.shape[0] * 1000 / max(1, sample_rate))
        speech_ms = int(round(self._segment_signal_speech_ms))
        speech_ratio = float(speech_ms) / float(max(1, audio_ms))
        mean_rms = self._segment_signal_rms_weighted_sum / max(1.0, self._segment_signal_audio_ms)
        mean_probability = self._segment_signal_probability_weighted_sum / max(1.0, self._segment_signal_audio_ms)
        return SegmentSignalStats(
            audio_ms=audio_ms,
            speech_ms=speech_ms,
            trailing_silence_ms=int(round(self._segment_signal_trailing_silence_ms)),
            speech_ratio=max(0.0, min(1.0, speech_ratio)),
            mean_rms=max(0.0, float(mean_rms)),
            max_rms=max(0.0, float(self._segment_signal_max_rms)),
            mean_probability=max(0.0, float(mean_probability)),
            max_probability=max(0.0, float(self._segment_signal_max_probability)),
            speech_threshold=max(0.0, float(self._segment_signal_threshold)),
        )

    def _segment_frontend_stats(self, segment_stats: SegmentSignalStats) -> dict[str, object]:
        stats = dict(self._frontend.stats())
        stats["segment_speech_ratio"] = round(segment_stats.speech_ratio, 4)
        stats["segment_audio_ms"] = segment_stats.audio_ms
        stats["segment_speech_ms"] = segment_stats.speech_ms
        stats["segment_trailing_silence_ms"] = segment_stats.trailing_silence_ms
        if segment_stats.audio_ms > 0:
            stats["speech_ratio"] = round(segment_stats.speech_ratio, 4)
        return stats

    @staticmethod
    def _tail_hallucination_drop_reason(
        text: str,
        *,
        audio_ms: int,
        segment_stats: SegmentSignalStats,
    ) -> str:
        value = (text or "").strip()
        if not value:
            return ""
        normalized = "".join(ch.lower() for ch in value if ch.isalnum())
        compact = re.sub(r"\s+", " ", value)
        effective_audio_ms = max(0, int(audio_ms), int(segment_stats.audio_ms))
        trailing_silence_ms = max(0, int(segment_stats.trailing_silence_ms))
        speech_ratio = max(0.0, min(1.0, float(segment_stats.speech_ratio)))
        mean_rms = max(0.0, float(segment_stats.mean_rms))
        max_rms = max(0.0, float(segment_stats.max_rms))

        if SourceRuntimeV2._looks_like_short_cta_tail(value):
            return "cjk-tail-hallucination"

        weak_speech = speech_ratio < 0.24
        weak_energy = mean_rms < 0.008 and max_rms < 0.030
        tail_dominant = trailing_silence_ms >= max(650, int(effective_audio_ms * 0.55))
        tiny_uncertain_segment = effective_audio_ms <= 650 and speech_ratio < 0.45
        weak_evidence = weak_speech or weak_energy or tail_dominant or tiny_uncertain_segment

        if normalized in _SHORT_TAIL_HALLUCINATION_NORMALIZED and weak_evidence:
            return "short-tail-hallucination"
        if any(pattern.match(compact) for pattern in _TAIL_HALLUCINATION_PATTERNS) and weak_evidence:
            return "short-tail-hallucination"
        if any(pattern.match(compact) for pattern in _CJK_TAIL_HALLUCINATION_PATTERNS) and (
            weak_evidence or effective_audio_ms <= 1200
        ):
            return "cjk-tail-hallucination"
        return ""

    def _debug(self, message: str) -> None:
        with self._stats_lock:
            self._last_debug = message
        if self._on_debug:
            self._on_debug(message)

    def _backend_runtime_info(self) -> dict[str, object]:
        final_info = self._single_backend_runtime_info(self._final_backend)
        partial_info = self._single_backend_runtime_info(self._partial_backend)
        return {
            "device_effective": str(final_info.get("device_effective", "")),
            "model_init_mode": str(final_info.get("model_init_mode", "lazy")),
            "init_failure": str(final_info.get("init_failure", "")),
            "runtime_label": str(final_info.get("runtime_label", "")),
            "postprocessor": {
                "partial": partial_info.get("postprocessor", {}),
                "final": final_info.get("postprocessor", {}),
            },
            "partial_backend_runtime": partial_info,
            "final_backend_runtime": final_info,
        }

    @staticmethod
    def _single_backend_runtime_info(backend: object) -> dict[str, object]:
        info_fn = getattr(backend, "runtime_info", None)
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
        frontend_stats: dict[str, object] | None = None,
    ):
        method = getattr(backend, method_name)
        effective_frontend_stats = dict(self._frontend.stats())
        if frontend_stats:
            effective_frontend_stats.update(frontend_stats)
        try:
            return method(audio, sample_rate, frontend_stats=effective_frontend_stats)
        except TypeError as exc:
            if "frontend_stats" not in str(exc):
                raise
            return method(audio, sample_rate)

    def _adaptive_length_limit_ms(self, *, signal: EndpointSignal) -> int:
        if signal.pause_ms >= 280.0:
            return max(self._adaptive_length_floor_ms, 2800)
        if signal.pause_ms >= 180.0:
            return max(self._adaptive_length_floor_ms, 3800)
        if signal.pause_ms >= 90.0:
            return max(self._adaptive_length_floor_ms, 5200)
        return self._adaptive_length_ceiling_ms

    def _recompute_adaptive_tuning(self, *, now_ms: int, segment_audio_ms: int, signal: EndpointSignal) -> None:
        if not self._adaptive_enabled:
            return

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
        avg_final_audio_ms = (
            sum(self._adaptive_recent_audio_ms) / len(self._adaptive_recent_audio_ms)
            if self._adaptive_recent_audio_ms
            else 0.0
        )

        next_partial_interval_ms = self._base_partial_interval_ms
        next_soft_final_audio_ms = self._base_soft_final_audio_ms
        next_soft_endpoint_finalize_audio_ms = self._base_soft_endpoint_finalize_audio_ms
        next_speech_end_finalize_audio_ms = self._base_speech_end_finalize_audio_ms

        if avg_final_audio_ms and avg_final_audio_ms <= 1600:
            next_partial_interval_ms = max(self._partial_interval_floor_ms, self._base_partial_interval_ms - 140)
            next_soft_final_audio_ms = max(self._force_final_audio_ms, int(round(self._base_soft_final_audio_ms * 0.76)))
        elif avg_final_audio_ms >= 3200:
            next_partial_interval_ms = max(self._partial_interval_floor_ms, self._base_partial_interval_ms - 40)
            next_soft_final_audio_ms = max(self._force_final_audio_ms, int(round(self._base_soft_final_audio_ms * 0.70)))
        elif avg_final_audio_ms >= 2400:
            next_soft_final_audio_ms = max(self._force_final_audio_ms, int(round(self._base_soft_final_audio_ms * 0.80)))

        if avg_partial_latency_ms >= 850 or avg_final_latency_ms >= 1400:
            next_partial_interval_ms = max(next_partial_interval_ms, self._base_partial_interval_ms + 120)
            next_soft_final_audio_ms = max(self._force_final_audio_ms, int(round(self._base_soft_final_audio_ms * 0.72)))

        if self._last_degradation_level == DEGRADATION_CONGESTED:
            next_partial_interval_ms = max(next_partial_interval_ms, self._base_partial_interval_ms + 120)
        elif self._last_degradation_level == DEGRADATION_DEGRADED:
            next_partial_interval_ms = max(next_partial_interval_ms, self._base_partial_interval_ms + 200)
            next_soft_final_audio_ms = max(self._force_final_audio_ms, int(round(self._base_soft_final_audio_ms * 0.68)))

        if signal.pause_ms >= 160.0 and segment_audio_ms >= max(1800, int(self._base_soft_final_audio_ms * 0.7)):
            next_soft_final_audio_ms = min(
                next_soft_final_audio_ms,
                max(self._force_final_audio_ms, int(round(self._base_soft_final_audio_ms * 0.78))),
            )

        next_partial_interval_ms = max(self._partial_interval_floor_ms, int(next_partial_interval_ms))
        next_soft_final_audio_ms = max(self._force_final_audio_ms, int(next_soft_final_audio_ms))
        next_soft_endpoint_finalize_audio_ms = min(
            next_soft_final_audio_ms,
            max(
                self._force_final_audio_ms,
                int(round(self._base_soft_endpoint_finalize_audio_ms * (next_soft_final_audio_ms / max(1, self._base_soft_final_audio_ms)))),
            ),
        )
        next_speech_end_finalize_audio_ms = min(
            next_soft_endpoint_finalize_audio_ms,
            max(
                900,
                int(round(self._base_speech_end_finalize_audio_ms * (next_soft_final_audio_ms / max(1, self._base_soft_final_audio_ms)))),
            ),
        )

        changed = any(
            (
                next_partial_interval_ms != self._partial_interval_ms,
                next_soft_final_audio_ms != self._soft_final_audio_ms,
                next_soft_endpoint_finalize_audio_ms != self._soft_endpoint_finalize_audio_ms,
                next_speech_end_finalize_audio_ms != self._speech_end_finalize_audio_ms,
            )
        )
        self._partial_interval_ms = next_partial_interval_ms
        self._soft_final_audio_ms = next_soft_final_audio_ms
        self._soft_endpoint_finalize_audio_ms = next_soft_endpoint_finalize_audio_ms
        self._speech_end_finalize_audio_ms = next_speech_end_finalize_audio_ms
        if changed:
            self._debug(
                "v2 adaptive "
                f"partial_ms={self._partial_interval_ms} "
                f"soft_final_ms={self._soft_final_audio_ms} "
                f"soft_finalize_ms={self._soft_endpoint_finalize_audio_ms} "
                f"speech_end_finalize_ms={self._speech_end_finalize_audio_ms}"
            )


__all__ = ["V2RuntimeEvent", "SourceRuntimeV2", "SourceRuntimeV2Stats"]
