"""Minimal compatibility helpers around the v2-only ASR runtime.

This module intentionally keeps only the small API surface still referenced by
tests and compatibility imports.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Callable
import re
import time
import unicodedata

import numpy as np


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
    """Small compatibility stub for tests and old imports.

    The old decode thread and endpointing loop are gone.  This class only keeps
    the helper methods and adaptive tuning state that existing tests still
    assert against.
    """

    def __init__(
        self,
        *,
        engine,
        vad,
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
        speaker_diarizer=None,
        on_event: Callable[[AsrEvent], None] | None = None,
        on_debug: Callable[[str], None] | None = None,
    ) -> None:
        self._engine = engine
        self._vad = vad
        self._speaker_diarizer = speaker_diarizer
        self._on_event = on_event
        self._on_debug = on_debug
        self._queue_maxsize = max(4, int(queue_maxsize))
        self._partial_history_seconds = max(1, int(partial_history_seconds))
        self._final_history_seconds = max(1, int(final_history_seconds))
        self._pre_roll_ms = max(0, int(pre_roll_ms))
        self._min_partial_audio_ms = max(120, int(min_partial_audio_ms))
        self._partial_interval_floor_ms = max(120, int(partial_interval_floor_ms))
        self._base_partial_interval_ms = max(self._partial_interval_floor_ms, int(partial_interval_ms))
        self._partial_interval_ms = self._base_partial_interval_ms
        self._base_min_silence_duration_ms = int(
            max(120.0, float(getattr(vad, "effective_min_silence_duration_ms", 240.0)))
        )
        self._force_final_audio_ms = 1800
        self._base_soft_final_audio_ms = max(self._force_final_audio_ms, int(soft_final_audio_ms))
        self._soft_final_audio_ms = self._base_soft_final_audio_ms
        self._adaptive_enabled = bool(adaptive_enabled)
        self._adaptive_mode = "baseline"
        self._adaptive_recent_audio_ms: deque[int] = deque(maxlen=10)
        self._adaptive_partial_latencies: deque[int] = deque(maxlen=10)
        self._adaptive_final_latencies: deque[int] = deque(maxlen=10)
        self._adaptive_cps_samples: deque[float] = deque(maxlen=8)
        self._adaptive_load_backoff_until_ms = 0
        self._soft_split_pause_ms = 220.0
        self._segment_chunks: list[np.ndarray] = []
        self._segment_sample_rate = 16000
        self._segment_start_ms = 0
        self._segment_end_ms = 0
        self._last_partial_ms = 0
        self._drop_partial_until_final = False
        self._in_speech_segment = False
        self._pre_roll_chunks: deque[np.ndarray] = deque()
        self._pre_roll_sample_count = 0
        self._pre_roll_sample_rate = 16000
        self._overflow_count = 0
        self._partial_count = 0
        self._final_count = 0
        self._last_debug = ""
        self._stats_lock = Lock()
        self._early_final_enabled = bool(early_final_enabled)

        self._apply_adaptive_tuning(
            mode="baseline",
            partial_interval_ms=self._base_partial_interval_ms,
            min_silence_duration_ms=self._base_min_silence_duration_ms,
            soft_final_audio_ms=self._base_soft_final_audio_ms,
            force=True,
        )

    def start(self, on_event: Callable[[AsrEvent], None]) -> None:
        self._on_event = on_event

    def request_stop(self) -> None:
        return None

    def is_running(self) -> bool:
        return False

    def cleanup_if_stopped(self) -> bool:
        return True

    def stop(self) -> None:
        return None

    def submit_chunk(self, chunk: np.ndarray, sample_rate: float) -> None:
        if self._pre_roll_ms > 0:
            self._append_pre_roll_chunk(chunk=np.asarray(chunk, dtype=np.float32), sample_rate=sample_rate)

    def stats(self) -> StreamingAsrStats:
        with self._stats_lock:
            return StreamingAsrStats(
                queue_size=0,
                dropped_chunks=self._overflow_count,
                partial_count=self._partial_count,
                final_count=self._final_count,
                last_debug=self._last_debug,
                vad_rms=float(getattr(self._vad, "last_rms", 0.0)),
                vad_threshold=float(getattr(self._vad, "effective_rms_threshold", 0.0)),
                adaptive_mode=self._adaptive_mode,
                adaptive_partial_interval_ms=self._partial_interval_ms,
                adaptive_min_silence_duration_ms=int(
                    getattr(self._vad, "effective_min_silence_duration_ms", self._base_min_silence_duration_ms)
                ),
                adaptive_soft_final_audio_ms=self._soft_final_audio_ms,
            )

    def _should_emit_soft_split(self, decision) -> bool:
        if not self._segment_chunks:
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

    def _record_final_adaptation(self, *, audio_ms: int, latency_ms: int, now_ms: int, cps: float = 0.0) -> None:
        if not self._adaptive_enabled:
            return
        self._adaptive_recent_audio_ms.append(max(0, int(audio_ms)))
        self._adaptive_final_latencies.append(max(0, int(latency_ms)))
        if cps > 0.0:
            self._adaptive_cps_samples.append(float(cps))
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
        avg_cps = (
            sum(self._adaptive_cps_samples) / len(self._adaptive_cps_samples)
            if self._adaptive_cps_samples
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

        if avg_cps > 8.0 and "short_turn" in mode_parts:
            min_silence_duration_ms = max(160, int(min_silence_duration_ms * 0.85))
            mode_parts.append("fast_speaker")

        self._apply_adaptive_tuning(
            mode="+".join(mode_parts) if mode_parts else "baseline",
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
        copied = np.asarray(chunk, dtype=np.float32).copy()
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
    "謝謝大家",
    "谢谢大家",
    "感謝大家",
    "感谢大家",
    "晚安",
    "感謝您的收看",
    "感谢您的收看",
}

_NON_SPEECH_TEXT_PATTERNS = (
    re.compile(r"yoyo\s+television\s+series\s+exclusive", re.IGNORECASE),
    re.compile(r"amara\.org", re.IGNORECASE),
    re.compile(r"點贊.*訂閱.*轉發.*打賞", re.IGNORECASE),
    re.compile(r"ming\s+pao\s+canada\s+ming\s+pao\s+toronto", re.IGNORECASE),
    re.compile(r"謝謝觀看.*下次見", re.IGNORECASE),
)

_NON_SPEECH_NORMALIZED_SUBSTRINGS = (
    "yoyotelevisionseriesexclusive",
    "amaraorg",
    "點贊訂閱轉發打賞",
    "mingpaocanadamingpaotoronto",
    "謝謝觀看下次見",
)


def _format_asr_exception_message(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    lowered = message.lower()
    if "dll load failed while importing _ssl" in lowered:
        return (
            f"{message} OpenSSL runtime is missing. "
            "Check _internal/libssl-3-x64.dll, _internal/libcrypto-3-x64.dll, "
            "_internal/_ssl.pyd, and dist/SyncTranslate-onedir/SyncTranslate.exe."
        )
    return message


def _looks_like_silence_hallucination(text: str, *, audio_ms: int, vad_rms: float) -> bool:
    value = (text or "").strip()
    if not value:
        return False
    if audio_ms > 1800 or vad_rms >= 0.035:
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


__all__ = [
    "AsrEvent",
    "StreamingAsr",
    "StreamingAsrStats",
    "_format_asr_exception_message",
    "_looks_like_known_non_speech_text",
    "_looks_like_script_mismatch_junk",
    "_looks_like_silence_hallucination",
    "_transcript_drop_reason",
]
