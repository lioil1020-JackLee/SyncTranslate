"""PipelineMetricsCollector — per-utterance latency tracking.

Tracks the journey of each utterance through the pipeline:
  ASR first partial → ASR final → LLM final → TTS enqueue → TTS playback start.

This is extracted from AudioRouter's latency tracking to make it independently
testable and reusable.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass
class UtteranceMilestones:
    """Timing milestones for one utterance."""
    source: str
    utterance_id: str
    started_at: float = field(default_factory=time.monotonic)
    asr_first_partial_at: float | None = None
    asr_final_at: float | None = None
    llm_final_at: float | None = None
    tts_enqueue_at: float | None = None
    tts_playback_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        now = time.monotonic()
        return {
            "source": self.source,
            "utterance_id": self.utterance_id,
            "asr_first_partial_ms": round((self.asr_first_partial_at - self.started_at) * 1000) if self.asr_first_partial_at else None,
            "asr_final_ms": round((self.asr_final_at - self.started_at) * 1000) if self.asr_final_at else None,
            "llm_final_ms": round((self.llm_final_at - self.started_at) * 1000) if self.llm_final_at else None,
            "tts_enqueue_ms": round((self.tts_enqueue_at - self.started_at) * 1000) if self.tts_enqueue_at else None,
            "tts_playback_ms": round((self.tts_playback_at - self.started_at) * 1000) if self.tts_playback_at else None,
        }


class PipelineMetricsCollector:
    """Thread-safe per-utterance latency tracker with a rolling window.

    Parameters
    ----------
    window_size:
        Maximum number of recent utterance records to keep in memory.
    """

    def __init__(self, *, window_size: int = 32) -> None:
        self._lock = Lock()
        self._in_flight: dict[tuple[str, str], UtteranceMilestones] = {}
        self._recent: deque[dict[str, Any]] = deque(maxlen=max(4, window_size))

    # ------------------------------------------------------------------
    # Record milestones
    # ------------------------------------------------------------------

    def record_asr_partial(self, *, source: str, utterance_id: str) -> None:
        key = (source, utterance_id)
        with self._lock:
            if key not in self._in_flight:
                self._in_flight[key] = UtteranceMilestones(source=source, utterance_id=utterance_id)
            u = self._in_flight[key]
            if u.asr_first_partial_at is None:
                u.asr_first_partial_at = time.monotonic()

    def record_asr_final(self, *, source: str, utterance_id: str) -> None:
        key = (source, utterance_id)
        with self._lock:
            if key not in self._in_flight:
                self._in_flight[key] = UtteranceMilestones(source=source, utterance_id=utterance_id)
            self._in_flight[key].asr_final_at = time.monotonic()

    def record_llm_final(self, *, source: str, utterance_id: str) -> None:
        key = (source, utterance_id)
        with self._lock:
            if key in self._in_flight:
                self._in_flight[key].llm_final_at = time.monotonic()

    def record_tts_enqueue(self, *, source: str, utterance_id: str) -> None:
        key = (source, utterance_id)
        with self._lock:
            if key in self._in_flight:
                self._in_flight[key].tts_enqueue_at = time.monotonic()

    def record_tts_playback(self, *, source: str, utterance_id: str) -> None:
        key = (source, utterance_id)
        with self._lock:
            if key in self._in_flight:
                u = self._in_flight.pop(key)
                u.tts_playback_at = time.monotonic()
                self._recent.append(u.to_dict())
            # If no in-flight record, ignore (TTS may have been enqueued before collector started)

    def finalize_utterance(self, *, source: str, utterance_id: str) -> None:
        """Explicitly retire an utterance (e.g. when TTS is not used)."""
        key = (source, utterance_id)
        with self._lock:
            if key in self._in_flight:
                u = self._in_flight.pop(key)
                self._recent.append(u.to_dict())

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def recent_latencies(self) -> list[dict[str, Any]]:
        """Return the most recent completed utterance latency records."""
        with self._lock:
            return list(self._recent)

    def in_flight_count(self) -> int:
        with self._lock:
            return len(self._in_flight)

    def reset(self) -> None:
        with self._lock:
            self._in_flight.clear()
            self._recent.clear()


__all__ = ["PipelineMetricsCollector", "UtteranceMilestones"]
