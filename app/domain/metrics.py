"""Runtime metrics 領域模型。"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass(slots=True)
class UtteranceLatency:
    """單一 utterance 的延遲追蹤記錄。"""
    source: str = ""
    utterance_id: str = ""
    backend_name: str = ""
    asr_first_partial_ms: int = 0
    asr_final_ms: int = 0
    llm_final_ms: int = 0
    tts_enqueue_ms: int = 0
    tts_playback_ms: int = 0
    created_at: float = field(default_factory=time.time)

    def total_e2e_ms(self) -> int:
        """從 created_at 到 TTS 入隊的端到端延遲（若有）。"""
        if self.tts_enqueue_ms:
            return self.tts_enqueue_ms
        if self.llm_final_ms:
            return self.llm_final_ms
        return self.asr_final_ms


@dataclass(slots=True)
class PipelineMetrics:
    """單一 session 的累計 pipeline metrics。"""
    partial_count: int = 0
    final_count: int = 0
    translation_count: int = 0
    tts_enqueue_count: int = 0
    dropped_asr_events: int = 0
    dropped_translation_events: int = 0
    degradation_events: int = 0


class MetricsCollector:
    """收集並彙整 pipeline metrics。

    Thread-safe。可由多個模組同時更新。
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._metrics = PipelineMetrics()
        self._utterances: dict[str, UtteranceLatency] = {}

    def record_partial(self, source: str, utterance_id: str, *, latency_ms: int = 0, backend_name: str = "") -> None:
        with self._lock:
            self._metrics.partial_count += 1
            rec = self._utterances.setdefault(
                utterance_id,
                UtteranceLatency(source=source, utterance_id=utterance_id, backend_name=backend_name),
            )
            if not rec.asr_first_partial_ms and latency_ms:
                rec.asr_first_partial_ms = latency_ms

    def record_final(self, source: str, utterance_id: str, *, latency_ms: int = 0, backend_name: str = "") -> None:
        with self._lock:
            self._metrics.final_count += 1
            rec = self._utterances.setdefault(
                utterance_id,
                UtteranceLatency(source=source, utterance_id=utterance_id, backend_name=backend_name),
            )
            if latency_ms:
                rec.asr_final_ms = latency_ms

    def record_translation(self, utterance_id: str, *, latency_ms: int = 0) -> None:
        with self._lock:
            self._metrics.translation_count += 1
            if utterance_id in self._utterances and latency_ms:
                self._utterances[utterance_id].llm_final_ms = latency_ms

    def record_tts_enqueue(self, utterance_id: str, *, latency_ms: int = 0) -> None:
        with self._lock:
            self._metrics.tts_enqueue_count += 1
            if utterance_id in self._utterances and latency_ms:
                self._utterances[utterance_id].tts_enqueue_ms = latency_ms

    def record_tts_playback(self, utterance_id: str, *, latency_ms: int = 0) -> None:
        with self._lock:
            if utterance_id in self._utterances and latency_ms:
                self._utterances[utterance_id].tts_playback_ms = latency_ms

    def record_dropped_asr(self) -> None:
        with self._lock:
            self._metrics.dropped_asr_events += 1

    def record_dropped_translation(self) -> None:
        with self._lock:
            self._metrics.dropped_translation_events += 1

    def record_degradation(self) -> None:
        with self._lock:
            self._metrics.degradation_events += 1

    def snapshot(self) -> dict[str, Any]:
        """回傳當前 metrics 快照（dict）。"""
        with self._lock:
            recent_latencies = [
                {
                    "utterance_id": u.utterance_id[:8],
                    "source": u.source,
                    "backend": u.backend_name,
                    "asr_first_partial_ms": u.asr_first_partial_ms,
                    "asr_final_ms": u.asr_final_ms,
                    "llm_final_ms": u.llm_final_ms,
                    "tts_enqueue_ms": u.tts_enqueue_ms,
                }
                for u in list(self._utterances.values())[-16:]
            ]
            return {
                "partial_count": self._metrics.partial_count,
                "final_count": self._metrics.final_count,
                "translation_count": self._metrics.translation_count,
                "tts_enqueue_count": self._metrics.tts_enqueue_count,
                "dropped_asr_events": self._metrics.dropped_asr_events,
                "dropped_translation_events": self._metrics.dropped_translation_events,
                "degradation_events": self._metrics.degradation_events,
                "recent_utterances": recent_latencies,
            }

    def reset(self) -> None:
        with self._lock:
            self._metrics = PipelineMetrics()
            self._utterances.clear()


__all__ = ["UtteranceLatency", "PipelineMetrics", "MetricsCollector"]
