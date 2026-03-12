from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Callable

import numpy as np

from app.local_ai.faster_whisper_engine import FasterWhisperEngine
from app.local_ai.streaming_asr import AsrEvent, StreamingAsr
from app.local_ai.vad_segmenter import VadConfig, VadSegmenter
from app.schemas import AppConfig


@dataclass(slots=True)
class ASREventWithSource:
    source: str
    text: str
    is_final: bool
    start_ms: int
    end_ms: int
    latency_ms: int


class ASRManager:
    def __init__(self, config: AppConfig, on_error: Callable[[str], None] | None = None) -> None:
        self._config = config
        self._on_error = on_error
        self._streams: dict[str, StreamingAsr] = {}
        self._enabled = {"local": True, "remote": True}
        self._lock = Lock()

    def start(self, source: str, on_event: Callable[[ASREventWithSource], None]) -> None:
        stream = self._stream_of(source)

        def _wrapped(event: AsrEvent) -> None:
            on_event(
                ASREventWithSource(
                    source=source,
                    text=event.text,
                    is_final=event.is_final,
                    start_ms=event.start_ms,
                    end_ms=event.end_ms,
                    latency_ms=event.latency_ms,
                )
            )

        stream.start(_wrapped)

    def stop(self, source: str) -> None:
        stream = self._streams.get(source)
        if stream:
            stream.stop()

    def stop_all(self) -> None:
        for stream in self._streams.values():
            stream.stop()

    def submit(self, source: str, chunk: np.ndarray, sample_rate: float) -> None:
        if not self.is_enabled(source):
            return
        stream = self._stream_of(source)
        stream.submit_chunk(chunk, sample_rate)

    def set_enabled(self, source: str, enabled: bool) -> None:
        key = source if source in ("local", "remote") else "local"
        with self._lock:
            self._enabled[key] = bool(enabled)

    def is_enabled(self, source: str) -> bool:
        key = source if source in ("local", "remote") else "local"
        with self._lock:
            return self._enabled.get(key, False)

    def stats(self) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for source in ("local", "remote"):
            stream = self._streams.get(source)
            if not stream:
                result[source] = {
                    "queue_size": 0,
                    "dropped_chunks": 0,
                    "partial_count": 0,
                    "final_count": 0,
                    "last_debug": "",
                    "vad_rms": 0.0,
                    "vad_threshold": 0.0,
                }
                continue
            stats = stream.stats()
            result[source] = {
                "queue_size": stats.queue_size,
                "dropped_chunks": stats.dropped_chunks,
                "partial_count": stats.partial_count,
                "final_count": stats.final_count,
                "last_debug": stats.last_debug,
                "vad_rms": stats.vad_rms,
                "vad_threshold": stats.vad_threshold,
            }
        return result

    def _stream_of(self, source: str) -> StreamingAsr:
        key = source if source in ("local", "remote") else "local"
        if key in self._streams:
            return self._streams[key]

        language = self._config.language.local_source if key == "local" else self._config.language.meeting_source
        asr_cfg = self._config.asr
        runtime_cfg = self._config.runtime
        vad_cfg = asr_cfg.vad
        stream_cfg = asr_cfg.streaming

        stream = StreamingAsr(
            engine=FasterWhisperEngine(
                model=asr_cfg.model,
                device=asr_cfg.device,
                compute_type=asr_cfg.compute_type,
                beam_size=asr_cfg.beam_size,
                condition_on_previous_text=asr_cfg.condition_on_previous_text,
                language=language,
            ),
            vad=VadSegmenter(
                VadConfig(
                    enabled=vad_cfg.enabled,
                    min_speech_duration_ms=vad_cfg.min_speech_duration_ms,
                    min_silence_duration_ms=vad_cfg.min_silence_duration_ms,
                    max_speech_duration_s=vad_cfg.max_speech_duration_s,
                    speech_pad_ms=vad_cfg.speech_pad_ms,
                    rms_threshold=vad_cfg.rms_threshold,
                )
            ),
            partial_interval_ms=stream_cfg.partial_interval_ms,
            partial_history_seconds=stream_cfg.partial_history_seconds,
            final_history_seconds=stream_cfg.final_history_seconds,
            queue_maxsize=runtime_cfg.asr_queue_maxsize,
            on_debug=self._on_error,
        )
        self._streams[key] = stream
        return stream
