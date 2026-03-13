from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from threading import Lock
from typing import Callable
from uuid import uuid4

import numpy as np

from app.local_ai.faster_whisper_engine import FasterWhisperEngine
from app.local_ai.streaming_asr import AsrEvent, StreamingAsr
from app.local_ai.vad_segmenter import VadConfig, VadSegmenter
from app.events import ErrorEvent
from app.schemas import AppConfig


@dataclass(slots=True)
class ASREventWithSource:
    source: str
    utterance_id: str
    revision: int
    pipeline_revision: int
    config_fingerprint: str
    created_at: float
    text: str
    is_final: bool
    start_ms: int
    end_ms: int
    latency_ms: int


class ASRManager:
    def __init__(
        self,
        config: AppConfig,
        on_error: Callable[[str | ErrorEvent], None] | None = None,
        *,
        pipeline_revision: int = 1,
    ) -> None:
        self._config = config
        self._on_error = on_error
        self._streams: dict[str, StreamingAsr] = {}
        self._stream_fingerprints: dict[str, str] = {}
        self._enabled = {"local": True, "remote": True}
        self._lock = Lock()
        self._active_utterance: dict[str, str | None] = {"local": None, "remote": None}
        self._revision: dict[str, int] = {"local": 0, "remote": 0}
        self._pipeline_revision = max(1, int(pipeline_revision))
        self._runtime_fingerprint = self._build_runtime_fingerprint()

    def configure_pipeline(self, config: AppConfig, pipeline_revision: int) -> None:
        self._config = config
        self._pipeline_revision = max(1, int(pipeline_revision))
        new_fingerprint = self._build_runtime_fingerprint()
        if new_fingerprint == self._runtime_fingerprint:
            return
        self._runtime_fingerprint = new_fingerprint
        for stream in self._streams.values():
            stream.stop()
        self._streams.clear()
        self._stream_fingerprints.clear()

    def start(self, source: str, on_event: Callable[[ASREventWithSource], None]) -> None:
        stream = self._stream_of(source)

        def _wrapped(event: AsrEvent) -> None:
            key = source if source in ("local", "remote") else "local"
            with self._lock:
                utterance_id = self._active_utterance.get(key)
                if not utterance_id:
                    utterance_id = uuid4().hex
                    self._active_utterance[key] = utterance_id
                    self._revision[key] = 0
                revision = self._revision.get(key, 0) + 1
                self._revision[key] = revision

            on_event(
                ASREventWithSource(
                    source=key,
                    utterance_id=utterance_id,
                    revision=revision,
                    pipeline_revision=self._pipeline_revision,
                    config_fingerprint=self._runtime_fingerprint,
                    created_at=time.time(),
                    text=event.text,
                    is_final=event.is_final,
                    start_ms=event.start_ms,
                    end_ms=event.end_ms,
                    latency_ms=event.latency_ms,
                )
            )

            if event.is_final:
                with self._lock:
                    self._active_utterance[key] = None
                    self._revision[key] = 0

        stream.start(_wrapped)

    def stop(self, source: str) -> None:
        stream = self._streams.get(source)
        if stream:
            stream.stop()

    def stop_all(self) -> None:
        for stream in self._streams.values():
            stream.stop()
        with self._lock:
            self._active_utterance = {"local": None, "remote": None}
            self._revision = {"local": 0, "remote": 0}

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
        fingerprint = self._build_stream_fingerprint(key)
        existing = self._streams.get(key)
        if existing and self._stream_fingerprints.get(key) == fingerprint:
            return existing
        if existing:
            existing.stop()
            self._streams.pop(key, None)
            self._stream_fingerprints.pop(key, None)

        language = self._config.language.local_source if key == "local" else self._config.language.meeting_source
        asr_cfg = self._asr_profile_for_language(language)
        runtime_cfg = self._config.runtime
        vad_cfg = asr_cfg.vad
        stream_cfg = asr_cfg.streaming
        model_name = self._resolve_model_for_language(asr_cfg.model, language)
        queue_maxsize = self._asr_queue_maxsize_for_language(language)

        stream = StreamingAsr(
            engine=FasterWhisperEngine(
                model=model_name,
                device=asr_cfg.device,
                compute_type=asr_cfg.compute_type,
                beam_size=asr_cfg.beam_size,
                condition_on_previous_text=asr_cfg.condition_on_previous_text,
                temperature_fallback=asr_cfg.temperature_fallback,
                no_speech_threshold=asr_cfg.no_speech_threshold,
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
            queue_maxsize=queue_maxsize,
            on_debug=self._on_error,
        )
        self._streams[key] = stream
        self._stream_fingerprints[key] = fingerprint
        return stream

    def _asr_profile_for_language(self, language: str):
        normalized = (language or "").strip().lower()
        if "-" in normalized:
            normalized = normalized.split("-", 1)[0]
        # local profile = 中文, remote profile = 英文
        if normalized.startswith("en"):
            return self._config.asr_channels.remote
        return self._config.asr_channels.local

    def _asr_queue_maxsize_for_language(self, language: str) -> int:
        normalized = (language or "").strip().lower()
        if "-" in normalized:
            normalized = normalized.split("-", 1)[0]
        runtime = self._config.runtime
        if normalized.startswith("en"):
            value = int(getattr(runtime, "asr_queue_maxsize_remote", runtime.asr_queue_maxsize))
        else:
            value = int(getattr(runtime, "asr_queue_maxsize_local", runtime.asr_queue_maxsize))
        return max(8, value)

    def _resolve_model_for_language(self, model: str, language: str) -> str:
        normalized_model = (model or "").strip()
        normalized_lang = (language or "").strip().lower()
        if "-" in normalized_lang:
            normalized_lang = normalized_lang.split("-", 1)[0]
        if normalized_model == "distil-large-v3" and normalized_lang in {"zh", "ja"}:
            if self._on_error:
                self._on_error(
                    f"asr model fallback: distil-large-v3 is weak for {normalized_lang}, use large-v3 instead"
                )
            return "large-v3"
        return normalized_model or "large-v3"

    def _build_runtime_fingerprint(self) -> str:
        payload = {
            "asr": {
                "local": {
                    "model": self._config.asr_channels.local.model,
                    "device": self._config.asr_channels.local.device,
                    "compute_type": self._config.asr_channels.local.compute_type,
                    "beam_size": self._config.asr_channels.local.beam_size,
                    "condition_on_previous_text": self._config.asr_channels.local.condition_on_previous_text,
                    "temperature_fallback": self._config.asr_channels.local.temperature_fallback,
                    "no_speech_threshold": self._config.asr_channels.local.no_speech_threshold,
                    "vad": {
                        "enabled": self._config.asr_channels.local.vad.enabled,
                        "min_speech_duration_ms": self._config.asr_channels.local.vad.min_speech_duration_ms,
                        "min_silence_duration_ms": self._config.asr_channels.local.vad.min_silence_duration_ms,
                        "max_speech_duration_s": self._config.asr_channels.local.vad.max_speech_duration_s,
                        "speech_pad_ms": self._config.asr_channels.local.vad.speech_pad_ms,
                        "rms_threshold": self._config.asr_channels.local.vad.rms_threshold,
                    },
                    "streaming": {
                        "partial_interval_ms": self._config.asr_channels.local.streaming.partial_interval_ms,
                        "partial_history_seconds": self._config.asr_channels.local.streaming.partial_history_seconds,
                        "final_history_seconds": self._config.asr_channels.local.streaming.final_history_seconds,
                    },
                },
                "remote": {
                    "model": self._config.asr_channels.remote.model,
                    "device": self._config.asr_channels.remote.device,
                    "compute_type": self._config.asr_channels.remote.compute_type,
                    "beam_size": self._config.asr_channels.remote.beam_size,
                    "condition_on_previous_text": self._config.asr_channels.remote.condition_on_previous_text,
                    "temperature_fallback": self._config.asr_channels.remote.temperature_fallback,
                    "no_speech_threshold": self._config.asr_channels.remote.no_speech_threshold,
                    "vad": {
                        "enabled": self._config.asr_channels.remote.vad.enabled,
                        "min_speech_duration_ms": self._config.asr_channels.remote.vad.min_speech_duration_ms,
                        "min_silence_duration_ms": self._config.asr_channels.remote.vad.min_silence_duration_ms,
                        "max_speech_duration_s": self._config.asr_channels.remote.vad.max_speech_duration_s,
                        "speech_pad_ms": self._config.asr_channels.remote.vad.speech_pad_ms,
                        "rms_threshold": self._config.asr_channels.remote.vad.rms_threshold,
                    },
                    "streaming": {
                        "partial_interval_ms": self._config.asr_channels.remote.streaming.partial_interval_ms,
                        "partial_history_seconds": self._config.asr_channels.remote.streaming.partial_history_seconds,
                        "final_history_seconds": self._config.asr_channels.remote.streaming.final_history_seconds,
                    },
                },
            },
            "language": {
                "local_source": self._config.language.local_source,
                "meeting_source": self._config.language.meeting_source,
            },
            "runtime": {
                "asr_queue_maxsize_local": self._config.runtime.asr_queue_maxsize_local,
                "asr_queue_maxsize_remote": self._config.runtime.asr_queue_maxsize_remote,
            },
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _build_stream_fingerprint(self, source: str) -> str:
        return f"{self._runtime_fingerprint}:{source}:rev={self._pipeline_revision}"
