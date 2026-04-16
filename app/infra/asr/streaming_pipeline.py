"""Legacy Whisper-based ASR pipeline.

This module is frozen for compatibility while ASR v2 is being built. New ASR
capabilities should be added under the v2 modules and wired through the ASR
factory instead of extending this pipeline further.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from threading import Lock
from typing import Callable
from uuid import uuid4

import numpy as np

from app.infra.asr.contracts import ASREventWithSource
from app.domain.events import ErrorEvent
from app.infra.asr.faster_whisper_adapter import FasterWhisperEngine
from app.infra.asr.language_policy import VadSegmenter
from app.infra.asr.speaker_diarizer import OnlineSpeakerDiarizer
from app.infra.asr.stream_worker import AsrEvent, StreamingAsr
from app.infra.config.schema import AppConfig


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
        self._callbacks: dict[str, Callable[[ASREventWithSource], None] | None] = {"local": None, "remote": None}
        self._retired_streams: list[StreamingAsr] = []
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
        previous_streams = list(self._streams.items())
        self._streams.clear()
        self._stream_fingerprints.clear()
        for source, stream in previous_streams:
            self._retire_stream(stream)
            callback = self._callbacks.get(source)
            if callback is not None:
                self._stream_of(source).start(callback)
        self._prune_retired_streams()

    def refresh_runtime(self) -> None:
        self.configure_pipeline(self._config, self._pipeline_revision)

    def start(self, source: str, on_event: Callable[[ASREventWithSource], None]) -> None:
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
                    is_early_final=event.is_early_final,
                    start_ms=event.start_ms,
                    end_ms=event.end_ms,
                    latency_ms=event.latency_ms,
                    detected_language=event.detected_language,
                    raw_text=event.text,
                    correction_applied=False,
                    speaker_label=event.speaker_label,
                )
            )

            if event.is_final:
                with self._lock:
                    self._active_utterance[key] = None
                    self._revision[key] = 0

        key = source if source in ("local", "remote") else "local"
        self._callbacks[key] = _wrapped
        self._prune_retired_streams()
        stream = self._stream_of(key)
        stream.start(_wrapped)

    def stop(self, source: str) -> None:
        key = source if source in ("local", "remote") else "local"
        self._callbacks[key] = None
        stream = self._streams.get(key)
        if stream:
            stream.stop()
        self._prune_retired_streams()

    def stop_all(self) -> None:
        for stream in self._streams.values():
            stream.stop()
        for stream in self._retired_streams:
            try:
                stream.stop()
            except Exception:
                pass
        self._retired_streams.clear()
        self._callbacks = {"local": None, "remote": None}
        with self._lock:
            self._active_utterance = {"local": None, "remote": None}
            self._revision = {"local": 0, "remote": 0}

    def submit(self, source: str, chunk: np.ndarray, sample_rate: float) -> None:
        if not self.is_enabled(source):
            return
        self._prune_retired_streams()
        payload = chunk
        if payload.ndim == 2 and payload.shape[1] > 1:
            payload = self._collapse_stereo_for_asr(payload)
        stream = self._stream_of(source)
        stream.submit_chunk(payload, sample_rate)

    @staticmethod
    def _collapse_stereo_for_asr(chunk: np.ndarray) -> np.ndarray:
        channels = chunk.astype(np.float32, copy=False)
        if channels.ndim != 2 or channels.shape[1] <= 1:
            return channels.astype(np.float32, copy=False)

        channel_energy = np.sqrt(np.mean(np.square(channels), axis=0, dtype=np.float32))
        strongest = int(np.argmax(channel_energy))
        strongest_energy = float(channel_energy[strongest]) if channel_energy.size else 0.0
        weakest_energy = float(np.min(channel_energy)) if channel_energy.size else 0.0

        # If one channel is effectively dominant, prefer it over averaging in noise.
        if strongest_energy >= max(0.01, weakest_energy * 2.5):
            return channels[:, strongest].astype(np.float32, copy=False)

        # When stereo channels are broadly similar, average them so both local and
        # remote sources produce identical mono input for ASR.
        left = channels[:, 0].astype(np.float32, copy=False)
        right = channels[:, 1].astype(np.float32, copy=False)
        left_std = float(np.std(left))
        right_std = float(np.std(right))
        if left_std > 1e-6 and right_std > 1e-6:
            corr = float(np.corrcoef(left, right)[0, 1])
            if corr >= 0.3:
                return ((left + right) * 0.5).astype(np.float32, copy=False)
            if corr <= -0.3:
                return channels[:, strongest].astype(np.float32, copy=False)

        return channels[:, strongest].astype(np.float32, copy=False)

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
                    "adaptive_mode": "baseline",
                    "adaptive_partial_interval_ms": 0,
                    "adaptive_min_silence_duration_ms": 0,
                    "adaptive_soft_final_audio_ms": 0,
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
                "adaptive_mode": stats.adaptive_mode,
                "adaptive_partial_interval_ms": stats.adaptive_partial_interval_ms,
                "adaptive_min_silence_duration_ms": stats.adaptive_min_silence_duration_ms,
                "adaptive_soft_final_audio_ms": stats.adaptive_soft_final_audio_ms,
            }
        return result

    def _stream_of(self, source: str) -> StreamingAsr:
        key = source if source in ("local", "remote") else "local"
        fingerprint = self._build_stream_fingerprint(key)
        existing = self._streams.get(key)
        if existing and self._stream_fingerprints.get(key) == fingerprint:
            return existing
        if existing:
            self._retire_stream(existing)
            self._streams.pop(key, None)
            self._stream_fingerprints.pop(key, None)

        language = self._asr_language_for_source(key)
        asr_cfg = self._asr_profile_for_source(key)
        runtime_cfg = self._config.runtime
        vad_cfg = asr_cfg.vad
        stream_cfg = asr_cfg.streaming
        model_name = self._resolve_model_for_language(asr_cfg.model, language)
        queue_maxsize = self._asr_queue_maxsize_for_source(key)
        pre_roll_ms = self._effective_pre_roll_ms(
            configured_pre_roll_ms=int(getattr(runtime_cfg, "asr_pre_roll_ms", 500)),
            min_speech_duration_ms=int(getattr(vad_cfg, "min_speech_duration_ms", 150)),
            speech_pad_ms=int(getattr(vad_cfg, "speech_pad_ms", 320)),
        )

        stream = StreamingAsr(
            engine=FasterWhisperEngine(
                model=model_name,
                device=asr_cfg.device,
                compute_type=asr_cfg.compute_type,
                beam_size=asr_cfg.beam_size,
                final_beam_size=asr_cfg.final_beam_size,
                condition_on_previous_text=asr_cfg.condition_on_previous_text,
                final_condition_on_previous_text=asr_cfg.final_condition_on_previous_text,
                initial_prompt=asr_cfg.initial_prompt,
                hotwords=asr_cfg.hotwords,
                speculative_draft_model=asr_cfg.speculative_draft_model,
                speculative_num_beams=asr_cfg.speculative_num_beams,
                temperature_fallback=asr_cfg.temperature_fallback,
                no_speech_threshold=asr_cfg.no_speech_threshold,
                language=language,
            ),
            vad=VadSegmenter(vad_cfg),
            partial_interval_ms=stream_cfg.partial_interval_ms,
            partial_history_seconds=stream_cfg.partial_history_seconds,
            final_history_seconds=stream_cfg.final_history_seconds,
            soft_final_audio_ms=stream_cfg.soft_final_audio_ms,
            pre_roll_ms=pre_roll_ms,
            min_partial_audio_ms=int(getattr(runtime_cfg, "asr_partial_min_audio_ms", 280)),
            partial_interval_floor_ms=int(getattr(runtime_cfg, "asr_partial_interval_floor_ms", 280)),
            early_final_enabled=bool(getattr(runtime_cfg, "early_final_enabled", True)),
            adaptive_enabled=bool(getattr(runtime_cfg, "adaptive_asr_enabled", True)),
            queue_maxsize=queue_maxsize,
            speaker_diarizer=self._speaker_diarizer_for_source(key),
            on_debug=self._on_error,
        )
        self._streams[key] = stream
        self._stream_fingerprints[key] = fingerprint
        return stream

    @staticmethod
    def _effective_pre_roll_ms(
        *,
        configured_pre_roll_ms: int,
        min_speech_duration_ms: int,
        speech_pad_ms: int,
    ) -> int:
        configured = max(0, int(configured_pre_roll_ms))
        # Keep enough lead-in audio to survive the time it takes VAD to confirm speech.
        recommended = max(
            int(min_speech_duration_ms) + 160,
            int(round(max(0, int(speech_pad_ms)) * 0.75)),
        )
        return max(configured, min(1200, recommended))

    def _retire_stream(self, stream: StreamingAsr) -> None:
        stream.request_stop()
        self._retired_streams.append(stream)

    def _prune_retired_streams(self) -> None:
        if not self._retired_streams:
            return
        still_running: list[StreamingAsr] = []
        for stream in self._retired_streams:
            if stream.cleanup_if_stopped():
                continue
            still_running.append(stream)
        self._retired_streams = still_running

    def _asr_profile_for_source(self, source: str):
        key = source if source in ("local", "remote") else "local"
        if key == "remote":
            return self._config.asr_channels.remote
        return self._config.asr_channels.local

    def _asr_queue_maxsize_for_source(self, source: str) -> int:
        key = source if source in ("local", "remote") else "local"
        runtime = self._config.runtime
        if key == "remote":
            value = int(getattr(runtime, "asr_queue_maxsize_remote", runtime.asr_queue_maxsize))
        else:
            value = int(getattr(runtime, "asr_queue_maxsize_local", runtime.asr_queue_maxsize))
        # Avoid too-small queues that can trigger aggressive overflow churn.
        return max(24, value)

    def _speaker_diarizer_for_source(self, source: str) -> OnlineSpeakerDiarizer | None:
        runtime = self._config.runtime
        if not bool(getattr(runtime, "speaker_diarization_enabled", False)):
            return None
        return OnlineSpeakerDiarizer(
            enabled=True,
            min_audio_ms=int(getattr(runtime, "speaker_diarization_min_audio_ms", 900)),
            max_speakers=int(getattr(runtime, "speaker_diarization_max_speakers", 3)),
            similarity_threshold=float(getattr(runtime, "speaker_diarization_similarity_threshold", 0.82)),
        )

    def _resolve_model_for_language(self, model: str, language: str) -> str:
        normalized_model = (model or "").strip()
        normalized_lang = (language or "").strip().lower()
        if "-" in normalized_lang:
            normalized_lang = normalized_lang.split("-", 1)[0]
        low_quality_models_for_zh = {
            "tiny",
            "tiny.en",
            "base",
            "base.en",
            "small",
            "small.en",
        }
        if normalized_lang in {"zh", "ja"} and normalized_model in low_quality_models_for_zh:
            if self._on_error:
                self._on_error(
                    f"asr model quality fallback: {normalized_model} is weak for {normalized_lang}, use large-v3 instead"
                )
            return "large-v3"
        if normalized_model == "distil-large-v3" and normalized_lang in {"zh", "ja"}:
            if self._on_error:
                self._on_error(
                    f"asr model fallback: distil-large-v3 is weak for {normalized_lang}, use large-v3 instead"
                )
            return "large-v3"
        return normalized_model or "large-v3"

    def _asr_language_for_source(self, source: str) -> str:
        key = source if source in ("local", "remote") else "local"
        runtime = self._config.runtime
        if key == "remote":
            asr_lang = str(getattr(runtime, "remote_asr_language", "auto") or "auto").strip().lower()
        else:
            asr_lang = str(getattr(runtime, "local_asr_language", "auto") or "auto").strip().lower()

        if asr_lang == "auto" or not asr_lang:
            return ""

        if asr_lang.startswith("zh"):
            return "zh"
        if asr_lang.startswith("en"):
            return "en"
        if asr_lang.startswith("ja"):
            return "ja"
        if asr_lang.startswith("ko"):
            return "ko"
        if asr_lang.startswith("th"):
            return "th"
        return asr_lang

    def _build_runtime_fingerprint(self) -> str:
        payload = {
            "asr": {
                "local": {
                    "model": self._config.asr_channels.local.model,
                    "device": self._config.asr_channels.local.device,
                    "compute_type": self._config.asr_channels.local.compute_type,
                    "beam_size": self._config.asr_channels.local.beam_size,
                    "final_beam_size": self._config.asr_channels.local.final_beam_size,
                    "condition_on_previous_text": self._config.asr_channels.local.condition_on_previous_text,
                    "final_condition_on_previous_text": self._config.asr_channels.local.final_condition_on_previous_text,
                    "initial_prompt": self._config.asr_channels.local.initial_prompt,
                    "hotwords": self._config.asr_channels.local.hotwords,
                    "speculative_draft_model": self._config.asr_channels.local.speculative_draft_model,
                    "speculative_num_beams": self._config.asr_channels.local.speculative_num_beams,
                    "temperature_fallback": self._config.asr_channels.local.temperature_fallback,
                    "no_speech_threshold": self._config.asr_channels.local.no_speech_threshold,
                    "funasr": {
                        "model": self._config.asr_channels.local.funasr.model,
                        "use_itn": self._config.asr_channels.local.funasr.use_itn,
                        "batch_size_s_offline": self._config.asr_channels.local.funasr.batch_size_s_offline,
                        "batch_size_s_online": self._config.asr_channels.local.funasr.batch_size_s_online,
                        "online_chunk_size": self._config.asr_channels.local.funasr.online_chunk_size,
                        "online_encoder_chunk_look_back": self._config.asr_channels.local.funasr.online_encoder_chunk_look_back,
                        "online_decoder_chunk_look_back": self._config.asr_channels.local.funasr.online_decoder_chunk_look_back,
                        "benchmark_window_ms": self._config.asr_channels.local.funasr.benchmark_window_ms,
                        "benchmark_overlap_ms": self._config.asr_channels.local.funasr.benchmark_overlap_ms,
                        "suppress_low_confidence_short": self._config.asr_channels.local.funasr.suppress_low_confidence_short,
                        "short_text_max_chars": self._config.asr_channels.local.funasr.short_text_max_chars,
                        "min_speech_ratio_for_short_text": self._config.asr_channels.local.funasr.min_speech_ratio_for_short_text,
                        "low_peak_threshold": self._config.asr_channels.local.funasr.low_peak_threshold,
                    },
                    "vad": {
                        "enabled": self._config.asr_channels.local.vad.enabled,
                        "backend": self._config.asr_channels.local.vad.backend,
                        "min_speech_duration_ms": self._config.asr_channels.local.vad.min_speech_duration_ms,
                        "min_silence_duration_ms": self._config.asr_channels.local.vad.min_silence_duration_ms,
                        "max_speech_duration_s": self._config.asr_channels.local.vad.max_speech_duration_s,
                        "speech_pad_ms": self._config.asr_channels.local.vad.speech_pad_ms,
                        "rms_threshold": self._config.asr_channels.local.vad.rms_threshold,
                        "neural_threshold": self._config.asr_channels.local.vad.neural_threshold,
                    },
                    "streaming": {
                        "partial_interval_ms": self._config.asr_channels.local.streaming.partial_interval_ms,
                        "partial_history_seconds": self._config.asr_channels.local.streaming.partial_history_seconds,
                        "final_history_seconds": self._config.asr_channels.local.streaming.final_history_seconds,
                        "soft_final_audio_ms": self._config.asr_channels.local.streaming.soft_final_audio_ms,
                    },
                },
                "remote": {
                    "model": self._config.asr_channels.remote.model,
                    "device": self._config.asr_channels.remote.device,
                    "compute_type": self._config.asr_channels.remote.compute_type,
                    "beam_size": self._config.asr_channels.remote.beam_size,
                    "final_beam_size": self._config.asr_channels.remote.final_beam_size,
                    "condition_on_previous_text": self._config.asr_channels.remote.condition_on_previous_text,
                    "final_condition_on_previous_text": self._config.asr_channels.remote.final_condition_on_previous_text,
                    "initial_prompt": self._config.asr_channels.remote.initial_prompt,
                    "hotwords": self._config.asr_channels.remote.hotwords,
                    "speculative_draft_model": self._config.asr_channels.remote.speculative_draft_model,
                    "speculative_num_beams": self._config.asr_channels.remote.speculative_num_beams,
                    "temperature_fallback": self._config.asr_channels.remote.temperature_fallback,
                    "no_speech_threshold": self._config.asr_channels.remote.no_speech_threshold,
                    "funasr": {
                        "model": self._config.asr_channels.remote.funasr.model,
                        "use_itn": self._config.asr_channels.remote.funasr.use_itn,
                        "batch_size_s_offline": self._config.asr_channels.remote.funasr.batch_size_s_offline,
                        "batch_size_s_online": self._config.asr_channels.remote.funasr.batch_size_s_online,
                        "online_chunk_size": self._config.asr_channels.remote.funasr.online_chunk_size,
                        "online_encoder_chunk_look_back": self._config.asr_channels.remote.funasr.online_encoder_chunk_look_back,
                        "online_decoder_chunk_look_back": self._config.asr_channels.remote.funasr.online_decoder_chunk_look_back,
                        "benchmark_window_ms": self._config.asr_channels.remote.funasr.benchmark_window_ms,
                        "benchmark_overlap_ms": self._config.asr_channels.remote.funasr.benchmark_overlap_ms,
                        "suppress_low_confidence_short": self._config.asr_channels.remote.funasr.suppress_low_confidence_short,
                        "short_text_max_chars": self._config.asr_channels.remote.funasr.short_text_max_chars,
                        "min_speech_ratio_for_short_text": self._config.asr_channels.remote.funasr.min_speech_ratio_for_short_text,
                        "low_peak_threshold": self._config.asr_channels.remote.funasr.low_peak_threshold,
                    },
                    "vad": {
                        "enabled": self._config.asr_channels.remote.vad.enabled,
                        "backend": self._config.asr_channels.remote.vad.backend,
                        "min_speech_duration_ms": self._config.asr_channels.remote.vad.min_speech_duration_ms,
                        "min_silence_duration_ms": self._config.asr_channels.remote.vad.min_silence_duration_ms,
                        "max_speech_duration_s": self._config.asr_channels.remote.vad.max_speech_duration_s,
                        "speech_pad_ms": self._config.asr_channels.remote.vad.speech_pad_ms,
                        "rms_threshold": self._config.asr_channels.remote.vad.rms_threshold,
                        "neural_threshold": self._config.asr_channels.remote.vad.neural_threshold,
                    },
                    "streaming": {
                        "partial_interval_ms": self._config.asr_channels.remote.streaming.partial_interval_ms,
                        "partial_history_seconds": self._config.asr_channels.remote.streaming.partial_history_seconds,
                        "final_history_seconds": self._config.asr_channels.remote.streaming.final_history_seconds,
                        "soft_final_audio_ms": self._config.asr_channels.remote.streaming.soft_final_audio_ms,
                    },
                },
            },
            "language": {
                "local_source": "auto",
                "meeting_source": "auto",
                "asr_language_mode": "auto",
                "local_asr_language": str(getattr(self._config.runtime, "local_asr_language", "auto") or "auto"),
                "remote_asr_language": str(getattr(self._config.runtime, "remote_asr_language", "auto") or "auto"),
            },
            "runtime": {
                "asr_queue_maxsize_local": self._config.runtime.asr_queue_maxsize_local,
                "asr_queue_maxsize_remote": self._config.runtime.asr_queue_maxsize_remote,
                "asr_pre_roll_ms": int(getattr(self._config.runtime, "asr_pre_roll_ms", 500)),
                "asr_partial_min_audio_ms": int(getattr(self._config.runtime, "asr_partial_min_audio_ms", 280)),
                "asr_partial_interval_floor_ms": int(getattr(self._config.runtime, "asr_partial_interval_floor_ms", 280)),
                "early_final_enabled": bool(getattr(self._config.runtime, "early_final_enabled", True)),
                "asr_final_correction_enabled": bool(getattr(self._config.runtime, "asr_final_correction_enabled", False)),
                "asr_final_correction_context_items": int(getattr(self._config.runtime, "asr_final_correction_context_items", 3)),
                "asr_final_correction_max_chars": int(getattr(self._config.runtime, "asr_final_correction_max_chars", 120)),
                "speaker_diarization_enabled": bool(getattr(self._config.runtime, "speaker_diarization_enabled", False)),
                "speaker_diarization_min_audio_ms": int(getattr(self._config.runtime, "speaker_diarization_min_audio_ms", 900)),
                "speaker_diarization_max_speakers": int(getattr(self._config.runtime, "speaker_diarization_max_speakers", 3)),
                "speaker_diarization_similarity_threshold": float(getattr(self._config.runtime, "speaker_diarization_similarity_threshold", 0.82)),
            },
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _build_stream_fingerprint(self, source: str) -> str:
        return f"{self._runtime_fingerprint}:{source}:rev={self._pipeline_revision}"
