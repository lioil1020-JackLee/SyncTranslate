from __future__ import annotations

import time
from threading import Lock
from typing import Callable
from uuid import uuid4

import numpy as np

from app.infra.asr.backend_resolution import resolve_backend_for_language
from app.domain.events import ErrorEvent
from app.infra.asr.backend_v2 import build_backend_pair
from app.infra.asr.contracts import ASREventWithSource
from app.infra.asr.endpoint_profiles import get_endpoint_profile
from app.infra.asr.endpointing_v2 import EndpointSignal, build_endpointing_runtime
from app.infra.asr.language_profiles import resolve_language_asr_profile
from app.infra.asr.pipeline_v2 import AsrV2PipelineSpec, build_v2_pipeline_spec, resolve_requested_asr_language
from app.infra.asr.worker_v2 import SourceRuntimeV2, V2RuntimeEvent
from app.infra.config.schema import AppConfig, RuntimeConfig


class ASRManagerV2:
    """Independent ASR v2 execution path.

    This manager owns its own endpointing runtime, chunk queue, worker thread,
    and partial/final emission path.
    """

    pipeline_mode = "v2"

    def __init__(
        self,
        config: AppConfig,
        on_error: Callable[[str | ErrorEvent], None] | None = None,
        *,
        pipeline_revision: int = 1,
    ) -> None:
        self._config = config
        self._on_error = on_error
        self._pipeline_revision = max(1, int(pipeline_revision))
        self._spec: AsrV2PipelineSpec = build_v2_pipeline_spec(config)
        self._user_callbacks: dict[str, Callable[[ASREventWithSource], None] | None] = {"local": None, "remote": None}
        self._callbacks: dict[str, Callable[[ASREventWithSource], None] | None] = {"local": None, "remote": None}
        self._enabled = {"local": True, "remote": True}
        self._runtimes: dict[str, SourceRuntimeV2] = {}
        self._source_resolution: dict[str, dict[str, object]] = {}
        self._lock = Lock()
        self._active_utterance: dict[str, str | None] = {"local": None, "remote": None}
        self._revision: dict[str, int] = {"local": 0, "remote": 0}
        self._runtime_fingerprint = self._build_runtime_fingerprint()

    def configure_pipeline(self, config: AppConfig, pipeline_revision: int) -> None:
        self._config = config
        self._pipeline_revision = max(1, int(pipeline_revision))
        self._spec = build_v2_pipeline_spec(config)
        self._runtime_fingerprint = self._build_runtime_fingerprint()
        previous_callbacks = dict(self._user_callbacks)
        self.stop_all()
        self._user_callbacks = previous_callbacks
        self._callbacks = previous_callbacks
        for source, callback in previous_callbacks.items():
            if callback is not None:
                self.start(source, callback)

    def refresh_runtime(self) -> None:
        self.configure_pipeline(self._config, self._pipeline_revision)

    def start(self, source: str, on_event: Callable[[ASREventWithSource], None]) -> None:
        key = source if source in {"local", "remote"} else "local"
        self._user_callbacks[key] = on_event

        def _wrapped(event: V2RuntimeEvent) -> None:
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
                    speaker_label="",
                )
            )

            if event.is_final:
                with self._lock:
                    self._active_utterance[key] = None
                    self._revision[key] = 0

        self._callbacks[key] = _wrapped
        if resolve_backend_for_language(self._asr_language_for_source(key)).disabled:
            self._source_resolution[key] = {
                "backend_name": "disabled",
                "language_family": "disabled",
                "reason": "ASR is disabled for this channel",
                "requested_language": self._asr_language_for_source(key),
                "normalized_language": "none",
            }
            return
        runtime = self._runtime_of(key)
        runtime.start(_wrapped)

    def stop(self, source: str) -> None:
        key = source if source in {"local", "remote"} else "local"
        self._callbacks[key] = None
        self._user_callbacks[key] = None
        runtime = self._runtimes.pop(key, None)
        if runtime is not None:
            runtime.stop()
        with self._lock:
            self._active_utterance[key] = None
            self._revision[key] = 0

    def stop_all(self) -> None:
        for runtime in list(self._runtimes.values()):
            runtime.stop()
        self._runtimes.clear()
        self._source_resolution.clear()
        self._callbacks = {"local": None, "remote": None}
        self._user_callbacks = {"local": None, "remote": None}
        with self._lock:
            self._active_utterance = {"local": None, "remote": None}
            self._revision = {"local": 0, "remote": 0}

    def submit(self, source: str, chunk: np.ndarray, sample_rate: float) -> None:
        key = source if source in {"local", "remote"} else "local"
        if not self.is_enabled(key):
            return
        runtime = self._stream_of(key)
        runtime.submit_chunk(self._collapse_stereo_for_asr(chunk), sample_rate)

    def set_enabled(self, source: str, enabled: bool) -> None:
        key = source if source in {"local", "remote"} else "local"
        with self._lock:
            self._enabled[key] = bool(enabled)

    def is_enabled(self, source: str) -> bool:
        key = source if source in {"local", "remote"} else "local"
        with self._lock:
            return self._enabled.get(key, False)

    def stats(self) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for source in ("local", "remote"):
            runtime = self._runtimes.get(source)
            if runtime is None:
                result[source] = self._empty_stats(source)
                continue
            stats = runtime.stats()
            signal = stats.last_signal
            resolution = self._source_resolution.get(source, {})
            backend_runtime = stats.backend_runtime
            result[source] = {
                "queue_size": stats.queue_size,
                "dropped_chunks": stats.dropped_chunks,
                "partial_count": stats.partial_count,
                "final_count": stats.final_count,
                "last_debug": stats.last_debug,
                "vad_rms": float(stats.endpointing.get("rms", 0.0)),
                "vad_threshold": float(self._threshold_for_source(source)),
                "adaptive_mode": "v2_native",
                "adaptive_partial_interval_ms": int(stats.adaptive_partial_interval_ms),
                "adaptive_min_silence_duration_ms": int(self._min_silence_for_source(source)),
                "adaptive_soft_final_audio_ms": int(stats.adaptive_soft_final_audio_ms),
                "pipeline_mode": self.pipeline_mode,
                "execution_mode": self._spec.execution_mode,
                "partial_backend": stats.partial_backend_name,
                "final_backend": stats.final_backend_name,
                "endpointing_backend": str(stats.endpointing.get("backend", "")),
                "resolved_backend": str(resolution.get("backend_name", "")),
                "resolved_language_family": str(resolution.get("language_family", "")),
                "backend_resolution_reason": str(resolution.get("reason", "")),
                "requested_asr_language": str(resolution.get("requested_language", "")),
                "device_effective": str(backend_runtime.get("device_effective", "")),
                "model_init_mode": str(backend_runtime.get("model_init_mode", "lazy")),
                "init_failure": str(backend_runtime.get("init_failure", "")),
                "postprocessor": backend_runtime.get("postprocessor", {}),
                "frontend": stats.frontend,
                "endpointing": stats.endpointing,
                "endpoint_signal": {
                    "speech_active": signal.speech_active,
                    "speech_started": signal.speech_started,
                    "speech_ended": signal.speech_ended,
                    "soft_endpoint": signal.soft_endpoint,
                    "hard_endpoint": signal.hard_endpoint,
                    "pause_ms": int(round(signal.pause_ms)),
                    "speech_probability": round(signal.speech_probability, 4),
                },
            }
        return result

    @property
    def pipeline_spec(self) -> AsrV2PipelineSpec:
        return self._spec

    def endpoint_snapshot(self, source: str) -> dict[str, object]:
        key = source if source in {"local", "remote"} else "local"
        runtime = self._runtimes.get(key)
        if runtime is None:
            resolution = resolve_backend_for_language(self._asr_language_for_source(key))
            profile = self._profile_for_source(key)
            endpoint_runtime = build_endpointing_runtime(
                str(getattr(self._config.runtime, "asr_v2_endpointing", "neural_endpoint")),
                profile.vad,
                device=profile.device,
                resolved_backend_name=resolution.backend_name,
            )
            return endpoint_runtime.snapshot()
        return runtime.stats().endpointing

    def _runtime_of(self, source: str) -> SourceRuntimeV2:
        runtime = self._runtimes.get(source)
        if runtime is not None:
            return runtime
        language = self._asr_language_for_source(source)
        resolution = resolve_backend_for_language(language)
        profile = self._profile_for_source(source)
        language_profile = resolve_language_asr_profile(profile, language=language)
        build_result = build_backend_pair(
            self._config,
            source=source,
            language=language,
            profile_override=language_profile.asr,
        )
        if isinstance(build_result, tuple):
            partial_backend, final_backend = build_result
            resolution = resolve_backend_for_language(language)
        else:
            partial_backend = build_result.partial_backend
            final_backend = build_result.final_backend
            resolution = build_result.resolution
        endpointing = build_endpointing_runtime(
            str(getattr(self._config.runtime, "asr_v2_endpointing", "neural_endpoint")),
            profile.vad,
            device=profile.device,
            resolved_backend_name=resolution.backend_name,
        )
        self._source_resolution[source] = {
            "backend_name": resolution.backend_name,
            "language_family": resolution.language_family,
            "reason": resolution.reason,
            "requested_language": resolution.requested_language,
            "normalized_language": resolution.normalized_language,
        }
        # Resolve endpoint profile for this source channel
        _profile_name = self._endpoint_profile_name_for_source(source, language_profile=language_profile)
        _ep_profile = get_endpoint_profile(_profile_name)
        _ep_kwargs = _ep_profile.to_worker_kwargs()
        noise_reduce_strength, music_suppress_strength = self._frontend_enhancement_strengths(
            language,
            source=source,
        )

        runtime = SourceRuntimeV2(
            source=source,
            partial_backend=partial_backend,
            final_backend=final_backend,
            endpointing=endpointing,
            partial_interval_ms=_ep_kwargs.get("partial_interval_ms", language_profile.asr.streaming.partial_interval_ms),
            partial_history_seconds=language_profile.asr.streaming.partial_history_seconds,
            final_history_seconds=language_profile.asr.streaming.final_history_seconds,
            soft_final_audio_ms=_ep_kwargs.get("soft_final_audio_ms", language_profile.asr.streaming.soft_final_audio_ms),
            pre_roll_ms=_ep_kwargs.get("pre_roll_ms", int(getattr(self._config.runtime, "asr_pre_roll_ms", 500))),
            min_partial_audio_ms=_ep_kwargs.get("min_partial_audio_ms", int(getattr(self._config.runtime, "asr_partial_min_audio_ms", 280))),
            queue_maxsize=self._queue_maxsize_for_source(source),
            early_final_enabled=bool(getattr(self._config.runtime, "early_final_enabled", True)),
            partial_interval_floor_ms=int(getattr(self._config.runtime, "asr_partial_interval_floor_ms", 280)),
            adaptive_enabled=bool(getattr(self._config.runtime, "adaptive_asr_enabled", True)),
            degradation_enabled=bool(getattr(self._config.runtime, "degradation_policy_enabled", True)),
            soft_endpoint_finalize_audio_ms=_ep_kwargs.get("soft_endpoint_finalize_audio_ms"),
            speech_end_finalize_audio_ms=_ep_kwargs.get("speech_end_finalize_audio_ms"),
            frontend_enabled=bool(getattr(self._config.runtime, "asr_frontend_enabled", True)),
            frontend_target_rms=float(getattr(self._config.runtime, "asr_frontend_target_rms", 0.05)),
            frontend_max_gain=float(getattr(self._config.runtime, "asr_frontend_max_gain", 3.0)),
            frontend_highpass_alpha=float(getattr(self._config.runtime, "asr_frontend_highpass_alpha", 0.96)),
            enhancement_enabled=bool(getattr(self._config.runtime, "asr_enhancement_enabled", True)),
            enhancement_noise_reduce_strength=noise_reduce_strength,
            enhancement_noise_adapt_rate=float(
                getattr(self._config.runtime, "asr_enhancement_noise_adapt_rate", 0.18)
            ),
            enhancement_music_suppress_strength=music_suppress_strength,
            on_debug=self._on_error,
        )
        self._runtimes[source] = runtime
        return runtime

    def _stream_of(self, source: str) -> SourceRuntimeV2:
        return self._runtime_of(source)

    def _profile_for_source(self, source: str):
        return self._config.asr_channels.remote if source == "remote" else self._config.asr_channels.local

    def _asr_profile_for_source(self, source: str):
        return self._profile_for_source(source)

    def _profile_for_language(self, language: str):
        resolution = resolve_backend_for_language(language)
        if resolution.language_family == "non_chinese":
            return self._config.asr_channels.remote
        return self._config.asr_channels.local

    def _endpoint_profile_name_for_source(self, source: str, *, language_profile) -> str:
        configured = (
            getattr(self._config.runtime, "asr_profile_remote", None)
            if source == "remote"
            else getattr(self._config.runtime, "asr_profile_local", None)
        )
        configured_name = str(configured or "").strip()
        if configured_name and configured_name not in {"default", "meeting_room"}:
            return configured_name
        return language_profile.endpoint_profile or configured_name or "default"

    def _frontend_enhancement_strengths(self, language: str, *, source: str = "local") -> tuple[float, float]:
        profile = resolve_language_asr_profile(self._profile_for_source(source), language=language)
        runtime = self._config.runtime
        defaults = RuntimeConfig()
        configured_noise = float(
            getattr(runtime, "asr_enhancement_noise_reduce_strength", defaults.asr_enhancement_noise_reduce_strength)
        )
        configured_music = float(
            getattr(
                runtime,
                "asr_enhancement_music_suppress_strength",
                defaults.asr_enhancement_music_suppress_strength,
            )
        )
        noise = (
            float(profile.frontend.noise_reduce_strength)
            if abs(configured_noise - defaults.asr_enhancement_noise_reduce_strength) < 1e-9
            else configured_noise
        )
        music = (
            float(profile.frontend.music_suppress_strength)
            if abs(configured_music - defaults.asr_enhancement_music_suppress_strength) < 1e-9
            else configured_music
        )
        return (
            noise,
            music,
        )

    def _queue_maxsize_for_source(self, source: str) -> int:
        runtime = self._config.runtime
        if source == "remote":
            return int(getattr(runtime, "asr_queue_maxsize_remote", getattr(runtime, "asr_queue_maxsize", 128)))
        return int(getattr(runtime, "asr_queue_maxsize_local", getattr(runtime, "asr_queue_maxsize", 128)))

    def _asr_queue_maxsize_for_source(self, source: str) -> int:
        return self._queue_maxsize_for_source(source)

    def _asr_language_for_source(self, source: str) -> str:
        return resolve_requested_asr_language(self._config, source)

    def _speaker_diarizer_for_source(self, source: str):
        return None

    def _effective_language_profile_for_source(self, source: str):
        return resolve_language_asr_profile(
            self._profile_for_source(source),
            language=self._asr_language_for_source(source),
        )

    def _threshold_for_source(self, source: str) -> float:
        vad = self._effective_language_profile_for_source(source).asr.vad
        backend = str(getattr(vad, "backend", "rms") or "rms").strip().lower()
        if backend in {"silero", "silero_vad", "neural", "neural_endpoint"}:
            return float(getattr(vad, "neural_threshold", 0.5))
        return float(getattr(vad, "rms_threshold", 0.02))

    def _partial_interval_for_source(self, source: str) -> int:
        return int(self._effective_language_profile_for_source(source).asr.streaming.partial_interval_ms)

    def _min_silence_for_source(self, source: str) -> int:
        return int(self._effective_language_profile_for_source(source).asr.vad.min_silence_duration_ms)

    def _soft_final_for_source(self, source: str) -> int:
        return int(self._effective_language_profile_for_source(source).asr.streaming.soft_final_audio_ms)

    @staticmethod
    def _collapse_stereo_for_asr(chunk: np.ndarray) -> np.ndarray:
        channels = np.asarray(chunk, dtype=np.float32)
        if channels.ndim != 2 or channels.shape[1] <= 1:
            return channels.reshape(-1).astype(np.float32, copy=False)

        channel_energy = np.sqrt(np.mean(np.square(channels), axis=0, dtype=np.float32))
        strongest = int(np.argmax(channel_energy)) if channel_energy.size else 0
        strongest_energy = float(channel_energy[strongest]) if channel_energy.size else 0.0
        weakest_energy = float(np.min(channel_energy)) if channel_energy.size else 0.0
        if strongest_energy >= max(0.01, weakest_energy * 2.5):
            return channels[:, strongest].astype(np.float32, copy=False)

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

    @staticmethod
    def _effective_pre_roll_ms(
        *,
        configured_pre_roll_ms: int,
        min_speech_duration_ms: int,
        speech_pad_ms: int,
    ) -> int:
        configured = max(0, int(configured_pre_roll_ms))
        recommended = max(
            int(min_speech_duration_ms) + 160,
            int(round(max(0, int(speech_pad_ms)) * 0.75)),
        )
        return max(configured, min(1200, recommended))

    def _build_runtime_fingerprint(self) -> str:
        return f"v2:{self._pipeline_revision}:{self._spec.partial_backend.name}:{self._spec.endpointing.name}"

    def _empty_stats(self, source: str) -> dict[str, object]:
        language = self._asr_language_for_source(source)
        resolution = resolve_backend_for_language(language)
        return {
            "queue_size": 0,
            "dropped_chunks": 0,
            "partial_count": 0,
            "final_count": 0,
            "last_debug": "",
            "vad_rms": 0.0,
            "vad_threshold": 0.0,
            "adaptive_mode": "v2_native",
            "adaptive_partial_interval_ms": 0,
            "adaptive_min_silence_duration_ms": 0,
            "adaptive_soft_final_audio_ms": 0,
            "pipeline_mode": "v2",
            "execution_mode": "native_v2",
            "partial_backend": "",
            "final_backend": "",
            "endpointing_backend": "",
            "resolved_backend": resolution.backend_name,
            "resolved_language_family": resolution.language_family,
            "backend_resolution_reason": resolution.reason,
            "requested_asr_language": resolution.requested_language,
            "device_effective": "",
            "model_init_mode": "lazy",
            "init_failure": "",
            "postprocessor": {},
            "frontend": {},
            "endpointing": {},
            "endpoint_signal": {
                "speech_active": False,
                "speech_started": False,
                "speech_ended": False,
                "soft_endpoint": False,
                "hard_endpoint": False,
                "pause_ms": 0,
                "speech_probability": 0.0,
            },
        }


__all__ = ["ASRManagerV2"]
