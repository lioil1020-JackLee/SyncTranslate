from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.infra.asr.backend_resolution import BackendResolution, resolve_backend_for_language
from app.infra.asr.faster_whisper_adapter import FasterWhisperEngine
from app.infra.asr.lexical_bias_v2 import AsrLexicalBiaser
from app.infra.asr.transcript_validator_v2 import AsrTranscriptValidatorV2
from app.infra.config.schema import AppConfig, AsrConfig


@dataclass(slots=True)
class BackendDescriptor:
    name: str
    mode: str
    streaming: bool
    notes: str = ""


@dataclass(slots=True)
class BackendTranscript:
    text: str
    is_final: bool
    detected_language: str = ""
    start_ms: int = 0
    end_ms: int = 0
    confidence: float = 0.0


@dataclass(slots=True)
class BackendBuildResult:
    partial_backend: object
    final_backend: object
    resolution: BackendResolution


class BackendPostProcessor:
    def __init__(
        self,
        *,
        language: str,
        biaser: AsrLexicalBiaser,
        validator: AsrTranscriptValidatorV2,
    ) -> None:
        self._language = language
        self._biaser = biaser
        self._validator = validator

    def process(
        self,
        text: str,
        *,
        audio: np.ndarray,
        sample_rate: int,
        frontend_stats: dict[str, object] | None = None,
    ) -> BackendTranscript:
        biased = self._biaser.apply(text, language=self._language)
        result = self._validator.validate(
            biased,
            audio=audio,
            sample_rate=sample_rate,
            language=self._language,
            frontend_stats=frontend_stats,
        )
        return BackendTranscript(
            text=result.text if result.accepted else "",
            is_final=True,
            detected_language=self._language,
            confidence=result.score,
        )


class FasterWhisperStreamingBackend:
    def __init__(
        self,
        *,
        engine: FasterWhisperEngine,
        descriptor: BackendDescriptor,
        post_processor: BackendPostProcessor | None = None,
    ) -> None:
        self._engine = engine
        self.descriptor = descriptor
        self._post_processor = post_processor

    def warmup(self) -> None:
        self._engine.warmup()

    def transcribe_partial(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        frontend_stats: dict[str, object] | None = None,
    ) -> BackendTranscript:
        result = self._engine.transcribe_partial_result(audio, sample_rate)
        text = result.text
        if self._post_processor is not None and text:
            text = self._post_processor.process(
                text,
                audio=audio,
                sample_rate=sample_rate,
                frontend_stats=frontend_stats,
            ).text
        return BackendTranscript(
            text=text,
            is_final=False,
            detected_language=result.detected_language,
        )

    def transcribe_final(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        frontend_stats: dict[str, object] | None = None,
    ) -> BackendTranscript:
        result = self._engine.transcribe_final_result(audio, sample_rate)
        text = result.text
        confidence = 0.0
        if self._post_processor is not None and text:
            processed = self._post_processor.process(
                text,
                audio=audio,
                sample_rate=sample_rate,
                frontend_stats=frontend_stats,
            )
            text = processed.text
            confidence = processed.confidence
        if text:
            self._engine.push_context(text)
        return BackendTranscript(
            text=text,
            is_final=True,
            detected_language=result.detected_language,
            confidence=confidence,
        )

    def runtime_info(self) -> dict[str, object]:
        return {
            "device_effective": str(getattr(self._engine, "_runtime_device", "") or getattr(self._engine, "device", "")),
            "model_init_mode": "warm" if getattr(self._engine, "_model", None) is not None else "lazy",
            "init_failure": str(getattr(self._engine, "_fallback_reason", "") or ""),
            "runtime_label": self._engine.runtime_label(),
        }


def build_backend_pair(
    config: AppConfig,
    *,
    source: str,
    language: str,
) -> BackendBuildResult:
    resolution = resolve_backend_for_language(language)
    if resolution.disabled:
        raise ValueError("ASR runtime must not be created when language is set to none")

    profile = config.asr_channels.remote if source == "remote" else config.asr_channels.local
    post_processor = _build_post_processor(config, language=language)

    partial = FasterWhisperStreamingBackend(
        engine=_build_engine(profile, language=language),
        descriptor=BackendDescriptor(
            name="faster_whisper_v2:partial",
            mode="native_v2",
            streaming=True,
            notes="Low-latency partial decoding backend.",
        ),
        post_processor=post_processor,
    )
    final = FasterWhisperStreamingBackend(
        engine=_build_engine(profile, language=language),
        descriptor=BackendDescriptor(
            name="faster_whisper_v2:final",
            mode="native_v2",
            streaming=False,
            notes="High-quality final decoding backend.",
        ),
        post_processor=post_processor,
    )
    return BackendBuildResult(
        partial_backend=partial,
        final_backend=final,
        resolution=resolution,
    )


def _build_post_processor(config: AppConfig, *, language: str) -> BackendPostProcessor:
    runtime = config.runtime
    biaser = AsrLexicalBiaser(
        getattr(runtime, "asr_hotword_lexicon", ""),
        enabled=bool(getattr(runtime, "asr_hotwords_enabled", False)),
    )
    validator = AsrTranscriptValidatorV2(
        enabled=bool(getattr(runtime, "asr_result_validator_enabled", True)),
        max_chars_per_second=float(getattr(runtime, "asr_result_max_chars_per_second", 14.0)),
        min_speech_ratio_for_long_text=float(
            getattr(runtime, "asr_result_min_speech_ratio_for_long_text", 0.18)
        ),
    )
    return BackendPostProcessor(language=language, biaser=biaser, validator=validator)


def _build_engine(profile: AsrConfig, *, language: str) -> FasterWhisperEngine:
    return FasterWhisperEngine(
        model=profile.model,
        device=profile.device,
        compute_type=profile.compute_type,
        beam_size=profile.beam_size,
        final_beam_size=profile.final_beam_size,
        condition_on_previous_text=profile.condition_on_previous_text,
        final_condition_on_previous_text=profile.final_condition_on_previous_text,
        initial_prompt=profile.initial_prompt,
        hotwords=profile.hotwords,
        speculative_draft_model=profile.speculative_draft_model,
        speculative_num_beams=profile.speculative_num_beams,
        temperature_fallback=profile.temperature_fallback,
        no_speech_threshold=profile.no_speech_threshold,
        hallucination_filter=bool(getattr(profile, "hallucination_filter", True)),
        language=language,
    )


def _prepare_audio16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    mono = np.asarray(audio, dtype=np.float32)
    if mono.ndim == 2 and mono.shape[1] > 1:
        mono = np.mean(mono, axis=1, dtype=np.float32)
    else:
        mono = mono.reshape(-1).astype(np.float32, copy=False)
    if mono.size == 0:
        return mono
    mono = mono - float(np.mean(mono))
    rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
    if rms > 1e-6:
        target_rms = 0.05
        gain = min(3.0, target_rms / rms)
        mono = np.clip(mono * gain, -1.0, 1.0)
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if peak > 1.0:
        mono = mono / max(peak, 1.0)
    return _resample_linear(mono, sample_rate=sample_rate, target_rate=16000)


def _resample_linear(audio: np.ndarray, *, sample_rate: int, target_rate: int) -> np.ndarray:
    if sample_rate <= 0 or target_rate <= 0 or sample_rate == target_rate or audio.size <= 1:
        return audio.astype(np.float32, copy=False)
    src_len = int(audio.shape[0])
    dst_len = max(1, int(round(src_len * target_rate / sample_rate)))
    src_x = np.linspace(0.0, 1.0, src_len, endpoint=False)
    dst_x = np.linspace(0.0, 1.0, dst_len, endpoint=False)
    return np.interp(dst_x, src_x, audio).astype(np.float32, copy=False)


__all__ = [
    "BackendBuildResult",
    "BackendDescriptor",
    "BackendTranscript",
    "FasterWhisperStreamingBackend",
    "build_backend_pair",
    "_build_engine",
    "_prepare_audio16k",
]
