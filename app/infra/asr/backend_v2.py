from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
import re
from typing import Callable

import numpy as np

from app.infra.asr.backend_resolution import BackendResolution, resolve_backend_for_language
from app.infra.asr.faster_whisper_adapter import FasterWhisperEngine
from app.infra.asr.funasr_registry import get_funasr_registry
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


class FunASRStreamingBackend:
    def __init__(
        self,
        *,
        runner: "_FunASRRunner",
        descriptor: BackendDescriptor,
        transcribe: Callable[[np.ndarray, int], BackendTranscript],
        frontend_stats: Callable[[], dict[str, object]],
    ) -> None:
        self._runner = runner
        self.descriptor = descriptor
        self._transcribe = transcribe
        self._frontend_stats = frontend_stats

    def warmup(self) -> None:
        self._runner.warmup()

    def transcribe_partial(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        frontend_stats: dict[str, object] | None = None,
    ) -> BackendTranscript:
        return self._transcribe(audio, sample_rate, frontend_stats=frontend_stats)

    def transcribe_final(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        frontend_stats: dict[str, object] | None = None,
    ) -> BackendTranscript:
        return self._transcribe(audio, sample_rate, frontend_stats=frontend_stats)

    def runtime_info(self) -> dict[str, object]:
        return self._runner.runtime_info()

    def frontend_stats(self) -> dict[str, object]:
        return self._frontend_stats()


class _FunASRRunner:
    def __init__(self, *, model_name: str, device: str, language: str, post_processor: BackendPostProcessor | None = None) -> None:
        self._model_name = model_name
        self._device = str(device or "cpu")
        self._language = _normalize_funasr_language(language)
        self._registry = get_funasr_registry()
        self._post_processor = post_processor

    def warmup(self) -> None:
        self._registry.get_asr(model_name=self._model_name, requested_device=self._device)

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        frontend_stats: dict[str, object] | None = None,
    ) -> BackendTranscript:
        normalized = _prepare_audio16k(audio, sample_rate)
        if normalized.size == 0:
            return BackendTranscript(text="", is_final=True, detected_language=self._language)
        handle = self._registry.get_asr(model_name=self._model_name, requested_device=self._device)
        kwargs = {
            "input": normalized,
            "cache": {},
            "language": self._language,
            "use_itn": True,
            "batch_size_s": 0,
            "progress_callback": None,
        }
        invoke_lock = handle.invoke_lock
        if invoke_lock is None:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result = handle.model.generate(**kwargs)
        else:
            with invoke_lock:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    result = handle.model.generate(**kwargs)
        text = _extract_funasr_text(result)
        if handle.postprocess is not None and text:
            try:
                text = str(handle.postprocess(text) or "").strip()
            except Exception:
                text = str(text).strip()
        text = _sanitize_funasr_text(text)
        if _should_drop_funasr_text(text, audio=normalized):
            text = ""
        confidence = 0.0
        if self._post_processor is not None and text:
            processed = self._post_processor.process(
                text,
                audio=normalized,
                sample_rate=16000,
                frontend_stats=frontend_stats,
            )
            text = processed.text
            confidence = processed.confidence
        return BackendTranscript(
            text=text,
            is_final=True,
            detected_language=self._language,
            confidence=confidence,
        )

    def runtime_info(self) -> dict[str, object]:
        info = self._registry.snapshot_asr(model_name=self._model_name, requested_device=self._device)
        info["language_effective"] = self._language
        return info


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
    if resolution.backend_name == "funasr_v2":
        partial, final = _build_funasr_backend_pair(profile, language=language, post_processor=post_processor)
        return BackendBuildResult(
            partial_backend=partial,
            final_backend=final,
            resolution=resolution,
        )
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


def _build_funasr_backend_pair(
    profile: AsrConfig,
    *,
    language: str,
    post_processor: BackendPostProcessor | None = None,
) -> tuple[FunASRStreamingBackend, FunASRStreamingBackend]:
    runner = _FunASRRunner(
        model_name="iic/SenseVoiceSmall",
        device=profile.device,
        language=language,
        post_processor=post_processor,
    )
    partial = FunASRStreamingBackend(
        runner=runner,
        descriptor=BackendDescriptor(
            name="funasr_v2:partial",
            mode="native_v2",
            streaming=True,
            notes="FunASR SenseVoice partial decoding backend.",
        ),
        transcribe=lambda audio, sample_rate, frontend_stats=None: runner.transcribe(
            audio,
            sample_rate,
            frontend_stats=frontend_stats,
        ),
        frontend_stats=lambda: {},
    )
    final = FunASRStreamingBackend(
        runner=runner,
        descriptor=BackendDescriptor(
            name="funasr_v2:final",
            mode="native_v2",
            streaming=False,
            notes="FunASR SenseVoice final decoding backend.",
        ),
        transcribe=lambda audio, sample_rate, frontend_stats=None: runner.transcribe(
            audio,
            sample_rate,
            frontend_stats=frontend_stats,
        ),
        frontend_stats=lambda: {},
    )
    return partial, final


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
        temperature_fallback=profile.temperature_fallback,
        no_speech_threshold=profile.no_speech_threshold,
        language=language,
    )


def _normalize_funasr_language(language: str) -> str:
    normalized = str(language or "").strip().lower()
    if not normalized or normalized == "auto":
        return "auto"
    if normalized in {"zh", "zh-tw", "zh_tw", "zh-cn", "zh_cn", "cmn"}:
        return "zh"
    if normalized in {"yue", "cantonese"}:
        return "yue"
    if normalized in {"en", "english"}:
        return "en"
    if normalized in {"ja", "jp", "japanese"}:
        return "ja"
    if normalized in {"ko", "kr", "korean"}:
        return "ko"
    return "auto"


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


def _extract_funasr_text(result: object) -> str:
    if result is None:
        return ""
    if isinstance(result, dict):
        return str(result.get("text", "") or "").strip()
    if isinstance(result, list):
        parts: list[str] = []
        for item in result:
            if isinstance(item, dict):
                text = str(item.get("text", "") or "").strip()
                if text:
                    parts.append(text)
        return " ".join(parts).strip()
    return ""


def _sanitize_funasr_text(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    value = re.sub(r"<\|[^|>]{1,64}\|>", " ", value)
    value = re.sub(r"[\U0001F300-\U0001FAFF\u2600-\u27BF\u200d\ufe0f]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _should_drop_funasr_text(text: str, *, audio: np.ndarray) -> bool:
    normalized = str(text or "").strip()
    if not normalized:
        return True
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    speech_ratio = _estimate_speech_ratio(audio)
    if peak < 0.003 and normalized in {"嗯", "嗯。", "啊", "啊。", "噢", "哦"}:
        return True
    if speech_ratio < 0.12 and len(normalized) <= 10:
        return True
    return False


def _estimate_speech_ratio(audio: np.ndarray, *, sample_rate: int = 16000) -> float:
    signal = np.asarray(audio, dtype=np.float32).reshape(-1)
    if signal.size == 0 or sample_rate <= 0:
        return 0.0
    frame = max(160, int(sample_rate * 0.02))
    baseline = float(np.sqrt(np.mean(np.square(signal), dtype=np.float32))) if signal.size else 0.0
    threshold = max(0.003, baseline * 1.35)
    voiced = 0
    total = 0
    for start in range(0, max(1, signal.size - frame + 1), frame):
        end = min(signal.size, start + frame)
        frame_rms = float(np.sqrt(np.mean(np.square(signal[start:end]), dtype=np.float32)))
        if frame_rms >= threshold:
            voiced += 1
        total += 1
    return float(voiced) / float(max(1, total))


__all__ = [
    "BackendBuildResult",
    "BackendDescriptor",
    "BackendTranscript",
    "FasterWhisperStreamingBackend",
    "FunASRStreamingBackend",
    "build_backend_pair",
]
