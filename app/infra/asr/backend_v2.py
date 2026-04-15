from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import Callable

import numpy as np

from app.infra.asr.backend_resolution import BackendResolution, resolve_backend_for_language
from app.infra.asr.faster_whisper_adapter import FasterWhisperEngine
from app.infra.asr.funasr_registry import get_funasr_registry
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


class FasterWhisperStreamingBackend:
    def __init__(self, *, engine: FasterWhisperEngine, descriptor: BackendDescriptor) -> None:
        self._engine = engine
        self.descriptor = descriptor

    def warmup(self) -> None:
        self._engine.warmup()

    def transcribe_partial(self, audio: np.ndarray, sample_rate: int) -> BackendTranscript:
        result = self._engine.transcribe_partial_result(audio, sample_rate)
        return BackendTranscript(
            text=result.text,
            is_final=False,
            detected_language=result.detected_language,
        )

    def transcribe_final(self, audio: np.ndarray, sample_rate: int) -> BackendTranscript:
        result = self._engine.transcribe_final_result(audio, sample_rate)
        return BackendTranscript(
            text=result.text,
            is_final=True,
            detected_language=result.detected_language,
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
    ) -> None:
        self._runner = runner
        self.descriptor = descriptor
        self._transcribe = transcribe

    def warmup(self) -> None:
        self._runner.warmup()

    def transcribe_partial(self, audio: np.ndarray, sample_rate: int) -> BackendTranscript:
        return self._transcribe(audio, sample_rate)

    def transcribe_final(self, audio: np.ndarray, sample_rate: int) -> BackendTranscript:
        return self._transcribe(audio, sample_rate)

    def runtime_info(self) -> dict[str, object]:
        return self._runner.runtime_info()


class _FunASRRunner:
    def __init__(self, *, model_name: str, device: str, language: str) -> None:
        self._model_name = model_name
        self._device = str(device or "cpu")
        self._language = _normalize_funasr_language(language)
        self._registry = get_funasr_registry()

    def warmup(self) -> None:
        self._registry.get_asr(model_name=self._model_name, requested_device=self._device)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> BackendTranscript:
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
        if _should_drop_funasr_text(text, audio=normalized):
            text = ""
        return BackendTranscript(
            text=text,
            is_final=True,
            detected_language=self._language,
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
    if resolution.backend_name == "funasr_v2":
        partial, final = _build_funasr_backend_pair(profile, language=language)
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
    )
    final = FasterWhisperStreamingBackend(
        engine=_build_engine(profile, language=language),
        descriptor=BackendDescriptor(
            name="faster_whisper_v2:final",
            mode="native_v2",
            streaming=False,
            notes="High-quality final decoding backend.",
        ),
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
) -> tuple[FunASRStreamingBackend, FunASRStreamingBackend]:
    runner = _FunASRRunner(
        model_name="iic/SenseVoiceSmall",
        device=profile.device,
        language=language,
    )
    partial = FunASRStreamingBackend(
        runner=runner,
        descriptor=BackendDescriptor(
            name="funasr_v2:partial",
            mode="native_v2",
            streaming=True,
            notes="FunASR SenseVoice partial decoding backend.",
        ),
        transcribe=lambda audio, sample_rate: runner.transcribe(audio, sample_rate),
    )
    final = FunASRStreamingBackend(
        runner=runner,
        descriptor=BackendDescriptor(
            name="funasr_v2:final",
            mode="native_v2",
            streaming=False,
            notes="FunASR SenseVoice final decoding backend.",
        ),
        transcribe=lambda audio, sample_rate: runner.transcribe(audio, sample_rate),
    )
    return partial, final


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


def _should_drop_funasr_text(text: str, *, audio: np.ndarray) -> bool:
    normalized = str(text or "").strip()
    if not normalized:
        return True
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < 0.003 and normalized in {"嗯", "嗯。", "啊", "啊。", "噢", "哦"}:
        return True
    return False


__all__ = [
    "BackendBuildResult",
    "BackendDescriptor",
    "BackendTranscript",
    "FasterWhisperStreamingBackend",
    "FunASRStreamingBackend",
    "build_backend_pair",
]
