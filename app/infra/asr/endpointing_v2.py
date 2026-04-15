from __future__ import annotations

import contextlib
import io
from dataclasses import asdict
from dataclasses import dataclass
from uuid import uuid4

import numpy as np

from app.infra.asr.funasr_registry import get_funasr_registry
from app.infra.config.schema import VadSettings


@dataclass(slots=True)
class EndpointingDescriptor:
    name: str
    mode: str
    notes: str = ""


@dataclass(slots=True)
class EndpointSignal:
    speech_started: bool = False
    speech_active: bool = False
    speech_ended: bool = False
    soft_endpoint: bool = False
    hard_endpoint: bool = False
    pause_ms: float = 0.0
    speech_ms: float = 0.0
    silence_ms: float = 0.0
    speech_probability: float = 0.0
    rms: float = 0.0


class EndpointingRuntime:
    def __init__(self, descriptor: EndpointingDescriptor, config: VadSettings, *, device: str = "cpu") -> None:
        self.descriptor = descriptor
        self._config = config
        self._device = str(device or "cpu")
        self._backend = str(getattr(config, "backend", "rms") or "rms").strip().lower()
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._pause_ms = 0.0
        self._speech_hold_ms = 0.0
        self._speech_active = False
        self._speech_started_count = 0
        self._soft_endpoint_count = 0
        self._hard_endpoint_count = 0
        self._last_probability = 0.0
        self._last_rms = 0.0
        self._funasr_vad = (
            _FunASRStreamingVad(device=self._device)
            if self._backend in {"funasr", "funasr_vad", "fsmn_vad", "fsmn-vad"}
            else None
        )
        self._neural_vad = (
            _SileroStreamingVad(threshold=float(getattr(config, "neural_threshold", 0.5)))
            if self._backend in {"silero", "silero_vad", "neural", "neural_endpoint"}
            else None
        )

    def reset(self) -> None:
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._pause_ms = 0.0
        self._speech_hold_ms = 0.0
        self._speech_active = False
        self._last_probability = 0.0
        self._last_rms = 0.0
        if self._funasr_vad is not None:
            self._funasr_vad.reset()
        if self._neural_vad is not None:
            self._neural_vad.reset()

    def update(self, chunk: np.ndarray, sample_rate: float) -> EndpointSignal:
        if sample_rate <= 0:
            return EndpointSignal(speech_active=self._speech_active, pause_ms=self._pause_ms)
        mono = np.asarray(chunk, dtype=np.float32).reshape(-1)
        chunk_ms = (mono.size / float(sample_rate)) * 1000.0 if mono.size else 0.0
        if mono.size == 0 or chunk_ms <= 0.0:
            return EndpointSignal(speech_active=self._speech_active, pause_ms=self._pause_ms)

        rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
        if self._funasr_vad is not None and self._funasr_vad.available:
            return self._update_with_funasr_vad(mono=mono, sample_rate=sample_rate, chunk_ms=chunk_ms, rms=rms)
        probability = self._speech_probability(mono=mono, sample_rate=sample_rate, rms=rms)
        return self._update_with_probability(probability=probability, chunk_ms=chunk_ms, rms=rms)

    def _update_with_probability(self, *, probability: float, chunk_ms: float, rms: float) -> EndpointSignal:
        has_speech = probability >= self._speech_threshold()
        min_speech_ms = max(40.0, float(getattr(self._config, "min_speech_duration_ms", 150)))
        min_silence_ms = max(120.0, float(getattr(self._config, "min_silence_duration_ms", 520)))
        speech_pad_ms = max(0.0, float(getattr(self._config, "speech_pad_ms", 320)))
        soft_pause_ms = max(200.0, min(min_silence_ms * 0.6, 520.0))

        speech_started = False
        speech_ended = False
        soft_endpoint = False
        hard_endpoint = False

        if has_speech:
            self._speech_ms += chunk_ms
            self._silence_ms = 0.0
            self._pause_ms = 0.0
            self._speech_hold_ms = 0.0
            if (not self._speech_active) and self._speech_ms >= min_speech_ms:
                self._speech_active = True
                speech_started = True
                self._speech_started_count += 1
        else:
            self._silence_ms += chunk_ms
            if self._speech_active:
                self._pause_ms += chunk_ms
                if self._speech_hold_ms < speech_pad_ms:
                    self._speech_hold_ms += chunk_ms
                if self._pause_ms >= soft_pause_ms:
                    soft_endpoint = True
                if self._silence_ms >= min_silence_ms:
                    hard_endpoint = True
                    speech_ended = True
                    self._speech_active = False
                    self._speech_ms = 0.0
                    self._silence_ms = 0.0
                    self._pause_ms = 0.0
                    self._speech_hold_ms = 0.0
            else:
                self._speech_ms = 0.0
                self._speech_hold_ms = 0.0
                self._pause_ms = 0.0

        if soft_endpoint:
            self._soft_endpoint_count += 1
        if hard_endpoint:
            self._hard_endpoint_count += 1
        self._last_probability = probability
        self._last_rms = rms
        return EndpointSignal(
            speech_started=speech_started,
            speech_active=self._speech_active,
            speech_ended=speech_ended,
            soft_endpoint=soft_endpoint,
            hard_endpoint=hard_endpoint,
            pause_ms=self._pause_ms,
            speech_ms=self._speech_ms,
            silence_ms=self._silence_ms,
            speech_probability=probability,
            rms=rms,
        )

    def _update_with_funasr_vad(
        self,
        *,
        mono: np.ndarray,
        sample_rate: float,
        chunk_ms: float,
        rms: float,
    ) -> EndpointSignal:
        events = self._funasr_vad.detect(chunk=mono, sample_rate=sample_rate)
        start_detected = any(begin >= 0 for begin, _end in events)
        end_detected = any(end >= 0 for _begin, end in events)
        fallback_threshold = max(0.0, float(getattr(self._config, "rms_threshold", 0.02)))
        fallback_probability = 1.0 if rms >= fallback_threshold else 0.0

        speech_started = False
        speech_ended = False
        soft_endpoint = False
        hard_endpoint = False
        min_speech_ms = max(40.0, float(getattr(self._config, "min_speech_duration_ms", 150)))
        min_silence_ms = max(120.0, float(getattr(self._config, "min_silence_duration_ms", 520)))
        soft_pause_ms = max(200.0, min(min_silence_ms * 0.6, 520.0))

        if start_detected:
            self._speech_ms = max(self._speech_ms, min_speech_ms)
        elif not self._speech_active and fallback_probability > 0.0:
            self._speech_ms += chunk_ms

        if (not self._speech_active) and self._speech_ms >= min_speech_ms:
            self._speech_active = True
            speech_started = True
            self._speech_started_count += 1
            self._silence_ms = 0.0
            self._pause_ms = 0.0

        if self._speech_active:
            self._speech_ms += chunk_ms
            if end_detected:
                self._pause_ms = max(self._pause_ms, chunk_ms)
                self._silence_ms = max(self._silence_ms, chunk_ms)
                soft_endpoint = True
                hard_endpoint = True
                speech_ended = True
                self._speech_active = False
                self._speech_ms = 0.0
                self._silence_ms = 0.0
                self._pause_ms = 0.0
            elif fallback_probability > 0.0:
                self._silence_ms = 0.0
                self._pause_ms = 0.0
            else:
                self._silence_ms += chunk_ms
                self._pause_ms += chunk_ms
                if self._pause_ms >= soft_pause_ms:
                    soft_endpoint = True
                if self._silence_ms >= min_silence_ms:
                    hard_endpoint = True
                    speech_ended = True
                    self._speech_active = False
                    self._speech_ms = 0.0
                    self._silence_ms = 0.0
                    self._pause_ms = 0.0
        elif fallback_probability <= 0.0:
            self._speech_ms = 0.0
            self._silence_ms += chunk_ms
            self._pause_ms = 0.0

        if soft_endpoint:
            self._soft_endpoint_count += 1
        if hard_endpoint:
            self._hard_endpoint_count += 1
        self._last_probability = 1.0 if (self._speech_active or start_detected or fallback_probability > 0.0) else 0.0
        self._last_rms = rms
        return EndpointSignal(
            speech_started=speech_started,
            speech_active=self._speech_active,
            speech_ended=speech_ended,
            soft_endpoint=soft_endpoint,
            hard_endpoint=hard_endpoint,
            pause_ms=self._pause_ms,
            speech_ms=self._speech_ms,
            silence_ms=self._silence_ms,
            speech_probability=self._last_probability,
            rms=rms,
        )

    def snapshot(self) -> dict[str, object]:
        if self._funasr_vad is not None:
            runtime_info = self._funasr_vad.runtime_info()
            if not isinstance(runtime_info, dict):
                runtime_info = {}
            available = bool(getattr(self._funasr_vad, "available", False))
            if runtime_info:
                available = bool(runtime_info.get("init_failure", "") == "")
        elif self._neural_vad is not None:
            runtime_info = {"model_init_mode": "warm" if self._neural_vad.available else "lazy"}
            available = bool(self._neural_vad.available)
        else:
            runtime_info = {"model_init_mode": "warm"}
            available = True
        return {
            "backend": self.descriptor.name,
            "speech_active": self._speech_active,
            "speech_probability": round(self._last_probability, 4),
            "rms": round(self._last_rms, 5),
            "speech_started_count": self._speech_started_count,
            "soft_endpoint_count": self._soft_endpoint_count,
            "hard_endpoint_count": self._hard_endpoint_count,
            "pause_ms": int(round(self._pause_ms)),
            "available": available,
            "device_effective": str(runtime_info.get("device_effective", "")),
            "model_init_mode": str(runtime_info.get("model_init_mode", "lazy")),
            "init_failure": str(runtime_info.get("init_failure", "")),
            "model_path": str(runtime_info.get("model_path", "")),
        }

    def _speech_threshold(self) -> float:
        if self._neural_vad is not None and self._neural_vad.available:
            return max(0.0, min(1.0, float(getattr(self._config, "neural_threshold", 0.5))))
        return max(0.0, float(getattr(self._config, "rms_threshold", 0.02)))

    def _speech_probability(self, *, mono: np.ndarray, sample_rate: float, rms: float) -> float:
        if self._neural_vad is not None and self._neural_vad.available:
            probability = self._neural_vad.probability(chunk=mono, sample_rate=sample_rate)
            if probability >= self._speech_threshold():
                return probability
            if rms >= max(0.045, float(getattr(self._config, "rms_threshold", 0.02)) * 1.6):
                return max(probability, self._speech_threshold())
            return probability
        return rms


class _SileroStreamingVad:
    def __init__(self, *, threshold: float) -> None:
        self._threshold = max(0.0, min(1.0, float(threshold)))
        self._model = None
        self._torch = None
        self._buffer = np.zeros(0, dtype=np.float32)
        self._last_probability = 0.0
        try:
            import torch  # type: ignore
            from silero_vad import load_silero_vad  # type: ignore

            self._torch = torch
            self._model = load_silero_vad(onnx=True)
        except Exception:
            self._model = None
            self._torch = None

    @property
    def available(self) -> bool:
        return self._model is not None and self._torch is not None

    def reset(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)
        self._last_probability = 0.0
        if self._model is not None and hasattr(self._model, "reset_states"):
            self._model.reset_states()

    def probability(self, *, chunk: np.ndarray, sample_rate: float) -> float:
        if not self.available:
            return 0.0
        resampled = _resample_linear(np.asarray(chunk, dtype=np.float32), sample_rate=int(sample_rate), target_rate=16000)
        if resampled.size == 0:
            return self._last_probability
        if self._buffer.size:
            self._buffer = np.concatenate((self._buffer, resampled))
        else:
            self._buffer = resampled
        probabilities: list[float] = []
        while self._buffer.size >= 512:
            frame = self._buffer[:512]
            self._buffer = self._buffer[512:]
            tensor = self._torch.from_numpy(frame.copy())
            prob = float(self._model(tensor, 16000).item())
            probabilities.append(prob)
        if probabilities:
            self._last_probability = max(probabilities)
        return self._last_probability


class _FunASRStreamingVad:
    def __init__(self, *, device: str) -> None:
        self._device = str(device or "cpu")
        self._cache: dict[str, object] = {}
        self._registry = get_funasr_registry()
        self._session_key = uuid4().hex

    @property
    def available(self) -> bool:
        return not bool(self.runtime_info().get("init_failure", ""))

    def reset(self) -> None:
        self._cache = {}

    def runtime_info(self) -> dict[str, object]:
        return self._registry.snapshot_vad(requested_device=self._device, session_key=self._session_key)

    def detect(self, *, chunk: np.ndarray, sample_rate: float) -> list[tuple[int, int]]:
        resampled = _resample_linear(np.asarray(chunk, dtype=np.float32), sample_rate=int(sample_rate), target_rate=16000)
        if resampled.size == 0:
            return []
        chunk_size_ms = max(1, int(round((resampled.size / 16000.0) * 1000.0)))
        try:
            handle = self._registry.get_vad(requested_device=self._device, session_key=self._session_key)
        except Exception:
            return []
        try:
            invoke_lock = handle.invoke_lock
            if invoke_lock is None:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    result = handle.model.generate(
                        input=resampled,
                        cache=self._cache,
                        is_final=False,
                        chunk_size=chunk_size_ms,
                    )
            else:
                with invoke_lock:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        result = handle.model.generate(
                            input=resampled,
                            cache=self._cache,
                            is_final=False,
                            chunk_size=chunk_size_ms,
                        )
        except Exception:
            return []
        return _extract_funasr_vad_segments(result)


def resolve_endpointing_backend_name(
    config_name: str,
    vad_config: VadSettings | None = None,
    *,
    resolved_backend_name: str = "",
) -> str:
    if resolved_backend_name == "funasr_v2":
        return "fsmn_vad"
    normalized = (config_name or "").strip().lower()
    if normalized in {"", "neural", "neural_endpoint"}:
        candidate = str(getattr(vad_config, "backend", "silero_vad") or "silero_vad").strip().lower()
        if candidate in {"", "neural", "neural_endpoint", "fsmn_vad", "fsmn-vad"}:
            return "silero_vad"
        return candidate
    return normalized


def build_endpointing_descriptor(
    name: str,
    config: VadSettings | None = None,
    *,
    resolved_backend_name: str = "",
) -> EndpointingDescriptor:
    resolved_name = resolve_endpointing_backend_name(name, config, resolved_backend_name=resolved_backend_name)
    return EndpointingDescriptor(
        name=resolved_name,
        mode="native_v2",
        notes="Endpointing runtime for ASR v2.",
    )


def build_endpointing_runtime(
    name: str,
    config: VadSettings | None = None,
    *,
    device: str = "cpu",
    resolved_backend_name: str = "",
) -> EndpointingRuntime:
    vad_config = config or VadSettings()
    descriptor = build_endpointing_descriptor(name, vad_config, resolved_backend_name=resolved_backend_name)
    patched_config = VadSettings(**asdict(vad_config))
    patched_config.backend = descriptor.name
    return EndpointingRuntime(descriptor, patched_config, device=device)


def _resample_linear(audio: np.ndarray, *, sample_rate: int, target_rate: int) -> np.ndarray:
    if sample_rate <= 0 or target_rate <= 0 or sample_rate == target_rate or audio.size <= 1:
        return audio.astype(np.float32, copy=False)
    src_len = int(audio.shape[0])
    dst_len = max(1, int(round(src_len * target_rate / sample_rate)))
    src_x = np.linspace(0.0, 1.0, src_len, endpoint=False)
    dst_x = np.linspace(0.0, 1.0, dst_len, endpoint=False)
    return np.interp(dst_x, src_x, audio).astype(np.float32, copy=False)


def _extract_funasr_vad_segments(result: object) -> list[tuple[int, int]]:
    if result is None:
        return []
    if isinstance(result, dict):
        value = result.get("value", [])
        return _normalize_funasr_vad_segments(value)
    if isinstance(result, list):
        segments: list[tuple[int, int]] = []
        for item in result:
            segments.extend(_extract_funasr_vad_segments(item))
        return segments
    return []


def _normalize_funasr_vad_segments(value: object) -> list[tuple[int, int]]:
    if not isinstance(value, list):
        return []
    segments: list[tuple[int, int]] = []
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                segments.append((int(item[0]), int(item[1])))
            except Exception:
                continue
    return segments


__all__ = [
    "EndpointSignal",
    "EndpointingDescriptor",
    "EndpointingRuntime",
    "build_endpointing_descriptor",
    "build_endpointing_runtime",
    "resolve_endpointing_backend_name",
]
