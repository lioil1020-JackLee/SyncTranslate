from __future__ import annotations

import io
from threading import Lock
import wave
from dataclasses import dataclass, field

import numpy as np


_MODEL_CACHE_LOCK = Lock()
_MODEL_CACHE: dict[tuple[str, str, str], tuple[object, Lock]] = {}


@dataclass(slots=True)
class FasterWhisperEngine:
    model: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "float16"
    beam_size: int = 1
    final_beam_size: int = 3
    condition_on_previous_text: bool = True
    final_condition_on_previous_text: bool = False
    initial_prompt: str = ""
    hotwords: str = ""
    speculative_draft_model: str = ""
    speculative_num_beams: int = 1
    temperature_fallback: str = "0.0,0.2,0.4"
    no_speech_threshold: float = 0.55
    language: str = ""
    _model: object | None = field(default=None, init=False, repr=False)
    _runtime_device: str | None = field(default=None, init=False, repr=False)
    _runtime_compute_type: str | None = field(default=None, init=False, repr=False)
    _fallback_reason: str = field(default="", init=False, repr=False)
    _transcribe_lock: Lock | None = field(default=None, init=False, repr=False)
    _context_window: list[str] = field(default_factory=list, init=False, repr=False)
    _context_max_chars: int = field(default=200, init=False, repr=False)
    _draft_model: object | None = field(default=None, init=False, repr=False)
    _draft_failed: bool = field(default=False, init=False, repr=False)

    def transcribe_partial(self, audio: np.ndarray, sample_rate: int) -> str:
        return self.transcribe_partial_result(audio, sample_rate).text

    def transcribe_partial_result(self, audio: np.ndarray, sample_rate: int) -> TranscribeResult:
        if audio.size == 0:
            return TranscribeResult(text="", detected_language="")
        # Partial transcripts should stay cheap; keep this lightweight.
        result = self._transcribe(
            audio=audio,
            sample_rate=sample_rate,
            vad_filter=False,
            beam_size=max(1, int(self.beam_size)),
            condition_on_previous_text=bool(self.condition_on_previous_text),
        )
        return TranscribeResult(
            text=result.text.strip(),
            detected_language=_normalize_lang(result.detected_language or self.language),
        )

    def transcribe_final(self, audio: np.ndarray, sample_rate: int) -> str:
        return self.transcribe_final_result(audio, sample_rate).text

    def transcribe_final_result(self, audio: np.ndarray, sample_rate: int) -> TranscribeResult:
        if audio.size == 0:
            return TranscribeResult(text="", detected_language="")
        # Disable Whisper's internal VAD for final transcription: our custom VAD already
        # segmented the audio.  vad_filter=True re-scans and can clip trailing words that
        # fall just below the RMS threshold at utterance end.
        result = self._transcribe(
            audio=audio,
            sample_rate=sample_rate,
            vad_filter=False,
            beam_size=max(1, int(self.final_beam_size)),
            condition_on_previous_text=bool(self.final_condition_on_previous_text),
        )
        return TranscribeResult(
            text=result.text.strip(),
            detected_language=_normalize_lang(result.detected_language or self.language),
        )

    def warmup(self) -> None:
        self._get_model()

    def health_check(self) -> tuple[bool, str]:
        try:
            silence = np.zeros((16000, 1), dtype=np.float32)
            self.transcribe_partial(silence, 16000)
        except Exception as exc:
            return False, str(exc)
        runtime = self.runtime_label()
        return True, runtime or "ready"

    def runtime_label(self) -> str:
        device = self._runtime_device or self.device or "auto"
        compute_type = self._runtime_compute_type or self.compute_type or ""
        suffix = f" ({compute_type})" if compute_type else ""
        if self._fallback_reason:
            return f"{device}{suffix} fallback: {self._fallback_reason}"
        return f"{device}{suffix}"

    def push_context(self, text: str) -> None:
        stripped = str(text or "").strip()
        if not stripped:
            return
        self._context_window.append(stripped)
        while len(self._context_window) > 1 and sum(len(item) for item in self._context_window) > self._context_max_chars:
            self._context_window.pop(0)

    def _effective_initial_prompt(self) -> str:
        parts: list[str] = []
        base_prompt = self.initial_prompt.strip()
        if base_prompt:
            parts.append(base_prompt)
        if self._context_window:
            parts.append(" ".join(self._context_window))
        return " ".join(parts).strip()

    def _transcribe(
        self,
        *,
        audio: np.ndarray,
        sample_rate: int,
        vad_filter: bool,
        beam_size: int,
        condition_on_previous_text: bool,
    ) -> TranscribeResult:
        model = self._get_model()
        wav_audio, wav_sample_rate = self._prepare_audio(audio=audio, sample_rate=sample_rate)
        audio_bio = self._encode_wav_bytes(audio=wav_audio, sample_rate=wav_sample_rate)
        try:
            kwargs: dict[str, object] = {
                "beam_size": max(1, int(beam_size)),
                "condition_on_previous_text": bool(condition_on_previous_text),
                "vad_filter": vad_filter,
                "temperature": _parse_temperature_fallback(self.temperature_fallback),
                "no_speech_threshold": max(0.0, min(1.0, float(self.no_speech_threshold))),
            }
            normalized_language = _normalize_lang(self.language)
            if normalized_language:
                kwargs["language"] = normalized_language
            effective_prompt = self._effective_initial_prompt()
            if effective_prompt:
                kwargs["initial_prompt"] = effective_prompt
            if self.hotwords.strip():
                kwargs["hotwords"] = self.hotwords.strip()
            draft_model = self._get_draft_model()
            if draft_model is not None:
                kwargs["assistant_model"] = draft_model
            # Pass audio as an in-memory BinaryIO to avoid temp-file I/O while
            # still using the stable BinaryIO code path in faster_whisper.
            # Engines with the same runtime share a model and lock so we avoid
            # loading multiple large-v3 models onto the GPU while keeping access
            # to that shared decoder serialized.
            transcribe_lock = self._transcribe_lock or Lock()
            self._transcribe_lock = transcribe_lock
            with transcribe_lock:
                try:
                    segments, info = model.transcribe(audio_bio, **kwargs)
                except TypeError as exc:
                    # Keep compatibility with older faster-whisper versions.
                    message = str(exc)
                    retriable = False
                    if "assistant_model" in message and "assistant_model" in kwargs:
                        kwargs.pop("assistant_model", None)
                        self._draft_failed = True
                        retriable = True
                    if "hotwords" in message and "hotwords" in kwargs:
                        kwargs.pop("hotwords", None)
                        retriable = True
                    if retriable:
                        audio_bio.seek(0)
                        segments, info = model.transcribe(audio_bio, **kwargs)
                    else:
                        raise
            detected_language = ""
            if isinstance(info, dict):
                detected_language = str(info.get("language", "")).strip()
            else:
                detected_language = str(getattr(info, "language", "")).strip()
            return TranscribeResult(
                text="".join(str(getattr(seg, "text", "")) for seg in segments).strip(),
                detected_language=_normalize_lang(detected_language),
            )
        except RuntimeError as exc:
            if self._should_fallback_to_cpu(exc):
                self._fallback_to_cpu(str(exc))
                return self._transcribe(
                    audio=audio,
                    sample_rate=sample_rate,
                    vad_filter=vad_filter,
                    beam_size=beam_size,
                    condition_on_previous_text=condition_on_previous_text,
                )
            raise

    @staticmethod
    def _prepare_audio(*, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        mono = audio
        if mono.ndim == 2 and mono.shape[1] > 1:
            mono = mono[:, :1]
        mono = mono.reshape(-1).astype(np.float32, copy=False)
        if mono.size > 0:
            # Remove DC offset and apply only mild gain compensation.
            # Meeting audio from virtual devices often already contains codec noise
            # and room reverb; aggressive normalization can turn that into fluent
            # but incorrect Whisper output.
            mono = mono - float(np.mean(mono))
            rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
            if rms > 1e-6:
                target_rms = 0.05
                gain = min(3.0, target_rms / rms)
                mono = np.clip(mono * gain, -1.0, 1.0)
        target_sample_rate = 16000
        if sample_rate <= 0 or sample_rate == target_sample_rate or mono.size <= 1:
            return mono, int(sample_rate if sample_rate > 0 else target_sample_rate)

        src_len = int(mono.shape[0])
        dst_len = max(1, int(round(src_len * target_sample_rate / sample_rate)))
        src_x = np.linspace(0.0, 1.0, src_len, endpoint=False)
        dst_x = np.linspace(0.0, 1.0, dst_len, endpoint=False)
        resampled = np.interp(dst_x, src_x, mono).astype(np.float32, copy=False)
        return resampled, target_sample_rate

    @staticmethod
    def _encode_wav_bytes(*, audio: np.ndarray, sample_rate: int) -> io.BytesIO:
        """Encode float32 mono PCM as WAV in-memory (no temp file)."""
        mono = audio.reshape(-1)
        int16_data = np.clip(mono, -1.0, 1.0)
        int16_data = (int16_data * 32767.0).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(int16_data.tobytes())
        buf.seek(0)
        return buf

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except Exception as exc:
            raise ValueError(
                "faster-whisper not available. Install optional dependency: uv sync --extra local"
            ) from exc
        self._runtime_device = self._runtime_device or (self.device or "auto")
        self._runtime_compute_type = self._runtime_compute_type or self._resolve_compute_type(self._runtime_device)
        cache_key = (
            self.model or "large-v3-turbo",
            self._runtime_device,
            self._runtime_compute_type,
        )
        with _MODEL_CACHE_LOCK:
            cached = _MODEL_CACHE.get(cache_key)
            if cached is None:
                cached = (
                    WhisperModel(
                        cache_key[0],
                        device=cache_key[1],
                        compute_type=cache_key[2],
                    ),
                    Lock(),
                )
                _MODEL_CACHE[cache_key] = cached
        self._model, self._transcribe_lock = cached
        return self._model

    def _resolve_compute_type(self, runtime_device: str) -> str:
        requested = (self.compute_type or "int8_float16").strip() or "int8_float16"
        if runtime_device == "cpu" and requested == "float16":
            self._fallback_reason = self._fallback_reason or "cpu does not support efficient float16; switched to int8"
            return "int8"
        return requested

    def _should_fallback_to_cpu(self, exc: RuntimeError) -> bool:
        message = str(exc).lower()
        if (self._runtime_device or self.device or "auto") == "cpu":
            return False
        return any(token in message for token in ("cublas", "cudnn", "cuda", "cannot be loaded"))

    def _fallback_to_cpu(self, reason: str) -> None:
        self._fallback_reason = reason
        self._model = None
        self._draft_model = None
        self._runtime_device = "cpu"
        self._runtime_compute_type = "int8"

    def _get_draft_model(self):
        if self._draft_failed:
            return None
        if self._draft_model is not None:
            return self._draft_model
        draft_name = self.speculative_draft_model.strip()
        if not draft_name:
            return None
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except Exception:
            self._draft_failed = True
            return None
        try:
            self._draft_model = WhisperModel(
                draft_name,
                device=self._runtime_device or self.device or "auto",
                compute_type=self._runtime_compute_type or self.compute_type,
            )
        except Exception:
            self._draft_failed = True
            self._draft_model = None
        return self._draft_model


def _clear_model_cache_for_tests() -> None:
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()


def _normalize_lang(language: str) -> str:
    text = (language or "").strip().lower()
    if not text:
        return ""
    if "-" in text:
        return text.split("-", 1)[0]
    return text


@dataclass(slots=True)
class TranscribeResult:
    text: str
    detected_language: str


def _parse_temperature_fallback(raw: str) -> list[float]:
    text = (raw or "").strip()
    if not text:
        return [0.0, 0.2, 0.4]
    values: list[float] = []
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            value = float(token)
        except Exception:
            continue
        value = max(0.0, min(2.0, value))
        if value not in values:
            values.append(value)
    return values or [0.0, 0.2, 0.4]
