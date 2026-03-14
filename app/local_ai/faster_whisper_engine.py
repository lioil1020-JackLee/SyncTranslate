from __future__ import annotations

import io
from threading import Lock
import wave
from dataclasses import dataclass, field

import numpy as np


_TRANSCRIBE_LOCK = Lock()


@dataclass(slots=True)
class FasterWhisperEngine:
    model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    beam_size: int = 3
    final_beam_size: int = 3
    condition_on_previous_text: bool = True
    final_condition_on_previous_text: bool = False
    temperature_fallback: str = "0.0,0.2,0.4"
    no_speech_threshold: float = 0.55
    language: str = ""
    _model: object | None = field(default=None, init=False, repr=False)
    _runtime_device: str | None = field(default=None, init=False, repr=False)
    _runtime_compute_type: str | None = field(default=None, init=False, repr=False)
    _fallback_reason: str = field(default="", init=False, repr=False)

    def transcribe_partial(self, audio: np.ndarray, sample_rate: int) -> str:
        if audio.size == 0:
            return ""
        # Partial transcripts should stay cheap; keep this lightweight.
        text = self._transcribe(
            audio=audio,
            sample_rate=sample_rate,
            vad_filter=False,
            beam_size=max(1, int(self.beam_size)),
            condition_on_previous_text=bool(self.condition_on_previous_text),
        )
        return text.strip()

    def transcribe_final(self, audio: np.ndarray, sample_rate: int) -> str:
        if audio.size == 0:
            return ""
        # Disable Whisper's internal VAD for final transcription: our custom VAD already
        # segmented the audio.  vad_filter=True re-scans and can clip trailing words that
        # fall just below the RMS threshold at utterance end.
        text = self._transcribe(
            audio=audio,
            sample_rate=sample_rate,
            vad_filter=False,
            beam_size=max(1, int(self.final_beam_size)),
            condition_on_previous_text=bool(self.final_condition_on_previous_text),
        )
        return text.strip()

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

    def _transcribe(
        self,
        *,
        audio: np.ndarray,
        sample_rate: int,
        vad_filter: bool,
        beam_size: int,
        condition_on_previous_text: bool,
    ) -> str:
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
            # Pass audio as an in-memory BinaryIO to avoid temp-file I/O while
            # still using the stable BinaryIO code path in faster_whisper.
            # Serialize transcribe to prevent concurrent native decoder crashes.
            with _TRANSCRIBE_LOCK:
                segments, _ = model.transcribe(audio_bio, **kwargs)
            return "".join(str(getattr(seg, "text", "")) for seg in segments).strip()
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
            # Remove DC offset and normalize low-volume input to improve ASR recall.
            # Higher target_rms (0.12) and gain cap (8x) ensures quiet trailing
            # words get sufficient amplitude for Whisper to recognise them.
            mono = mono - float(np.mean(mono))
            rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
            if rms > 1e-6:
                target_rms = 0.12
                gain = min(8.0, target_rms / rms)
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
        self._model = WhisperModel(
            self.model or "large-v3",
            device=self._runtime_device,
            compute_type=self._runtime_compute_type,
        )
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
        self._runtime_device = "cpu"
        self._runtime_compute_type = "int8"


def _normalize_lang(language: str) -> str:
    text = (language or "").strip().lower()
    if not text:
        return ""
    if "-" in text:
        return text.split("-", 1)[0]
    return text


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
