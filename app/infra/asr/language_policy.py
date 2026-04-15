from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.infra.config.schema import VadSettings


@dataclass(slots=True)
class VadDecision:
    speech_active: bool
    finalize: bool
    force_split: bool
    chunk_ms: float
    speech_ms: float = 0.0
    silence_ms: float = 0.0
    pause_ms: float = 0.0


class VadSegmenter:
    def __init__(self, config: VadSettings) -> None:
        self._config = config
        self._backend = str(getattr(config, "backend", "rms") or "rms").strip().lower()
        self._runtime_min_silence_duration_ms: float | None = None
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._speech_holdoff_ms = 0.0  # elapsed ms since last real speech, within pad window
        self._pause_ms = 0.0
        self._speech_active = False
        self._last_rms = 0.0
        self._neural_vad = _SileroStreamingVad(threshold=float(getattr(config, "neural_threshold", 0.5))) if self._backend in {"silero", "silero_vad"} else None

    def reset(self) -> None:
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._speech_holdoff_ms = 0.0
        self._pause_ms = 0.0
        self._speech_active = False
        self._last_rms = 0.0
        if self._neural_vad is not None:
            self._neural_vad.reset()

    @property
    def last_rms(self) -> float:
        return self._last_rms

    @property
    def effective_rms_threshold(self) -> float:
        if self._neural_vad is not None and self._neural_vad.available:
            return max(0.0, min(1.0, float(getattr(self._config, "neural_threshold", 0.5))))
        return max(0.0, float(self._config.rms_threshold))

    @property
    def effective_min_silence_duration_ms(self) -> float:
        if self._runtime_min_silence_duration_ms is not None:
            return max(120.0, float(self._runtime_min_silence_duration_ms))
        return max(120.0, float(self._config.min_silence_duration_ms))

    def set_runtime_tuning(self, *, min_silence_duration_ms: int | None = None) -> None:
        if min_silence_duration_ms is None:
            self._runtime_min_silence_duration_ms = None
            return
        self._runtime_min_silence_duration_ms = max(120.0, float(min_silence_duration_ms))

    def update(self, chunk: np.ndarray, sample_rate: float) -> VadDecision:
        if sample_rate <= 0:
            return VadDecision(speech_active=False, finalize=False, force_split=False, chunk_ms=0.0)
        chunk_ms = (len(chunk) / float(sample_rate)) * 1000.0
        if not self._config.enabled:
            rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
            self._last_rms = rms
            has_signal = rms >= self.effective_rms_threshold
            finalize = False
            force_split = False

            if has_signal:
                self._speech_active = True
                self._speech_ms += chunk_ms
                self._silence_ms = 0.0
                self._speech_holdoff_ms = 0.0
                self._pause_ms = 0.0
                split_seconds = min(float(self._config.max_speech_duration_s), 4.0)
                if self._speech_ms >= split_seconds * 1000.0:
                    finalize = True
                    force_split = True
            elif self._speech_active:
                self._pause_ms += chunk_ms
                pad_ms = float(self._config.speech_pad_ms)
                if self._speech_holdoff_ms < pad_ms:
                    self._speech_holdoff_ms += chunk_ms
                else:
                    self._silence_ms += chunk_ms
                    if self._silence_ms >= self.effective_min_silence_duration_ms:
                        finalize = True

            if finalize:
                self._speech_ms = 0.0
                self._silence_ms = 0.0
                self._speech_holdoff_ms = 0.0
                self._pause_ms = 0.0
                self._speech_active = False

            return VadDecision(
                speech_active=self._speech_active,
                finalize=finalize,
                force_split=force_split,
                chunk_ms=chunk_ms,
                speech_ms=self._speech_ms,
                silence_ms=self._silence_ms,
                pause_ms=self._pause_ms,
            )

        rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
        self._last_rms = rms
        has_speech = self._has_speech_signal(chunk=chunk, sample_rate=sample_rate, rms=rms)

        finalize = False
        force_split = False

        if has_speech:
            self._speech_holdoff_ms = 0.0  # reset holdoff whenever real speech is detected
            self._pause_ms = 0.0
            self._speech_ms += chunk_ms
            self._silence_ms = 0.0
            if self._speech_ms >= float(self._config.min_speech_duration_ms):
                self._speech_active = True
            if self._speech_ms >= self._config.max_speech_duration_s * 1000.0:
                finalize = True
                force_split = True
        else:
            if self._speech_active:
                self._pause_ms += chunk_ms
                pad_ms = float(self._config.speech_pad_ms)
                if self._speech_holdoff_ms < pad_ms:
                    # Within the speech-pad window: treat as speech continuation so
                    # trailing words are not dropped before silence counting begins.
                    self._speech_holdoff_ms += chunk_ms
                else:
                    self._silence_ms += chunk_ms
                    if self._silence_ms >= self.effective_min_silence_duration_ms:
                        finalize = True

        if finalize:
            self._speech_ms = 0.0
            self._silence_ms = 0.0
            self._speech_holdoff_ms = 0.0
            self._pause_ms = 0.0
            self._speech_active = False

        return VadDecision(
            speech_active=self._speech_active,
            finalize=finalize,
            force_split=force_split,
            chunk_ms=chunk_ms,
            speech_ms=self._speech_ms,
            silence_ms=self._silence_ms,
            pause_ms=self._pause_ms,
        )

    def _has_speech_signal(self, *, chunk: np.ndarray, sample_rate: float, rms: float) -> bool:
        if self._neural_vad is not None and self._neural_vad.available:
            speech_prob = self._neural_vad.probability(chunk=chunk, sample_rate=sample_rate)
            if speech_prob >= self.effective_rms_threshold:
                return True
            # Let clear high-energy chunks still count as speech if the neural model is uncertain.
            if rms >= max(0.045, float(self._config.rms_threshold) * 1.6):
                return True
            return False
        return rms >= self.effective_rms_threshold


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
        mono = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if mono.size == 0:
            return self._last_probability
        resampled = _resample_linear(mono, sample_rate=int(sample_rate), target_rate=16000)
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


def _resample_linear(audio: np.ndarray, *, sample_rate: int, target_rate: int) -> np.ndarray:
    if sample_rate <= 0 or target_rate <= 0 or sample_rate == target_rate or audio.size <= 1:
        return audio.astype(np.float32, copy=False)
    src_len = int(audio.shape[0])
    dst_len = max(1, int(round(src_len * target_rate / sample_rate)))
    src_x = np.linspace(0.0, 1.0, src_len, endpoint=False)
    dst_x = np.linspace(0.0, 1.0, dst_len, endpoint=False)
    return np.interp(dst_x, src_x, audio).astype(np.float32, copy=False)
