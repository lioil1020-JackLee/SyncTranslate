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


class VadSegmenter:
    def __init__(self, config: VadSettings) -> None:
        self._config = config
        self._runtime_min_silence_duration_ms: float | None = None
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._speech_holdoff_ms = 0.0  # elapsed ms since last real speech, within pad window
        self._speech_active = False
        self._last_rms = 0.0

    def reset(self) -> None:
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._speech_holdoff_ms = 0.0
        self._speech_active = False
        self._last_rms = 0.0

    @property
    def last_rms(self) -> float:
        return self._last_rms

    @property
    def effective_rms_threshold(self) -> float:
        # Honor the user-configured threshold from UI/config directly.
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
                split_seconds = min(float(self._config.max_speech_duration_s), 4.0)
                if self._speech_ms >= split_seconds * 1000.0:
                    finalize = True
                    force_split = True
            elif self._speech_active:
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
                self._speech_active = False

            return VadDecision(
                speech_active=self._speech_active,
                finalize=finalize,
                force_split=force_split,
                chunk_ms=chunk_ms,
                speech_ms=self._speech_ms,
                silence_ms=self._silence_ms,
            )

        rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
        self._last_rms = rms
        has_speech = rms >= self.effective_rms_threshold

        finalize = False
        force_split = False

        if has_speech:
            self._speech_holdoff_ms = 0.0  # reset holdoff whenever real speech is detected
            self._speech_ms += chunk_ms
            self._silence_ms = 0.0
            if self._speech_ms >= float(self._config.min_speech_duration_ms):
                self._speech_active = True
            if self._speech_ms >= self._config.max_speech_duration_s * 1000.0:
                finalize = True
                force_split = True
        else:
            if self._speech_active:
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
            self._speech_active = False

        return VadDecision(
            speech_active=self._speech_active,
            finalize=finalize,
            force_split=force_split,
            chunk_ms=chunk_ms,
            speech_ms=self._speech_ms,
            silence_ms=self._silence_ms,
        )
