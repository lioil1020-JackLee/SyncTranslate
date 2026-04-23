from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np


@dataclass(slots=True)
class TranscriptValidationResult:
    text: str
    accepted: bool
    reason: str = ""
    score: float = 1.0


class AsrTranscriptValidatorV2:
    def __init__(
        self,
        *,
        enabled: bool = True,
        max_chars_per_second: float = 14.0,
        max_repeat_span: int = 2,
        min_speech_ratio_for_long_text: float = 0.18,
    ) -> None:
        self._enabled = bool(enabled)
        self._max_chars_per_second = max(4.0, float(max_chars_per_second))
        self._max_repeat_span = max(1, int(max_repeat_span))
        self._min_speech_ratio_for_long_text = max(0.0, min(1.0, float(min_speech_ratio_for_long_text)))

    def validate(
        self,
        text: str,
        *,
        audio: np.ndarray,
        sample_rate: int,
        language: str,
        frontend_stats: dict[str, object] | None = None,
        is_final: bool = False,
    ) -> TranscriptValidationResult:
        value = self._sanitize(text)
        if not self._enabled:
            return TranscriptValidationResult(text=value, accepted=bool(value), reason="", score=1.0)
        if not value:
            return TranscriptValidationResult(text="", accepted=False, reason="empty", score=0.0)

        is_cjk_language = self._is_cjk_family_language(language)
        duration_sec = max(0.001, float(np.asarray(audio).reshape(-1).size) / float(max(1, sample_rate)))
        chars_per_second = len(value) / duration_sec
        speech_ratio = self._speech_ratio_from_stats_or_audio(audio, frontend_stats=frontend_stats)
        if self._looks_like_markup_leak(value):
            return TranscriptValidationResult(text="", accepted=False, reason="markup-leak", score=0.0)
        # Non-CJK transcripts naturally need more characters to represent the same content.
        # Relax density threshold to avoid over-filtering English/Japanese/Korean outputs.
        density_limit = self._max_chars_per_second if is_cjk_language else self._max_chars_per_second * 2.2
        if chars_per_second > density_limit:
            return TranscriptValidationResult(text="", accepted=False, reason="too-dense", score=0.0)
        # Low speech-ratio gating is mainly useful on CJK channels where short noise bursts
        # may produce fluent hallucinations; keep non-CJK path less aggressive.
        effective_min_speech_ratio = self._min_speech_ratio_for_long_text * (0.32 if is_final else 0.72)
        long_cjk_text = len(value) >= (20 if is_final else 16)
        sustained_audio = duration_sec >= (1.2 if is_final else 0.9)
        if (
            is_cjk_language
            and long_cjk_text
            and sustained_audio
            and chars_per_second >= 3.0
            and speech_ratio < effective_min_speech_ratio
        ):
            return TranscriptValidationResult(text="", accepted=False, reason="low-speech-ratio", score=0.0)
        if self._looks_like_loop(value):
            return TranscriptValidationResult(text="", accepted=False, reason="looped-phrase", score=0.0)
        if language.lower().startswith(("zh", "cmn", "yue")) and self._looks_like_sparse_cjk_noise(value):
            return TranscriptValidationResult(text="", accepted=False, reason="sparse-cjk-noise", score=0.0)
        score = max(0.0, 1.0 - max(0.0, chars_per_second - 5.0) / max(1.0, density_limit))
        return TranscriptValidationResult(text=value, accepted=True, reason="", score=score)

    @staticmethod
    def _is_cjk_family_language(language: str) -> bool:
        normalized = str(language or "").strip().lower()
        return normalized.startswith(("zh", "cmn", "yue"))

    @staticmethod
    def _sanitize(text: str) -> str:
        value = str(text or "").strip()
        if not value:
            return ""
        value = re.sub(r"\s+", " ", value)
        value = re.sub(r"([。！？!?，,；;])\1{1,}", r"\1", value)
        return value.strip()

    def _looks_like_loop(self, text: str) -> bool:
        compact = re.sub(r"\s+", "", text)
        if len(compact) < 6:
            return False
        for span in range(1, min(self._max_repeat_span + 1, len(compact) // 2 + 1)):
            unit = compact[:span]
            repeated = unit * (len(compact) // span)
            if compact.startswith(repeated[: len(compact)]) and compact.count(unit) >= 3:
                return True
        return False

    @staticmethod
    def _looks_like_markup_leak(text: str) -> bool:
        lowered = text.strip().lower()
        if not lowered:
            return False
        if "```" in lowered:
            return True
        if re.search(r"</?[a-z][a-z0-9_-]*>", lowered):
            return True
        suspicious_tokens = (
            "<solution",
            "</solution",
            "<analysis",
            "</analysis",
            "<final",
            "</final",
            "<assistant",
            "</assistant",
            "<user",
            "</user",
        )
        if any(token in lowered for token in suspicious_tokens):
            return True
        if lowered.startswith(("assistant:", "user:", "system:", "translation:", "output:")):
            return True
        return False

    @staticmethod
    def _looks_like_sparse_cjk_noise(text: str) -> bool:
        cjk = sum("\u4e00" <= ch <= "\u9fff" for ch in text)
        punctuation = sum(ch in "，。！？、；：" for ch in text)
        return cjk > 0 and cjk <= 4 and punctuation == 0 and len(text) >= 6

    def _speech_ratio_from_stats_or_audio(self, audio: np.ndarray, *, frontend_stats: dict[str, object] | None) -> float:
        if frontend_stats:
            try:
                ratio = float(frontend_stats.get("speech_ratio", 0.0))
                if ratio > 0.0:
                    return ratio
            except Exception:
                pass
        signal = np.asarray(audio, dtype=np.float32).reshape(-1)
        if signal.size == 0:
            return 0.0
        frame = max(160, int(16000 * 0.02))
        baseline = float(np.sqrt(np.mean(np.square(signal), dtype=np.float32)))
        threshold = max(0.003, baseline * 1.35)
        voiced = 0
        total = 0
        for start in range(0, max(1, signal.size - frame + 1), frame):
            end = min(signal.size, start + frame)
            rms = float(np.sqrt(np.mean(np.square(signal[start:end]), dtype=np.float32)))
            if rms >= threshold:
                voiced += 1
            total += 1
        return float(voiced) / float(max(1, total))


__all__ = ["AsrTranscriptValidatorV2", "TranscriptValidationResult"]
