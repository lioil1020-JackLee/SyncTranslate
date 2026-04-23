from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import TYPE_CHECKING, Any

from app.infra.config.schema import DEFAULT_FIXED_LLM_MODEL, LlmConfig

if TYPE_CHECKING:
    from app.infra.translation.lm_studio_adapter import LmStudioClient


@dataclass(slots=True)
class AsrCorrectionResult:
    text: str
    raw_text: str
    applied: bool


class AsrTextCorrector:
    def __init__(
        self,
        config: LlmConfig,
        *,
        enabled: bool,
        context_items: int = 3,
        max_chars: int = 120,
    ) -> None:
        self._enabled = bool(enabled)
        self._context: deque[str] = deque(maxlen=max(0, int(context_items)))
        self._max_chars = max(24, int(max_chars))
        self._client: LmStudioClient | None | Any = None
        if self._enabled:
            from app.infra.translation.lm_studio_adapter import LmStudioClient

            self._client = LmStudioClient(
                base_url=config.base_url,
                model=DEFAULT_FIXED_LLM_MODEL,
                temperature=0.0,
                top_p=min(0.9, max(0.1, float(config.top_p))),
                max_output_tokens=min(192, max(64, int(config.max_output_tokens))),
                repeat_penalty=max(1.0, float(config.repeat_penalty)),
                stop_tokens=config.stop_tokens,
                request_timeout_sec=max(2.0, min(12.0, float(config.request_timeout_sec))),
            )

    def correct(self, text: str, *, language: str) -> AsrCorrectionResult:
        raw_text = (text or "").strip()
        if not raw_text:
            return AsrCorrectionResult(text="", raw_text="", applied=False)
        if (not self._enabled) or self._client is None or len(raw_text) > self._max_chars:
            self._remember(raw_text)
            return AsrCorrectionResult(text=raw_text, raw_text=raw_text, applied=False)
        try:
            corrected = self._client.correct_asr_text(
                text=raw_text,
                language=language,
                context=list(self._context),
            ).strip()
        except Exception:
            corrected = raw_text
        if not corrected:
            corrected = raw_text
        if not self._is_safe_correction(raw_text, corrected, language=language):
            corrected = raw_text
        applied = corrected != raw_text
        self._remember(corrected)
        return AsrCorrectionResult(text=corrected, raw_text=raw_text, applied=applied)

    def _remember(self, text: str) -> None:
        value = (text or "").strip()
        if value:
            self._context.append(value)

    @staticmethod
    def _is_safe_correction(raw_text: str, corrected: str, *, language: str) -> bool:
        raw = (raw_text or "").strip()
        candidate = (corrected or "").strip()
        if not raw or not candidate:
            return False
        if raw == candidate:
            return True
        if _looks_like_structured_junk(candidate):
            return False
        if len(candidate) > max(len(raw) + 12, int(len(raw) * 1.45) + 4):
            return False
        if len(candidate) < max(1, int(len(raw) * 0.45)) and len(raw) >= 8:
            return False

        normalized_raw = _normalize_for_compare(raw)
        normalized_candidate = _normalize_for_compare(candidate)
        ratio = SequenceMatcher(a=normalized_raw, b=normalized_candidate).ratio()
        if len(normalized_raw) >= 6 and ratio < 0.42:
            return False

        normalized_language = str(language or "").strip().lower()
        if normalized_language.startswith(("zh", "cmn", "yue")):
            if _contains_cjk(raw) and not _contains_cjk(candidate):
                return False
            if _looks_like_safe_cjk_surface_fix(raw, candidate):
                return True
            if len(normalized_raw) >= 10 and ratio < 0.55:
                return False
        return True


def _normalize_for_compare(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip().lower())


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in (text or ""))


def _looks_like_safe_cjk_surface_fix(raw_text: str, candidate_text: str) -> bool:
    raw = _strip_spacing_and_punctuation(raw_text)
    candidate = _strip_spacing_and_punctuation(candidate_text)
    if not raw or not candidate:
        return False
    if raw == candidate:
        return True
    if not (_is_all_cjk(raw) and _is_all_cjk(candidate)):
        return False
    if len(raw) != len(candidate):
        return False
    if not 2 <= len(raw) <= 12:
        return False
    if _ascii_digit_skeleton(raw_text) != _ascii_digit_skeleton(candidate_text):
        return False
    return True


def _strip_spacing_and_punctuation(text: str) -> str:
    return "".join(ch for ch in (text or "").strip() if _is_cjk_char(ch))


def _is_cjk_char(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def _is_all_cjk(text: str) -> bool:
    return bool(text) and all(_is_cjk_char(ch) for ch in text)


def _ascii_digit_skeleton(text: str) -> str:
    return "".join(ch for ch in (text or "") if ch.isascii() and ch.isdigit())


def _looks_like_structured_junk(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return True
    lowered = value.lower()
    if any(token in lowered for token in ("```", "\"correction\"", "\"translation\"", "<think", "</think")):
        return True
    if re.search(r"(?:^|\s)json(?:\s|$)", lowered):
        return True
    if "{" in value or "}" in value:
        return True
    return False


__all__ = ["AsrCorrectionResult", "AsrTextCorrector", "_looks_like_safe_cjk_surface_fix"]
