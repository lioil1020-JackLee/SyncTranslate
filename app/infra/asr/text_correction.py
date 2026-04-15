from __future__ import annotations

from collections import deque
from dataclasses import dataclass
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
        applied = corrected != raw_text
        self._remember(corrected)
        return AsrCorrectionResult(text=corrected, raw_text=raw_text, applied=applied)

    def _remember(self, text: str) -> None:
        value = (text or "").strip()
        if value:
            self._context.append(value)


__all__ = ["AsrCorrectionResult", "AsrTextCorrector"]
