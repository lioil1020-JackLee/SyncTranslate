from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from app.local_ai.ollama_client import OllamaClient
from app.local_ai.streaming_asr import AsrEvent


@dataclass(slots=True)
class StitchResult:
    text: str
    is_final: bool
    should_speak: bool


class TranslationStitcher:
    def __init__(
        self,
        *,
        translator: OllamaClient,
        source_lang: str,
        target_lang: str,
        trigger_tokens: int = 20,
        max_context_items: int = 6,
        min_partial_interval_ms: int = 800,
    ) -> None:
        self._translator = translator
        self._source_lang = source_lang
        self._target_lang = target_lang
        self._trigger_tokens = max(8, int(trigger_tokens))
        self._context: deque[str] = deque(maxlen=max(2, int(max_context_items)))
        self._last_partial_call_ms = 0
        self._min_partial_interval_ms = max(300, int(min_partial_interval_ms))
        self._last_spoken = ""

    def process(self, event: AsrEvent) -> StitchResult | None:
        text = event.text.strip()
        if not text:
            return None
        if text.startswith("[asr-error]"):
            return None

        now_ms = int(time.monotonic() * 1000)
        can_translate_partial = (
            (not event.is_final)
            and len(text.split()) >= self._trigger_tokens
            and (now_ms - self._last_partial_call_ms >= self._min_partial_interval_ms)
        )
        if not event.is_final and not can_translate_partial:
            return None

        translated = self._translator.translate(
            text=text,
            source_lang=self._source_lang,
            target_lang=self._target_lang,
            context=list(self._context),
        )
        if not translated:
            return None
        self._context.append(text)
        if not event.is_final:
            self._last_partial_call_ms = now_ms

        should_speak = event.is_final and translated != self._last_spoken
        if should_speak:
            self._last_spoken = translated
        return StitchResult(text=translated, is_final=event.is_final, should_speak=should_speak)
