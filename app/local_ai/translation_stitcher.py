from __future__ import annotations

import time
import hashlib
import re
from collections import deque
from dataclasses import dataclass
from collections import OrderedDict

from app.local_ai.llm_provider import TranslationProvider
from app.local_ai.streaming_asr import AsrEvent
from app.schemas import TranslationProfileConfig


@dataclass(slots=True)
class StitchResult:
    text: str
    is_final: bool
    should_speak: bool


class TranslationStitcher:
    def __init__(
        self,
        *,
        translator: TranslationProvider,
        source_lang: str,
        target_lang: str,
        profile: TranslationProfileConfig | None = None,
        enabled: bool = True,
        trigger_tokens: int = 20,
        max_context_items: int = 6,
        min_partial_interval_ms: int = 600,
        exact_cache_size: int = 256,
        prefix_min_delta_chars: int = 6,
    ) -> None:
        self._translator = translator
        self._source_lang = source_lang
        self._target_lang = target_lang
        self._profile = profile
        self._enabled = bool(enabled)
        self._trigger_tokens = max(8, int(trigger_tokens))
        self._context: deque[str] = deque(maxlen=max(2, int(max_context_items)))
        self._last_partial_call_ms = 0
        self._min_partial_interval_ms = max(300, int(min_partial_interval_ms))
        self._last_spoken = ""
        self._exact_cache_size = max(32, int(exact_cache_size))
        self._prefix_min_delta_chars = max(1, int(prefix_min_delta_chars))
        self._exact_cache: OrderedDict[str, str] = OrderedDict()
        self._last_partial_source = ""
        self._last_partial_translation = ""

    @staticmethod
    def _estimated_units(text: str) -> int:
        # CJK sentences often contain no spaces; split()-based token counting
        # underestimates partial readiness and delays translation.
        cjk_count = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        if cjk_count >= 4:
            return max(1, cjk_count // 2)
        words = [part for part in text.split() if part.strip()]
        return max(1, len(words))

    def process(self, event: AsrEvent) -> StitchResult | None:
        text = event.text.strip()
        if not text:
            return None
        if text.startswith("[asr-error]"):
            return None

        now_ms = int(time.monotonic() * 1000)
        effective_trigger = min(self._trigger_tokens, 24)
        units = self._estimated_units(text)
        segment_ms = max(0, int(event.end_ms) - int(event.start_ms))
        can_translate_partial = (
            (not event.is_final)
            and (
                units >= effective_trigger
                or (segment_ms >= 900 and units >= 6)
            )
            and (now_ms - self._last_partial_call_ms >= self._min_partial_interval_ms)
        )
        if not event.is_final and not can_translate_partial:
            return None

        if not event.is_final and self._last_partial_source:
            if text == self._last_partial_source and self._last_partial_translation:
                return StitchResult(text=self._last_partial_translation, is_final=False, should_speak=False)
            if text.startswith(self._last_partial_source):
                delta = len(text) - len(self._last_partial_source)
                if delta < self._prefix_min_delta_chars and self._last_partial_translation:
                    return StitchResult(text=self._last_partial_translation, is_final=False, should_speak=False)

        context = list(self._context) if self._enabled else []
        cache_key = self._cache_key(text, context)
        translated = self._read_exact_cache(cache_key)
        if not translated:
            translated = self._translator.translate(
                text=text,
                source_lang=self._source_lang,
                target_lang=self._target_lang,
                context=context,
                profile=self._profile,
            )
            if translated:
                self._write_exact_cache(cache_key, translated)
        if not translated:
            return None
        if _looks_like_format_contamination(translated):
            if event.is_final:
                self._last_partial_source = ""
                self._last_partial_translation = ""
            return None
        if self._target_lang.lower().startswith("zh") and not _looks_like_displayable_zh_translation(translated):
            return None
        if self._enabled:
            self._context.append(text)
        if not event.is_final:
            self._last_partial_call_ms = now_ms
            self._last_partial_source = text
            self._last_partial_translation = translated
        else:
            self._last_partial_source = ""
            self._last_partial_translation = ""

        should_speak = event.is_final and translated != self._last_spoken
        if should_speak:
            self._last_spoken = translated
        return StitchResult(text=translated, is_final=event.is_final, should_speak=should_speak)

    def _cache_key(self, text: str, context: list[str] | None = None) -> str:
        profile_name = (self._profile.name.strip() if self._profile else "default") or "default"
        context_hash = self._context_fingerprint(context or [])
        return f"{self._source_lang}>{self._target_lang}|profile={profile_name}|ctx={context_hash}|{text}"

    @staticmethod
    def _context_fingerprint(context: list[str]) -> str:
        if not context:
            return "-"
        normalized = "\n".join(item.strip() for item in context if item.strip())
        if not normalized:
            return "-"
        return hashlib.blake2s(normalized.encode("utf-8"), digest_size=8).hexdigest()

    def _read_exact_cache(self, key: str) -> str:
        hit = self._exact_cache.get(key, "")
        if hit:
            self._exact_cache.move_to_end(key)
        return hit

    def _write_exact_cache(self, key: str, value: str) -> None:
        self._exact_cache[key] = value
        self._exact_cache.move_to_end(key)
        while len(self._exact_cache) > self._exact_cache_size:
            self._exact_cache.popitem(last=False)


def _looks_like_displayable_zh_translation(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return False
    lowered = value.lower()
    banned = (
        "thinking process",
        "analyze the request",
        "common phrasing",
        "interpretation",
        "translation:",
        "->",
    )
    if any(token in lowered for token in banned):
        return False
    if _looks_like_format_contamination(value):
        return False
    cjk = sum("\u4e00" <= ch <= "\u9fff" for ch in value)
    ascii_letters = sum(ch.isascii() and ch.isalpha() for ch in value)
    return cjk > 0 and cjk >= max(2, ascii_letters // 2)


def _looks_like_format_contamination(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return True
    lowered = value.lower()
    markers = (
        "<translation",
        "</translation",
        "<target",
        "</target",
        "<p>",
        "</p>",
        "```",
        "{\"translation\"",
    )
    if any(marker in lowered for marker in markers):
        return True
    if re.search(r"</?[^>\n]{1,120}>", value):
        return True
    if value.count("<") + value.count(">") >= 4:
        return True
    return False
