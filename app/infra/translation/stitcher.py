from __future__ import annotations

import time
import hashlib
import re
from collections import deque
from dataclasses import dataclass
from collections import OrderedDict

from app.infra.asr.stream_worker import AsrEvent
from app.infra.translation.provider import TranslationProvider
from app.infra.config.schema import TranslationProfileConfig


@dataclass(slots=True)
class StitchResult:
    text: str
    is_final: bool
    is_stable_partial: bool
    is_early_final: bool
    should_display: bool
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
        partial_interval_floor_ms: int = 320,
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
        self._min_partial_interval_ms = max(int(partial_interval_floor_ms), int(min_partial_interval_ms))
        self._exact_cache_size = max(32, int(exact_cache_size))
        self._prefix_min_delta_chars = max(1, int(prefix_min_delta_chars))
        self._exact_cache: OrderedDict[str, str] = OrderedDict()
        self._last_partial_source = ""
        self._last_partial_translation = ""
        self._last_skip_reason = ""

    def set_languages(self, *, source_lang: str, target_lang: str) -> None:
        new_source = (source_lang or "").strip()
        new_target = (target_lang or "").strip()
        if new_source == self._source_lang and new_target == self._target_lang:
            return
        self._source_lang = new_source
        self._target_lang = new_target
        self._context.clear()
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
            self._last_skip_reason = "empty_source_text"
            return None
        if text.startswith("[asr-error]"):
            self._last_skip_reason = "asr_error_marker"
            return None

        now_ms = int(time.monotonic() * 1000)
        effective_trigger = min(self._trigger_tokens, 28)
        units = self._estimated_units(text)
        segment_ms = max(0, int(event.end_ms) - int(event.start_ms))
        can_translate_partial = (
            (not event.is_final)
            and (
                units >= effective_trigger
                or (segment_ms >= 1400 and units >= 8)
            )
            and (now_ms - self._last_partial_call_ms >= self._min_partial_interval_ms)
        )
        if not event.is_final and not can_translate_partial:
            self._last_skip_reason = (
                f"partial_gate units={units} segment_ms={segment_ms} min_interval_ms={self._min_partial_interval_ms}"
            )
            return None

        if not event.is_final and self._last_partial_source:
            if text == self._last_partial_source and self._last_partial_translation:
                return StitchResult(
                    text=self._last_partial_translation,
                    is_final=False,
                    is_stable_partial=True,
                    is_early_final=False,
                    should_display=True,
                    should_speak=False,
                )
            if text.startswith(self._last_partial_source):
                delta = len(text) - len(self._last_partial_source)
                if delta < self._prefix_min_delta_chars and self._last_partial_translation:
                    return StitchResult(
                        text=self._last_partial_translation,
                        is_final=False,
                        is_stable_partial=True,
                        is_early_final=False,
                        should_display=True,
                        should_speak=False,
                    )

        context = list(self._context) if self._enabled and not event.is_final else []
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
            self._last_skip_reason = self._build_skip_reason("empty_translation")
            return None
        if _looks_like_format_contamination(translated):
            if event.is_final:
                self._last_partial_source = ""
                self._last_partial_translation = ""
            self._last_skip_reason = self._build_skip_reason("format_contamination", translated=translated)
            return None
        if self._target_lang.lower().startswith("zh") and not _looks_like_displayable_zh_translation(translated):
            self._last_skip_reason = self._build_skip_reason("non_displayable_zh", translated=translated)
            return None
        self._last_skip_reason = ""
        if not event.is_final:
            self._last_partial_call_ms = now_ms
            self._last_partial_source = text
            self._last_partial_translation = translated
        else:
            if self._enabled:
                self._context.append(text)
            self._last_partial_source = ""
            self._last_partial_translation = ""

        is_early_final = bool(getattr(event, "is_early_final", False))
        is_stable_partial = (not event.is_final) and translated == self._last_partial_translation
        should_speak = bool(event.is_final)
        return StitchResult(
            text=translated,
            is_final=event.is_final,
            is_stable_partial=is_stable_partial,
            is_early_final=is_early_final,
            should_display=True,
            should_speak=should_speak,
        )

    def last_skip_reason(self) -> str:
        return self._last_skip_reason

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

    def _build_skip_reason(self, reason: str, *, translated: str = "") -> str:
        debug_snapshot = getattr(self._translator, "debug_snapshot", lambda: {})()
        details: list[str] = [reason]
        cleaned = str(debug_snapshot.get("cleaned_response", "") or "")
        raw = str(debug_snapshot.get("raw_response", "") or "")
        last_error = str(debug_snapshot.get("last_error", "") or "")
        if translated:
            details.append(f"translated={translated[:120]!r}")
        if cleaned:
            details.append(f"cleaned={cleaned!r}")
        if raw:
            details.append(f"raw={raw!r}")
        if last_error:
            details.append(f"error={last_error!r}")
        return " | ".join(details)


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
    if cjk == 0:
        return False
    if ascii_letters == 0 and cjk <= 2:
        # Accept short confirmations like "是。" / "不。" as valid subtitles.
        return True
    return cjk >= max(2, ascii_letters // 2)


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
