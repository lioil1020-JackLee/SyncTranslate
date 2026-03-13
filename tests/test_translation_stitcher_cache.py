from __future__ import annotations

import unittest

from app.local_ai.streaming_asr import AsrEvent
from app.local_ai.translation_stitcher import TranslationStitcher
from app.schemas import TranslationProfileConfig


class _FakeTranslator:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def translate(self, text: str, **kwargs) -> str:
        self.calls.append(text)
        return "這是測試翻譯"


class TranslationStitcherCacheTests(unittest.TestCase):
    def _event(self, text: str, *, is_final: bool, start_ms: int = 0, end_ms: int = 1500) -> AsrEvent:
        return AsrEvent(text=text, is_final=is_final, start_ms=start_ms, end_ms=end_ms, latency_ms=0)

    def test_exact_cache_reuses_translation(self) -> None:
        fake = _FakeTranslator()
        stitcher = TranslationStitcher(
            translator=fake,
            source_lang="en",
            target_lang="zh-TW",
            profile=TranslationProfileConfig(),
            enabled=False,
            trigger_tokens=1,
            min_partial_interval_ms=0,
            exact_cache_size=8,
        )
        first = stitcher.process(self._event("hello world", is_final=True))
        second = stitcher.process(self._event("hello world", is_final=True))
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(len(fake.calls), 1)

    def test_prefix_small_delta_reuses_last_partial(self) -> None:
        fake = _FakeTranslator()
        stitcher = TranslationStitcher(
            translator=fake,
            source_lang="en",
            target_lang="zh-TW",
            profile=TranslationProfileConfig(),
            enabled=False,
            trigger_tokens=1,
            min_partial_interval_ms=0,
            prefix_min_delta_chars=4,
        )
        p1 = stitcher.process(self._event("hello team this is a long partial", is_final=False))
        stitcher._last_partial_call_ms = 0  # type: ignore[attr-defined]
        p2 = stitcher.process(self._event("hello team this is a long partial!", is_final=False))
        self.assertIsNotNone(p1)
        self.assertIsNotNone(p2)
        self.assertEqual(len(fake.calls), 1)


if __name__ == "__main__":
    unittest.main()
