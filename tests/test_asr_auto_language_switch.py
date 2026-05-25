"""Tests for AutoLanguageState and observe_final_language (Phase 3)."""
from __future__ import annotations

import pytest

from app.infra.asr.auto_language_state import (
    AutoLanguageState,
    _classify_text_family,
    _is_too_short,
    observe_final_language,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

ZH_TEXT = "今天的會議討論了很多重要議題"   # 13 CJK chars — long enough to count
EN_TEXT = "the quick brown fox jumps over the lazy dog"  # 9 words
SHORT_ZH = "好"           # 1 CJK char — below _MIN_CHINESE_CHARS
SHORT_EN = "yes no"       # 2 words — below _MIN_LATIN_WORDS


# ---------------------------------------------------------------------------
# observe_final_language
# ---------------------------------------------------------------------------

class TestObserveFinalLanguage:
    def test_first_final_no_switch(self):
        state = AutoLanguageState(requested_language="auto")
        result = observe_final_language(state, "zh", ZH_TEXT, now_ms=1000)
        assert result is None
        assert state.chinese_streak == 1

    def test_non_auto_language_is_noop(self):
        state = AutoLanguageState(requested_language="en")
        result = observe_final_language(state, "zh", ZH_TEXT, now_ms=1000)
        assert result is None
        assert state.chinese_streak == 0

    def test_empty_requested_treated_as_auto(self):
        state = AutoLanguageState(requested_language="")
        for i in range(1, 4):
            result = observe_final_language(state, "zh", ZH_TEXT, now_ms=i * 1000)
        assert result == "zh-TW"

    def test_consecutive_chinese_triggers_switch(self):
        state = AutoLanguageState(requested_language="auto")
        results = [
            observe_final_language(state, "zh", ZH_TEXT, now_ms=i * 1000)
            for i in range(1, 4)
        ]
        assert results[-1] == "zh-TW"
        assert state.effective_language == "zh-TW"
        assert state.stable_family == "chinese"

    def test_below_threshold_no_switch(self):
        state = AutoLanguageState(requested_language="auto")
        for i in range(1, 3):  # only 2, threshold is 3
            result = observe_final_language(state, "zh", ZH_TEXT, now_ms=i * 1000)
        assert result is None
        assert state.effective_language == ""

    def test_switch_resets_streak(self):
        state = AutoLanguageState(requested_language="auto")
        for i in range(1, 4):
            observe_final_language(state, "zh", ZH_TEXT, now_ms=i * 1000)
        assert state.chinese_streak == 0

    def test_alternating_chinese_english_no_switch(self):
        state = AutoLanguageState(requested_language="auto")
        for i in range(10):
            lang = "zh" if i % 2 == 0 else "en"
            text = ZH_TEXT if i % 2 == 0 else EN_TEXT
            observe_final_language(state, lang, text, now_ms=i * 2000)
        assert state.effective_language == ""

    def test_too_short_not_counted(self):
        state = AutoLanguageState(requested_language="auto")
        for _ in range(10):
            result = observe_final_language(state, "zh", SHORT_ZH, now_ms=1000)
        assert result is None
        assert state.chinese_streak == 0

    def test_cooldown_blocks_immediate_switch_back(self):
        state = AutoLanguageState(requested_language="auto")
        for i in range(1, 4):
            observe_final_language(state, "zh", ZH_TEXT, now_ms=i * 1000)
        assert state.effective_language == "zh-TW"
        # Within cooldown window (< 25 s)
        for i in range(3):
            result = observe_final_language(state, "en", EN_TEXT, now_ms=5000 + i * 1000)
        assert result is None

    def test_cooldown_expires_and_allows_switch_back(self):
        state = AutoLanguageState(requested_language="auto")
        for i in range(1, 4):
            observe_final_language(state, "zh", ZH_TEXT, now_ms=i * 1000)
        assert state.effective_language == "zh-TW"
        # After 30 s (well past 25 s cooldown)
        for i in range(3):
            result = observe_final_language(state, "en", EN_TEXT, now_ms=30_000 + i * 1000)
        assert result == ""
        assert state.effective_language == ""

    def test_mixed_deflates_streak(self):
        state = AutoLanguageState(requested_language="auto")
        observe_final_language(state, "zh", ZH_TEXT, now_ms=1000)
        observe_final_language(state, "zh", ZH_TEXT, now_ms=2000)
        assert state.chinese_streak == 2
        # Text with ambiguous ratio + no detected_language → classified "mixed"
        # "12345 67890 你好": 3 words, CJK ratio ~0.17, Latin ratio 0 → mixed
        mixed_text = "12345 67890 你好"
        observe_final_language(state, "", mixed_text, now_ms=3000)
        assert state.chinese_streak == 1  # deflated by 1

    def test_consecutive_non_chinese_switch_to_empty(self):
        state = AutoLanguageState(requested_language="auto", effective_language="zh-TW")
        for i in range(3):
            result = observe_final_language(state, "en", EN_TEXT, now_ms=30_000 + i * 1000)
        assert result == ""
        assert state.effective_language == ""
        assert state.stable_family == "non_chinese"

    def test_updates_last_detected_language(self):
        state = AutoLanguageState(requested_language="auto")
        observe_final_language(state, "ja", ZH_TEXT, now_ms=1000)
        assert state.last_detected_language == "ja"


# ---------------------------------------------------------------------------
# _classify_text_family
# ---------------------------------------------------------------------------

class TestClassifyTextFamily:
    def test_cjk_heavy_is_chinese(self):
        assert _classify_text_family("zh", "今天的會議討論了很多重要議題") == "chinese"

    def test_latin_heavy_is_non_chinese(self):
        assert _classify_text_family("en", "the quick brown fox jumps over lazy dog") == "non_chinese"

    def test_empty_text_is_mixed(self):
        assert _classify_text_family("", "") == "mixed"

    def test_whitespace_only_is_mixed(self):
        assert _classify_text_family("zh", "   ") == "mixed"

    def test_latin_heavy_overrides_zh_tag(self):
        # Latin ratio >= 0.60 wins even with zh detected_language
        result = _classify_text_family("zh", "the quick brown fox jumps")
        assert result == "non_chinese"

    def test_zh_tag_tiebreaker_for_digits(self):
        # Digits-only: no CJK, no Latin → ambiguous → falls to tiebreaker
        result = _classify_text_family("zh", "12345 67890 99")
        assert result == "chinese"

    def test_unknown_lang_ambiguous_is_mixed(self):
        # Ambiguous script ratio + unknown language → mixed
        result = _classify_text_family("", "12345 67890 99")
        assert result == "mixed"

    def test_en_tag_non_chinese(self):
        result = _classify_text_family("en", "12345 numbers only")
        assert result == "non_chinese"


# ---------------------------------------------------------------------------
# _is_too_short
# ---------------------------------------------------------------------------

class TestIsTooShort:
    def test_single_cjk_too_short(self):
        assert _is_too_short("好", "chinese") is True

    def test_three_cjk_too_short(self):
        assert _is_too_short("你好嗎", "chinese") is True

    def test_four_cjk_not_too_short(self):
        assert _is_too_short("今天真的好", "chinese") is False

    def test_two_words_latin_too_short(self):
        assert _is_too_short("hello world", "non_chinese") is True

    def test_three_words_latin_ok(self):
        assert _is_too_short("hello world today", "non_chinese") is False

    def test_empty_too_short(self):
        assert _is_too_short("", "chinese") is True

    def test_whitespace_only_too_short(self):
        assert _is_too_short("   ", "chinese") is True

    def test_mixed_family_uses_word_count(self):
        # mixed family uses word-count heuristic
        assert _is_too_short("hello world", "mixed") is True
        assert _is_too_short("hello world today", "mixed") is False


# ---------------------------------------------------------------------------
# Manager-level integration
# ---------------------------------------------------------------------------

class TestASRManagerFixedLanguageRuntime:
    def _make_manager(self):
        from app.infra.asr.manager_v2 import ASRManagerV2
        from app.infra.config.schema import AppConfig
        return ASRManagerV2(AppConfig())

    def test_runtime_defaults_to_fixed_languages_not_auto(self):
        manager = self._make_manager()
        assert manager._effective_language_for_source("local") == "zh-TW"
        assert manager._effective_language_for_source("remote") == "en"

    def test_stats_show_fixed_requested_language_as_runtime_path(self):
        manager = self._make_manager()
        state = manager._auto_language_states["local"]
        for i in range(1, 4):
            observe_final_language(state, "zh", ZH_TEXT, now_ms=i * 1000)
        auto_lang = manager.stats()["local"]["auto_language"]
        assert auto_lang["requested"] == "zh-TW"
        assert auto_lang["effective"] == ""
