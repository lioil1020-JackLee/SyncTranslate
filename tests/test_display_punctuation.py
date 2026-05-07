"""Tests for display_punctuation (Phase 3)."""
from __future__ import annotations

import pytest

from app.application.display_punctuation import apply_display_punctuation


class TestApplyDisplayPunctuation:
    # --- guard conditions ---

    def test_disabled_passthrough(self):
        text = "今天的會議討論了很多重要議題"
        assert apply_display_punctuation(text, is_final=True, enabled=False) == text

    def test_partial_passthrough(self):
        text = "今天的會議討論了很多重要議題"
        assert apply_display_punctuation(text, is_final=False, enabled=True) == text

    def test_empty_string(self):
        assert apply_display_punctuation("", is_final=True, enabled=True) == ""

    def test_whitespace_only(self):
        text = "   "
        assert apply_display_punctuation(text, is_final=True, enabled=True) == text

    # --- existing terminal punctuation is not duplicated ---

    def test_already_has_period(self):
        text = "今天的會議討論了很多重要議題。"
        assert apply_display_punctuation(text, is_final=True, enabled=True) == text

    def test_already_has_question_mark(self):
        text = "你覺得這個問題嗎？"
        assert apply_display_punctuation(text, is_final=True, enabled=True) == text

    def test_already_has_exclamation(self):
        text = "今天的會議討論了很多重要議題！"
        assert apply_display_punctuation(text, is_final=True, enabled=True) == text

    def test_already_has_ellipsis(self):
        text = "今天的會議討論了很多重要議題…"
        assert apply_display_punctuation(text, is_final=True, enabled=True) == text

    # --- period appended for long CJK finals ---

    def test_long_cjk_gets_period(self):
        text = "今天的會議討論了很多重要議題"   # 13 CJK chars
        result = apply_display_punctuation(text, is_final=True, enabled=True)
        assert result.endswith("。")
        assert result == text.rstrip() + "。"

    def test_exactly_8_cjk_gets_period(self):
        text = "今天天氣非常好啊"   # 8 CJK chars
        result = apply_display_punctuation(text, is_final=True, enabled=True)
        assert result.endswith("。")

    def test_trailing_whitespace_trimmed_before_appending(self):
        text = "今天的會議討論了很多重要議題  "
        result = apply_display_punctuation(text, is_final=True, enabled=True)
        assert result.endswith("。")
        assert not result.endswith("  。")

    # --- question mark for question words ---

    def test_question_particle_ma(self):
        text = "你覺得這個問題嗎"
        result = apply_display_punctuation(text, is_final=True, enabled=True)
        assert result.endswith("？")

    def test_question_particle_ne(self):
        text = "你今天要去哪裡呢"
        result = apply_display_punctuation(text, is_final=True, enabled=True)
        assert result.endswith("？")

    def test_question_particle_ba(self):
        text = "你是對的吧"
        result = apply_display_punctuation(text, is_final=True, enabled=True)
        assert result.endswith("？")

    def test_question_word_shen_at_start(self):
        text = "什麼意思啊"
        result = apply_display_punctuation(text, is_final=True, enabled=True)
        assert result.endswith("？")

    def test_question_word_zen_at_start(self):
        text = "怎麼回事"
        result = apply_display_punctuation(text, is_final=True, enabled=True)
        assert result.endswith("？")

    # --- short fillers not modified ---

    def test_single_cjk_filler_unchanged(self):
        assert apply_display_punctuation("好", is_final=True, enabled=True) == "好"

    def test_three_cjk_filler_unchanged(self):
        assert apply_display_punctuation("對不起", is_final=True, enabled=True) == "對不起"

    def test_under_4_cjk_unchanged(self):
        # 3 CJK chars → below min threshold (cjk < 4 guard fires first)
        # no punctuation added even though 嗎 is a question word
        text = "你好嗎"
        result = apply_display_punctuation(text, is_final=True, enabled=True)
        assert result == text

    def test_four_cjk_no_question_no_period(self):
        # 4 CJK chars: enough for question-word check but < 8 for period
        # "今天好啊" has no question word → no punctuation added
        text = "今天好啊"
        result = apply_display_punctuation(text, is_final=True, enabled=True)
        assert result == text

    # --- non-CJK text untouched ---

    def test_english_text_unchanged(self):
        text = "the quick brown fox jumps over the lazy dog"
        assert apply_display_punctuation(text, is_final=True, enabled=True) == text

    def test_mixed_short_latin_unchanged(self):
        text = "hello world"
        assert apply_display_punctuation(text, is_final=True, enabled=True) == text
