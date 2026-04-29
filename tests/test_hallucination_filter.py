"""Tests for the hallucination-filter helpers in app.infra.asr._hallucination_filter.

These tests were extracted from test_multilingual_channel_policy.py to focus
exclusively on the stateless filter functions.
"""
from __future__ import annotations

import unittest

from app.infra.asr._hallucination_filter import (
    _looks_like_known_non_speech_text,
    _looks_like_script_mismatch_junk,
    _looks_like_silence_hallucination,
    _transcript_drop_reason,
)


class HallucinationFilterTests(unittest.TestCase):
    def test_short_low_energy_thank_you_like_text_is_treated_as_hallucination(self) -> None:
        self.assertTrue(
            _looks_like_silence_hallucination("Thank you all.", audio_ms=900, vad_rms=0.01)
        )
        self.assertTrue(
            _looks_like_silence_hallucination("謝謝大家", audio_ms=900, vad_rms=0.01)
        )
        self.assertTrue(
            _looks_like_silence_hallucination("by bwd6", audio_ms=900, vad_rms=0.0)
        )
        self.assertTrue(
            _looks_like_silence_hallucination("晚安", audio_ms=1200, vad_rms=0.01)
        )
        self.assertFalse(
            _looks_like_silence_hallucination("Can you hear me now?", audio_ms=1200, vad_rms=0.04)
        )
        self.assertTrue(
            _looks_like_silence_hallucination("Thank you for watching.", audio_ms=1200, vad_rms=0.01)
        )
        self.assertTrue(
            _looks_like_silence_hallucination("感謝您的收看", audio_ms=1200, vad_rms=0.01)
        )

    def test_known_non_speech_overlay_lines_are_filtered(self) -> None:
        self.assertTrue(
            _looks_like_known_non_speech_text("優優獨播劇場——YoYo Television Series Exclusive")
        )
        self.assertTrue(
            _looks_like_known_non_speech_text("字幕由 Amara.org 社群提供")
        )
        self.assertTrue(
            _looks_like_known_non_speech_text("請不吝點贊 訂閱 轉發 打賞支援明鏡與點點欄目")
        )
        self.assertTrue(
            _looks_like_known_non_speech_text("MING PAO CANADA MING PAO TORONTO")
        )
        self.assertFalse(
            _looks_like_known_non_speech_text("你趕快修煉我來處理船上的黑衣樓殺手")
        )

    def test_short_foreign_script_junk_is_filtered_when_language_is_pinned(self) -> None:
        self.assertTrue(
            _looks_like_script_mismatch_junk("Ак", expected_language="zh")
        )
        self.assertTrue(
            _looks_like_script_mismatch_junk("Ак", expected_language="en")
        )
        self.assertFalse(
            _looks_like_script_mismatch_junk("你好", expected_language="zh")
        )
        self.assertFalse(
            _looks_like_script_mismatch_junk("OK", expected_language="en")
        )

    def test_transcript_drop_reason_prioritizes_overlay_and_script_junk(self) -> None:
        self.assertEqual(
            _transcript_drop_reason(
                "謝謝觀看,下次見!",
                audio_ms=2400,
                vad_rms=0.08,
                expected_language="zh",
            ),
            "non-speech-overlay",
        )
        self.assertEqual(
            _transcript_drop_reason(
                "Ак",
                audio_ms=900,
                vad_rms=0.01,
                expected_language="zh",
            ),
            "script-mismatch",
        )
        self.assertEqual(
            _transcript_drop_reason(
                "by bwd6",
                audio_ms=900,
                vad_rms=0.0,
                expected_language="zh",
            ),
            "hallucinated",
        )
