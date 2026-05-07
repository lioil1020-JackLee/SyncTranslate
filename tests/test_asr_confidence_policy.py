"""Tests for ASR confidence-based suppression policy (Phase 2)."""
from __future__ import annotations

import pytest

from app.infra.asr.faster_whisper_adapter import TranscribeResult
from app.infra.asr.worker_v2 import SourceRuntimeV2


# ---------------------------------------------------------------------------
# Minimal fake result objects
# ---------------------------------------------------------------------------

class _FakeResult:
    def __init__(
        self,
        max_no_speech_prob: float | None = None,
        max_compression_ratio: float | None = None,
    ) -> None:
        self.max_no_speech_prob = max_no_speech_prob
        self.max_compression_ratio = max_compression_ratio


# ---------------------------------------------------------------------------
# _should_suppress_by_no_speech
# ---------------------------------------------------------------------------

class TestShouldSuppressByNoSpeech:
    def test_high_nsp_very_short_latin_suppressed(self):
        result = _FakeResult(max_no_speech_prob=0.80)
        assert SourceRuntimeV2._should_suppress_by_no_speech(result, "um") is True

    def test_high_nsp_short_noise_suppressed(self):
        result = _FakeResult(max_no_speech_prob=0.90)
        assert SourceRuntimeV2._should_suppress_by_no_speech(result, "uh") is True

    def test_threshold_above_075_triggers(self):
        result = _FakeResult(max_no_speech_prob=0.76)
        assert SourceRuntimeV2._should_suppress_by_no_speech(result, "ok") is True

    def test_threshold_at_075_does_not_trigger(self):
        # > 0.75 required, not >=
        result = _FakeResult(max_no_speech_prob=0.75)
        assert SourceRuntimeV2._should_suppress_by_no_speech(result, "ok") is False

    def test_below_threshold_not_suppressed(self):
        result = _FakeResult(max_no_speech_prob=0.5)
        assert SourceRuntimeV2._should_suppress_by_no_speech(result, "hello") is False

    def test_none_nsp_not_suppressed(self):
        result = _FakeResult(max_no_speech_prob=None)
        assert SourceRuntimeV2._should_suppress_by_no_speech(result, "hello") is False

    def test_high_nsp_long_latin_text_not_suppressed(self):
        # ≥ 3 words → substantive; should not suppress
        result = _FakeResult(max_no_speech_prob=0.95)
        text = "the quick brown fox jumps"
        assert SourceRuntimeV2._should_suppress_by_no_speech(result, text) is False

    def test_high_nsp_cjk_text_not_suppressed(self):
        # ≥ 4 CJK chars → do not suppress even with very high nsp
        result = _FakeResult(max_no_speech_prob=0.95)
        assert SourceRuntimeV2._should_suppress_by_no_speech(result, "今天天氣非常好") is False

    def test_high_nsp_three_cjk_not_suppressed(self):
        # Any CJK content → cjk_count > 0 → suppression guard applies (cjk_count==0 required)
        result = _FakeResult(max_no_speech_prob=0.85)
        assert SourceRuntimeV2._should_suppress_by_no_speech(result, "你好嗎") is False

    def test_real_transcribe_result_object(self):
        result = TranscribeResult(
            text="um",
            detected_language="en",
            max_no_speech_prob=0.88,
        )
        assert SourceRuntimeV2._should_suppress_by_no_speech(result, "um") is True


# ---------------------------------------------------------------------------
# _has_compression_loop_risk
# ---------------------------------------------------------------------------

class TestHasCompressionLoopRisk:
    def test_above_threshold_is_loop_risk(self):
        result = _FakeResult(max_compression_ratio=2.7)
        assert SourceRuntimeV2._has_compression_loop_risk(result) is True

    def test_well_above_threshold(self):
        result = _FakeResult(max_compression_ratio=5.0)
        assert SourceRuntimeV2._has_compression_loop_risk(result) is True

    def test_at_threshold_not_loop_risk(self):
        # > 2.6 required, not >=
        result = _FakeResult(max_compression_ratio=2.6)
        assert SourceRuntimeV2._has_compression_loop_risk(result) is False

    def test_below_threshold_not_loop_risk(self):
        result = _FakeResult(max_compression_ratio=1.5)
        assert SourceRuntimeV2._has_compression_loop_risk(result) is False

    def test_none_cr_not_loop_risk(self):
        result = _FakeResult(max_compression_ratio=None)
        assert SourceRuntimeV2._has_compression_loop_risk(result) is False

    def test_boundary_just_above_threshold(self):
        result = _FakeResult(max_compression_ratio=2.61)
        assert SourceRuntimeV2._has_compression_loop_risk(result) is True

    def test_real_transcribe_result_object(self):
        result = TranscribeResult(
            text="hello hello hello",
            detected_language="en",
            max_compression_ratio=3.2,
        )
        assert SourceRuntimeV2._has_compression_loop_risk(result) is True

    def test_normal_speech_no_loop_risk(self):
        result = TranscribeResult(
            text="today we discussed many important topics",
            detected_language="en",
            max_compression_ratio=1.1,
        )
        assert SourceRuntimeV2._has_compression_loop_risk(result) is False
