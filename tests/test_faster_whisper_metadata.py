"""Tests for FasterWhisper SegmentMetadata and TranscribeResult (Phase 2)."""
from __future__ import annotations

import pytest

from app.infra.asr.faster_whisper_adapter import (
    SegmentMetadata,
    TranscribeResult,
    _collect_segment_metadata,
    _safe_float,
)


# ---------------------------------------------------------------------------
# Fake segment object that mirrors faster-whisper segment attributes
# ---------------------------------------------------------------------------

class _FakeSeg:
    def __init__(
        self,
        text: str = "",
        start: float = 0.0,
        end: float = 1.0,
        avg_logprob: float | None = None,
        no_speech_prob: float | None = None,
        compression_ratio: float | None = None,
    ) -> None:
        self.text = text
        self.start = start
        self.end = end
        self.avg_logprob = avg_logprob
        self.no_speech_prob = no_speech_prob
        self.compression_ratio = compression_ratio


# ---------------------------------------------------------------------------
# _safe_float
# ---------------------------------------------------------------------------

class TestSafeFloat:
    def test_none_returns_none(self):
        assert _safe_float(None) is None

    def test_float_passthrough(self):
        assert _safe_float(0.5) == pytest.approx(0.5)

    def test_int_converts(self):
        assert _safe_float(1) == pytest.approx(1.0)

    def test_string_float_converts(self):
        assert _safe_float("0.75") == pytest.approx(0.75)

    def test_invalid_string_returns_none(self):
        assert _safe_float("abc") is None

    def test_object_returns_none(self):
        assert _safe_float(object()) is None


# ---------------------------------------------------------------------------
# SegmentMetadata
# ---------------------------------------------------------------------------

class TestSegmentMetadata:
    def test_required_fields(self):
        seg = SegmentMetadata(text="hello", start=0.0, end=1.5)
        assert seg.text == "hello"
        assert seg.start == pytest.approx(0.0)
        assert seg.end == pytest.approx(1.5)

    def test_optional_fields_default_to_none(self):
        seg = SegmentMetadata(text="", start=0.0, end=0.0)
        assert seg.avg_logprob is None
        assert seg.no_speech_prob is None
        assert seg.compression_ratio is None

    def test_optional_fields_set(self):
        seg = SegmentMetadata(
            text="test", start=0.0, end=2.0,
            avg_logprob=-0.5, no_speech_prob=0.1, compression_ratio=1.2,
        )
        assert seg.avg_logprob == pytest.approx(-0.5)
        assert seg.no_speech_prob == pytest.approx(0.1)
        assert seg.compression_ratio == pytest.approx(1.2)


# ---------------------------------------------------------------------------
# TranscribeResult — backwards compatibility
# ---------------------------------------------------------------------------

class TestTranscribeResult:
    def test_minimal_construction(self):
        result = TranscribeResult(text="hello", detected_language="en")
        assert result.text == "hello"
        assert result.detected_language == "en"

    def test_new_fields_default_to_none(self):
        result = TranscribeResult(text="hello", detected_language="en")
        assert result.segments == ()
        assert result.avg_logprob is None
        assert result.max_no_speech_prob is None
        assert result.max_compression_ratio is None

    def test_full_construction(self):
        seg = SegmentMetadata(text="hi", start=0.0, end=1.0, avg_logprob=-0.3)
        result = TranscribeResult(
            text="hi",
            detected_language="en",
            segments=(seg,),
            avg_logprob=-0.3,
            max_no_speech_prob=0.2,
            max_compression_ratio=1.4,
        )
        assert len(result.segments) == 1
        assert result.avg_logprob == pytest.approx(-0.3)
        assert result.max_no_speech_prob == pytest.approx(0.2)
        assert result.max_compression_ratio == pytest.approx(1.4)


# ---------------------------------------------------------------------------
# _collect_segment_metadata
# ---------------------------------------------------------------------------

class TestCollectSegmentMetadata:
    def test_empty_segments(self):
        metas, lp, nsp, cr = _collect_segment_metadata([])
        assert metas == []
        assert lp is None
        assert nsp is None
        assert cr is None

    def test_single_segment_all_fields(self):
        segs = [_FakeSeg(text="hello", start=0.0, end=2.0,
                         avg_logprob=-0.5, no_speech_prob=0.1, compression_ratio=1.2)]
        metas, lp, nsp, cr = _collect_segment_metadata(segs)
        assert len(metas) == 1
        assert metas[0].text == "hello"
        assert lp == pytest.approx(-0.5)
        assert nsp == pytest.approx(0.1)
        assert cr == pytest.approx(1.2)

    def test_duration_weighted_logprob(self):
        # seg1: 2 s × -0.4 = -0.8;  seg2: 1 s × -1.0 = -1.0
        # weighted = (-0.8 + -1.0) / (2+1) = -0.6
        segs = [
            _FakeSeg(text="a", start=0.0, end=2.0, avg_logprob=-0.4),
            _FakeSeg(text="b", start=2.0, end=3.0, avg_logprob=-1.0),
        ]
        _, lp, _, _ = _collect_segment_metadata(segs)
        assert lp == pytest.approx(-0.6, abs=1e-6)

    def test_max_no_speech_prob(self):
        segs = [
            _FakeSeg(text="a", start=0.0, end=1.0, no_speech_prob=0.3),
            _FakeSeg(text="b", start=1.0, end=2.0, no_speech_prob=0.8),
            _FakeSeg(text="c", start=2.0, end=3.0, no_speech_prob=0.1),
        ]
        _, _, nsp, _ = _collect_segment_metadata(segs)
        assert nsp == pytest.approx(0.8)

    def test_max_compression_ratio(self):
        segs = [
            _FakeSeg(text="a", start=0.0, end=1.0, compression_ratio=1.1),
            _FakeSeg(text="b", start=1.0, end=2.0, compression_ratio=2.9),
            _FakeSeg(text="c", start=2.0, end=3.0, compression_ratio=1.5),
        ]
        _, _, _, cr = _collect_segment_metadata(segs)
        assert cr == pytest.approx(2.9)

    def test_missing_metadata_fields_no_error(self):
        """Old-style segment objects without metadata attributes."""
        class _OldSeg:
            text = "hello"
            start = 0.0
            end = 1.0

        metas, lp, nsp, cr = _collect_segment_metadata([_OldSeg()])
        assert len(metas) == 1
        assert lp is None
        assert nsp is None
        assert cr is None

    def test_zero_duration_fallback_to_mean(self):
        # All zero-duration → weighted sum / 0 → fallback to simple mean
        segs = [
            _FakeSeg(text="a", start=0.0, end=0.0, avg_logprob=-0.4),
            _FakeSeg(text="b", start=0.0, end=0.0, avg_logprob=-1.0),
        ]
        _, lp, _, _ = _collect_segment_metadata(segs)
        assert lp == pytest.approx(-0.7, abs=1e-6)

    def test_generator_input(self):
        """Should materialise a lazy generator without error."""
        def _gen():
            yield _FakeSeg(text="a", start=0.0, end=1.0, no_speech_prob=0.5)
            yield _FakeSeg(text="b", start=1.0, end=2.0, no_speech_prob=0.3)

        metas, _, nsp, _ = _collect_segment_metadata(_gen())
        assert len(metas) == 2
        assert nsp == pytest.approx(0.5)

    def test_partial_metadata_fields(self):
        """Mix of segments with and without optional fields."""
        segs = [
            _FakeSeg(text="a", start=0.0, end=1.0, avg_logprob=-0.4),
            _FakeSeg(text="b", start=1.0, end=2.0),   # no metadata
        ]
        metas, lp, nsp, cr = _collect_segment_metadata(segs)
        assert len(metas) == 2
        assert lp == pytest.approx(-0.4)   # only seg1 contributes
        assert nsp is None
        assert cr is None
