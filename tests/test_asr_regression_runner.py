"""Tests for regression runner aggregation and threshold logic."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.asr_benchmark.run_regression_corpus import (
    SampleConfig,
    SampleResult,
    aggregate_summary,
    check_thresholds,
    write_summary_json,
    write_summary_md,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sample(
    sample_id: str = "s1",
    category: str = "news",
    min_accuracy: dict | None = None,
    max_dropped_chunks: int = 0,
    max_repetition_ratio: float = 0.02,
) -> SampleConfig:
    return SampleConfig(
        id=sample_id,
        language="zh-TW",
        category=category,
        audio="audio/test.wav",
        reference="refs/test.txt",
        min_accuracy=min_accuracy if min_accuracy is not None else {"news_turbo": 0.85},
        max_dropped_chunks=max_dropped_chunks,
        max_repetition_ratio=max_repetition_ratio,
    )


def _make_result(**kwargs) -> SampleResult:
    defaults: dict = dict(
        sample_id="s1",
        model="turbo",
        mode="meeting",
        accuracy=0.90,
        dropped_chunks=0,
        repetition_ratio=0.01,
        passed=False,
        failure_reasons=[],
        error="",
    )
    defaults.update(kwargs)
    return SampleResult(**defaults)


# ---------------------------------------------------------------------------
# check_thresholds
# ---------------------------------------------------------------------------

class TestCheckThresholds:
    def test_all_pass(self):
        sample = _make_sample(min_accuracy={"news_turbo": 0.85})
        result = _make_result(model="turbo", mode="meeting", accuracy=0.90)
        assert check_thresholds(result, sample) == []

    def test_accuracy_below_threshold_fails(self):
        sample = _make_sample(min_accuracy={"news_turbo": 0.85})
        result = _make_result(model="turbo", mode="meeting", accuracy=0.80)
        failures = check_thresholds(result, sample)
        assert len(failures) == 1
        assert "accuracy" in failures[0]

    def test_dropped_chunks_over_limit_fails(self):
        sample = _make_sample(max_dropped_chunks=0)
        result = _make_result(dropped_chunks=3)
        failures = check_thresholds(result, sample)
        assert any("dropped" in f for f in failures)

    def test_repetition_ratio_over_limit_fails(self):
        sample = _make_sample(max_repetition_ratio=0.02)
        result = _make_result(repetition_ratio=0.05)
        failures = check_thresholds(result, sample)
        assert any("repetition" in f for f in failures)

    def test_multiple_failures_all_reported(self):
        sample = _make_sample(
            min_accuracy={"news_turbo": 0.85},
            max_dropped_chunks=0,
            max_repetition_ratio=0.02,
        )
        result = _make_result(accuracy=0.60, dropped_chunks=5, repetition_ratio=0.10)
        failures = check_thresholds(result, sample)
        assert len(failures) == 3

    def test_no_min_accuracy_key_skips_accuracy_check(self):
        sample = _make_sample(min_accuracy={})
        result = _make_result(accuracy=0.10)
        failures = check_thresholds(result, sample)
        assert not any("accuracy" in f for f in failures)

    def test_model_name_fallback_key(self):
        # "category_model" key missing; falls back to bare model name
        sample = _make_sample(category="news", min_accuracy={"turbo": 0.85})
        result = _make_result(model="turbo", mode="meeting", accuracy=0.80)
        failures = check_thresholds(result, sample)
        assert any("accuracy" in f for f in failures)

    def test_none_accuracy_skips_check(self):
        sample = _make_sample(min_accuracy={"news_turbo": 0.85})
        result = _make_result(accuracy=None)
        failures = check_thresholds(result, sample)
        assert not any("accuracy" in f for f in failures)

    def test_accuracy_exactly_at_threshold_passes(self):
        sample = _make_sample(min_accuracy={"news_turbo": 0.85})
        result = _make_result(model="turbo", accuracy=0.85)
        failures = check_thresholds(result, sample)
        assert not any("accuracy" in f for f in failures)


# ---------------------------------------------------------------------------
# aggregate_summary
# ---------------------------------------------------------------------------

class TestAggregateSummary:
    def test_all_pass(self):
        results = [_make_result(passed=True), _make_result(sample_id="s2", passed=True)]
        summary = aggregate_summary(results)
        assert summary.total == 2
        assert summary.passed == 2
        assert summary.failed == 0
        assert summary.errors == 0

    def test_all_fail(self):
        results = [
            _make_result(passed=False, failure_reasons=["accuracy low"]),
            _make_result(sample_id="s2", passed=False, failure_reasons=["accuracy low"]),
        ]
        summary = aggregate_summary(results)
        assert summary.failed == 2
        assert summary.passed == 0

    def test_errors_counted_separately(self):
        results = [
            _make_result(passed=True),
            _make_result(sample_id="s2", error="timeout"),
            _make_result(sample_id="s3", passed=False, failure_reasons=["accuracy low"]),
        ]
        summary = aggregate_summary(results)
        assert summary.total == 3
        assert summary.passed == 1
        assert summary.failed == 1
        assert summary.errors == 1

    def test_empty_results(self):
        summary = aggregate_summary([])
        assert summary.total == 0
        assert summary.passed == 0
        assert summary.failed == 0
        assert summary.errors == 0

    def test_results_list_preserved(self):
        results = [_make_result(passed=True)]
        summary = aggregate_summary(results)
        assert len(summary.results) == 1
        assert summary.results[0].sample_id == "s1"


# ---------------------------------------------------------------------------
# write_summary_json
# ---------------------------------------------------------------------------

class TestWriteSummaryJson:
    def test_creates_file(self, tmp_path):
        summary = aggregate_summary([_make_result(passed=True)])
        path = write_summary_json(summary, tmp_path)
        assert path.is_file()
        assert path.name == "summary.json"

    def test_valid_json_content(self, tmp_path):
        results = [
            _make_result(passed=True),
            _make_result(
                sample_id="s2",
                passed=False,
                failure_reasons=["accuracy 0.70 < min 0.85"],
            ),
        ]
        summary = aggregate_summary(results)
        path = write_summary_json(summary, tmp_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["total"] == 2
        assert data["passed"] == 1
        assert data["failed"] == 1
        assert len(data["results"]) == 2
        assert data["results"][1]["failure_reasons"] == ["accuracy 0.70 < min 0.85"]

    def test_creates_nested_output_dir(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        summary = aggregate_summary([_make_result(passed=True)])
        path = write_summary_json(summary, nested)
        assert path.is_file()

    def test_none_accuracy_serialises_as_null(self, tmp_path):
        summary = aggregate_summary([_make_result(accuracy=None, error="timeout")])
        path = write_summary_json(summary, tmp_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["results"][0]["accuracy"] is None


# ---------------------------------------------------------------------------
# write_summary_md
# ---------------------------------------------------------------------------

class TestWriteSummaryMd:
    def test_creates_file(self, tmp_path):
        summary = aggregate_summary([_make_result(passed=True)])
        path = write_summary_md(summary, tmp_path)
        assert path.is_file()
        assert path.name == "summary.md"

    def test_contains_result_row(self, tmp_path):
        summary = aggregate_summary([_make_result(accuracy=0.92, passed=True)])
        path = write_summary_md(summary, tmp_path)
        content = path.read_text(encoding="utf-8")
        assert "| s1 |" in content
        assert "0.920" in content
        assert "✓" in content

    def test_fail_row_shows_cross(self, tmp_path):
        r = _make_result(passed=False, failure_reasons=["accuracy low"])
        summary = aggregate_summary([r])
        path = write_summary_md(summary, tmp_path)
        content = path.read_text(encoding="utf-8")
        assert "✗" in content

    def test_error_row_shows_err(self, tmp_path):
        r = _make_result(error="timeout")
        summary = aggregate_summary([r])
        path = write_summary_md(summary, tmp_path)
        content = path.read_text(encoding="utf-8")
        assert "ERR" in content

    def test_failures_section_present(self, tmp_path):
        r = _make_result(passed=False, failure_reasons=["accuracy 0.70 < 0.85"])
        summary = aggregate_summary([r])
        path = write_summary_md(summary, tmp_path)
        content = path.read_text(encoding="utf-8")
        assert "Failures" in content
        assert "accuracy" in content

    def test_no_failures_section_when_all_pass(self, tmp_path):
        summary = aggregate_summary([_make_result(passed=True)])
        path = write_summary_md(summary, tmp_path)
        content = path.read_text(encoding="utf-8")
        assert "Failures" not in content

    def test_header_line_present(self, tmp_path):
        summary = aggregate_summary([_make_result(passed=True)])
        path = write_summary_md(summary, tmp_path)
        content = path.read_text(encoding="utf-8")
        assert "# ASR Regression Summary" in content
        assert "Total:" in content
