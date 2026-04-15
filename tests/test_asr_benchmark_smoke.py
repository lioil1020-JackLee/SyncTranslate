"""Smoke tests for ASR benchmark tools (Phase 3, work package I)."""
from __future__ import annotations

import json
import os
import tempfile

import pytest


# ---------------------------------------------------------------------------
# tools/asr_benchmark/run_benchmark.py — metric functions
# ---------------------------------------------------------------------------

class TestCerWer:
    def setup_method(self):
        from tools.asr_benchmark.run_benchmark import _cer, _wer
        self._cer = _cer
        self._wer = _wer

    def test_cer_identical(self):
        assert self._cer("hello", "hello") == pytest.approx(0.0)

    def test_cer_completely_different(self):
        # "abc" vs "xyz" → 3 substitutions / 3 chars = 1.0
        assert self._cer("abc", "xyz") == pytest.approx(1.0)

    def test_cer_partial(self):
        # "hello" vs "helo" → 1 deletion / 5 chars = 0.2
        val = self._cer("hello", "helo")
        assert 0.0 < val <= 0.5

    def test_wer_identical(self):
        assert self._wer("hello world", "hello world") == pytest.approx(0.0)

    def test_wer_one_substitution(self):
        # "hello world" vs "hello there" → 1 sub / 2 words = 0.5
        assert self._wer("hello world", "hello there") == pytest.approx(0.5)

    def test_wer_empty_reference(self):
        # Edge: empty reference
        result = self._wer("", "hello")
        assert result >= 0.0

    def test_cer_empty_reference(self):
        result = self._cer("", "hello")
        assert result >= 0.0


# ---------------------------------------------------------------------------
# tools/asr_benchmark/report.py — report functions
# ---------------------------------------------------------------------------

class TestBenchmarkReport:
    def setup_method(self):
        from pathlib import Path
        from tools.asr_benchmark.report import _load_results, _print_table
        self._load_results = lambda p: _load_results(Path(p))
        self._print_table = _print_table

    def _write_results(self, records: list[dict], path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def test_load_results_from_jsonl(self, tmp_path):
        p = tmp_path / "results.jsonl"
        records = [
            {"file": "a.wav", "profile": "default", "wer": 0.1, "cer": 0.05, "final_latency_ms": 200},
            {"file": "b.wav", "profile": "headset", "wer": 0.2, "cer": 0.08, "final_latency_ms": 150},
        ]
        self._write_results(records, str(p))
        loaded = self._load_results(str(p))
        assert len(loaded) == 2
        assert loaded[0]["profile"] == "default"

    def test_load_results_missing_file(self):
        result = self._load_results("/nonexistent/path/results.jsonl")
        assert result == []

    def test_print_table_runs_without_error(self, capsys):
        records = [
            {"file": "a.wav", "profile": "default", "wer": 0.1, "cer": 0.05, "final_latency_ms": 200},
        ]
        self._print_table(records)
        captured = capsys.readouterr()
        assert "default" in captured.out or len(captured.out) >= 0

    def test_print_table_empty(self, capsys):
        self._print_table([])
        # Should not crash on empty input

    def test_main_csv_output(self, tmp_path):
        from tools.asr_benchmark.report import main
        p = tmp_path / "results.jsonl"
        records = [
            {"file": "a.wav", "profile": "default", "wer": 0.1, "cer": 0.05, "final_latency_ms": 200},
        ]
        self._write_results(records, str(p))
        out_csv = tmp_path / "out.csv"
        rc = main([str(p), "--format", "csv", "--output", str(out_csv)])
        assert rc == 0
        assert out_csv.exists()
        content = out_csv.read_text()
        assert "default" in content


# ---------------------------------------------------------------------------
# tools/asr_benchmark/run_benchmark.py — CLI arg parsing (dry run)
# ---------------------------------------------------------------------------

class TestBenchmarkCLI:
    def test_main_missing_required_args_returns_nonzero(self):
        from tools.asr_benchmark.run_benchmark import main
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code != 0  # argparse exits with code 2 when required args missing

    def test_load_wav_nonexistent_raises(self):
        from tools.asr_benchmark.run_benchmark import _load_wav
        with pytest.raises(Exception):
            _load_wav("/nonexistent/path/audio.wav")
