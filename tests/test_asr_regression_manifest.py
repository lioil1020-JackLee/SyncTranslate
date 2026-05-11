"""Tests for regression manifest loading and path validation."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml", reason="PyYAML not installed")

from tools.asr_benchmark.run_regression_corpus import (
    SampleConfig,
    load_manifest,
    parse_samples,
    validate_sample_paths,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_manifest(content: str, tmp_path: Path) -> Path:
    p = tmp_path / "manifest.yaml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


MINIMAL_MANIFEST = """\
    schema_version: 1
    samples:
      - id: test_01
        language: zh-TW
        category: news
        audio: audio/test.wav
        reference: refs/test.txt
        min_accuracy:
          news_turbo: 0.85
        max_dropped_chunks: 0
        max_repetition_ratio: 0.02
    """


# ---------------------------------------------------------------------------
# load_manifest
# ---------------------------------------------------------------------------

class TestLoadManifest:
    def test_loads_valid_yaml(self, tmp_path):
        p = _write_manifest(MINIMAL_MANIFEST, tmp_path)
        data = load_manifest(p)
        assert isinstance(data, dict)
        assert "samples" in data

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_manifest(tmp_path / "nonexistent.yaml")

    def test_yaml_list_raises_value_error(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("- just a list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mapping"):
            load_manifest(p)

    def test_accepts_path_object(self, tmp_path):
        p = _write_manifest(MINIMAL_MANIFEST, tmp_path)
        data = load_manifest(Path(p))
        assert "samples" in data

    def test_accepts_string_path(self, tmp_path):
        p = _write_manifest(MINIMAL_MANIFEST, tmp_path)
        data = load_manifest(str(p))
        assert "samples" in data


# ---------------------------------------------------------------------------
# parse_samples
# ---------------------------------------------------------------------------

class TestParseSamples:
    def test_parses_minimal_manifest(self, tmp_path):
        p = _write_manifest(MINIMAL_MANIFEST, tmp_path)
        data = load_manifest(p)
        samples = parse_samples(data)
        assert len(samples) == 1
        s = samples[0]
        assert s.id == "test_01"
        assert s.language == "zh-TW"
        assert s.category == "news"
        assert s.audio == "audio/test.wav"
        assert s.reference == "refs/test.txt"
        assert s.min_accuracy == {"news_turbo": 0.85}
        assert s.max_dropped_chunks == 0
        assert s.max_repetition_ratio == pytest.approx(0.02)
        assert s.max_missing_sentence_rate == pytest.approx(0.20)
        assert s.max_duplicate_rate == pytest.approx(0.05)
        assert s.max_hallucination_rate == pytest.approx(0.02)
        assert s.max_average_final_latency_ms == pytest.approx(0.0)

    def test_missing_samples_key_raises(self):
        with pytest.raises(ValueError, match="samples"):
            parse_samples({})

    def test_samples_not_list_raises(self):
        with pytest.raises(ValueError, match="samples"):
            parse_samples({"samples": "not a list"})

    def test_missing_id_raises(self, tmp_path):
        bad = """\
            schema_version: 1
            samples:
              - language: zh-TW
                audio: audio/test.wav
                reference: refs/test.txt
            """
        p = _write_manifest(bad, tmp_path)
        data = load_manifest(p)
        with pytest.raises(ValueError, match="'id'"):
            parse_samples(data)

    def test_missing_language_raises(self, tmp_path):
        bad = """\
            schema_version: 1
            samples:
              - id: s1
                audio: audio/test.wav
                reference: refs/test.txt
            """
        p = _write_manifest(bad, tmp_path)
        data = load_manifest(p)
        with pytest.raises(ValueError, match="'language'"):
            parse_samples(data)

    def test_missing_audio_raises(self, tmp_path):
        bad = """\
            schema_version: 1
            samples:
              - id: s1
                language: zh-TW
                reference: refs/test.txt
            """
        p = _write_manifest(bad, tmp_path)
        data = load_manifest(p)
        with pytest.raises(ValueError, match="'audio'"):
            parse_samples(data)

    def test_missing_reference_raises(self, tmp_path):
        bad = """\
            schema_version: 1
            samples:
              - id: s1
                language: zh-TW
                audio: audio/test.wav
            """
        p = _write_manifest(bad, tmp_path)
        data = load_manifest(p)
        with pytest.raises(ValueError, match="'reference'"):
            parse_samples(data)

    def test_defaults_applied(self, tmp_path):
        minimal = """\
            schema_version: 1
            samples:
              - id: s1
                language: en
                audio: a.wav
                reference: r.txt
            """
        p = _write_manifest(minimal, tmp_path)
        samples = parse_samples(load_manifest(p))
        s = samples[0]
        assert s.category == "unknown"
        assert s.description == ""
        assert s.min_accuracy == {}
        assert s.max_dropped_chunks == 0
        assert s.max_repetition_ratio == pytest.approx(0.05)
        assert s.max_missing_sentence_rate == pytest.approx(0.20)
        assert s.max_duplicate_rate == pytest.approx(0.05)
        assert s.max_hallucination_rate == pytest.approx(0.02)

    def test_multiple_samples(self, tmp_path):
        multi = """\
            schema_version: 1
            samples:
              - id: test_01
                language: zh-TW
                category: news
                audio: audio/test.wav
                reference: refs/test.txt
                min_accuracy:
                  news_turbo: 0.85
                max_dropped_chunks: 0
                max_repetition_ratio: 0.02
              - id: test_02
                language: en
                category: meeting
                audio: audio/en.wav
                reference: refs/en.txt
            """
        p = _write_manifest(multi, tmp_path)
        samples = parse_samples(load_manifest(p))
        assert len(samples) == 2
        assert samples[1].id == "test_02"
        assert samples[1].language == "en"

    def test_real_manifest_parses(self):
        real = Path(__file__).parent.parent / "downloads" / "asr_regression" / "manifest.yaml"
        if not real.is_file():
            pytest.skip("real manifest not present")
        data = load_manifest(real)
        samples = parse_samples(data)
        assert len(samples) >= 1
        for s in samples:
            assert s.id
            assert s.language
            assert s.audio
            assert s.reference

    def test_example_manifest_parses_with_metadata(self):
        example = Path(__file__).parent.parent / "tools" / "asr_benchmark" / "asr_regression_manifest.example.yaml"
        data = load_manifest(example)
        samples = parse_samples(data)
        assert len(samples) >= 3
        first = samples[0]
        assert first.duration_sec > 0
        assert first.speaker_count >= 1
        assert first.reference_quality == "human_verified"
        assert first.noise_level in {"low", "medium", "high", "unknown"}
        assert first.max_missing_sentence_rate > 0
        assert first.max_average_final_latency_ms > 0


# ---------------------------------------------------------------------------
# validate_sample_paths
# ---------------------------------------------------------------------------

class TestValidateSamplePaths:
    def _make_sample(self, tmp_path: Path, create_audio=True, create_ref=True):
        audio = tmp_path / "audio" / "test.wav"
        ref = tmp_path / "refs" / "test.txt"
        audio.parent.mkdir(exist_ok=True)
        ref.parent.mkdir(exist_ok=True)
        if create_audio:
            audio.write_bytes(b"RIFF")
        if create_ref:
            ref.write_text("reference text", encoding="utf-8")
        return SampleConfig(
            id="s1", language="zh-TW", category="news",
            audio="audio/test.wav", reference="refs/test.txt",
        )

    def test_both_files_exist_no_errors(self, tmp_path):
        sample = self._make_sample(tmp_path)
        errors = validate_sample_paths([sample], tmp_path)
        assert errors == []

    def test_missing_audio_reports_error(self, tmp_path):
        sample = self._make_sample(tmp_path, create_audio=False)
        errors = validate_sample_paths([sample], tmp_path)
        assert len(errors) == 1
        assert "audio" in errors[0].lower() or "s1" in errors[0]

    def test_missing_reference_reports_error(self, tmp_path):
        sample = self._make_sample(tmp_path, create_ref=False)
        errors = validate_sample_paths([sample], tmp_path)
        assert len(errors) == 1
        assert "reference" in errors[0].lower() or "s1" in errors[0]

    def test_both_missing_two_errors(self, tmp_path):
        sample = self._make_sample(tmp_path, create_audio=False, create_ref=False)
        errors = validate_sample_paths([sample], tmp_path)
        assert len(errors) == 2

    def test_multiple_samples_accumulate_errors(self, tmp_path):
        s1 = SampleConfig(id="s1", language="zh-TW", category="n", audio="a1.wav", reference="r1.txt")
        s2 = SampleConfig(id="s2", language="en",    category="n", audio="a2.wav", reference="r2.txt")
        errors = validate_sample_paths([s1, s2], tmp_path)
        assert len(errors) == 4   # 2 per missing sample
