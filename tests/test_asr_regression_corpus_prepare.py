"""Tests for local ASR regression corpus preparation helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

yaml = pytest.importorskip("yaml", reason="PyYAML not installed")

from tools.asr_benchmark.prepare_regression_corpus import (
    CorpusItem,
    _count_by_language,
    _thresholds_for,
    write_corpus,
)
from tools.asr_benchmark.run_regression_corpus import (
    _model_override_for_cli,
    load_manifest,
    parse_samples,
    validate_sample_paths,
)


def _item(item_id: str, language: str, category: str) -> CorpusItem:
    return CorpusItem(
        id=item_id,
        language=language,
        category=category,
        text="測試文字" if language.startswith("zh") else "test text",
        audio=np.zeros(1600, dtype=np.float32),
        sample_rate=16000,
        source_dataset="unit/test",
        source_config="default",
        source_split="test",
        description="unit test sample",
        speaker_count=1,
    )


def test_count_by_language_treats_mixed_category_separately():
    items = [
        _item("zh_1", "zh-CN", "news"),
        _item("en_1", "en", "meeting"),
        _item("mix_1", "zh-CN", "mixed_dialogue"),
    ]
    assert _count_by_language(items) == {"zh": 1, "en": 1, "mixed": 1}


def test_thresholds_are_more_lenient_for_dialogue_and_mixed_dialogue():
    news = _thresholds_for(_item("zh_1", "zh-CN", "news"))
    dialogue = _thresholds_for(_item("zh_2", "zh-CN", "dialogue"))
    mixed = _thresholds_for(_item("mix_1", "zh-CN", "mixed_dialogue"))

    assert dialogue["min_accuracy"]["dialogue_turbo"] < news["min_accuracy"]["news_turbo"]
    assert mixed["max_missing_sentence_rate"] >= dialogue["max_missing_sentence_rate"]


def test_write_corpus_outputs_manifest_refs_audio_and_metadata(tmp_path: Path):
    items = [
        _item("zh_1", "zh-CN", "news"),
        _item("en_1", "en", "meeting"),
        _item("mix_1", "zh-CN", "mixed_dialogue"),
    ]
    write_corpus(items, tmp_path, force=True)

    manifest = tmp_path / "manifest.yaml"
    samples = parse_samples(load_manifest(manifest))
    assert len(samples) == 3
    assert validate_sample_paths(samples, tmp_path) == []
    assert (tmp_path / "metadata" / "mix_1.yaml").is_file()
    assert (tmp_path / "sources.json").is_file()


def test_write_corpus_refuses_to_overwrite_without_force(tmp_path: Path):
    items = [_item("zh_1", "zh-CN", "news")]
    write_corpus(items, tmp_path, force=True)
    with pytest.raises(SystemExit):
        write_corpus(items, tmp_path, force=False)


def test_regression_model_aliases_map_to_runtime_models():
    assert _model_override_for_cli("turbo") == "large-v3-turbo"
    assert _model_override_for_cli("belle").endswith("belle-zh-ct2")
    assert _model_override_for_cli("default") == ""
