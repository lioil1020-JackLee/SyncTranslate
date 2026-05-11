"""Tests for local benchmark ASR regression corpus helpers."""
from __future__ import annotations

from pathlib import Path

from tools.asr_benchmark.prepare_local_benchmark_corpus import (
    SubtitleCue,
    _group_cues,
    _thresholds_for,
    parse_subtitles,
)


def test_parse_subtitles_supports_vtt_and_tags(tmp_path: Path):
    subtitle = tmp_path / "sample.vtt"
    subtitle.write_text(
        "\ufeffWEBVTT\n\n"
        "00:00:01.000 --> 00:00:02.500\n"
        "<v Speaker>你好&nbsp;世界</v>\n\n"
        "00:00:03.000 --> 00:00:04.000\n"
        "第二句\n",
        encoding="utf-8",
    )

    cues = parse_subtitles(subtitle)

    assert len(cues) == 2
    assert cues[0].start_sec == 1.0
    assert cues[0].end_sec == 2.5
    assert cues[0].text == "你好 世界"
    assert cues[1].text == "第二句"


def test_group_cues_respects_duration_and_target_count():
    cues = [
        SubtitleCue(start_sec=0.0, end_sec=10.0, text="a"),
        SubtitleCue(start_sec=10.0, end_sec=20.0, text="b"),
        SubtitleCue(start_sec=20.0, end_sec=31.0, text="c"),
        SubtitleCue(start_sec=31.0, end_sec=50.0, text="d"),
    ]

    groups = _group_cues(cues, target_count=2, target_sec=25.0, max_sec=35.0)

    assert len(groups) == 2
    assert [cue.text for cue in groups[0]] == ["a", "b"]
    assert [cue.text for cue in groups[1]] == ["c", "d"]


def test_local_thresholds_are_calibrated_for_subtitle_aligned_references():
    zh_novel = _thresholds_for("zh-TW", "novel")
    en_news = _thresholds_for("en", "news")
    mixed = _thresholds_for("zh-CN", "mixed_dialogue")

    assert zh_novel["min_accuracy"]["novel_turbo"] == 0.73
    assert en_news["min_accuracy"]["news_turbo"] == 0.80
    assert mixed["max_missing_sentence_rate"] == 0.25
