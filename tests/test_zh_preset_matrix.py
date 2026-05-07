from __future__ import annotations

from tools.asr_benchmark.run_zh_preset_matrix import (
    MODE_MODEL_PARAMS,
    SAMPLES,
    _selected_samples,
    _selected_values,
)


def test_zh_preset_matrix_defines_ui_mode_model_combinations() -> None:
    assert set(MODE_MODEL_PARAMS) == {
        ("meeting", "belle"),
        ("meeting", "turbo"),
        ("dialogue", "belle"),
        ("dialogue", "turbo"),
    }
    assert MODE_MODEL_PARAMS[("meeting", "belle")]["vad"]["min_silence_duration_ms"] == 600
    assert MODE_MODEL_PARAMS[("dialogue", "belle")]["vad"]["min_silence_duration_ms"] == 320
    assert MODE_MODEL_PARAMS[("dialogue", "turbo")]["streaming"]["soft_final_audio_ms"] == 1800
    assert MODE_MODEL_PARAMS[("meeting", "turbo")]["streaming"]["partial_interval_ms"] == 520
    assert MODE_MODEL_PARAMS[("dialogue", "turbo")]["streaming"]["partial_interval_ms"] == 240


def test_zh_preset_matrix_sample_selection_is_explicit() -> None:
    selected = _selected_samples("zh_kaidan5,zh_fruit_cow")

    assert [sample.label for sample in selected] == ["zh_kaidan5", "zh_fruit_cow"]
    assert len(SAMPLES) >= 5


def test_zh_preset_matrix_rejects_unknown_values() -> None:
    try:
        _selected_values("meeting,unknown", allowed={"meeting", "dialogue"}, default=("meeting",))
    except ValueError as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("expected ValueError")
