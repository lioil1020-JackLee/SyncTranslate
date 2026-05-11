from __future__ import annotations

import numpy as np

from app.infra.asr.backend_v2 import BackendTranscript
from app.infra.asr.confidence_gate import (
    FinalRescuePolicy,
    choose_better_candidate,
    confidence_failure_reasons,
)
from app.infra.asr.model_router import alternate_chinese_model, build_chinese_fallback_profile
from app.infra.asr.worker_v2 import SourceRuntimeV2
from app.infra.config.schema import AppConfig


class _RescueBackend:
    def __init__(self, text: str, *, avg_logprob: float = -0.2) -> None:
        self.text = text
        self.avg_logprob = avg_logprob

    def transcribe_final_rescue(self, audio, sample_rate, *, frontend_stats=None):
        return BackendTranscript(
            text=self.text,
            is_final=True,
            avg_logprob=self.avg_logprob,
            max_no_speech_prob=0.05,
            max_compression_ratio=1.1,
            rescue_used=True,
        )


class _Endpointing:
    def snapshot(self):
        return {}

    def reset(self):
        return None


def test_confidence_gate_flags_low_quality_final() -> None:
    result = BackendTranscript(
        text="嗯",
        is_final=True,
        avg_logprob=-1.4,
        max_no_speech_prob=0.72,
        max_compression_ratio=2.7,
    )

    reasons = confidence_failure_reasons(
        result,
        result.text,
        audio_ms=1800,
        policy=FinalRescuePolicy(enabled=True),
    )

    assert "low_logprob" in reasons
    assert "high_no_speech" in reasons
    assert "high_compression" in reasons
    assert "too_short_for_audio" in reasons


def test_confidence_gate_respects_disabled_policy() -> None:
    result = BackendTranscript(text="ok", is_final=True, avg_logprob=-2.0)

    reasons = confidence_failure_reasons(
        result,
        result.text,
        audio_ms=1500,
        policy=FinalRescuePolicy(enabled=False),
    )

    assert reasons == []


def test_confidence_gate_flags_empty_text_with_audio() -> None:
    result = BackendTranscript(text="", is_final=True, avg_logprob=None)

    reasons = confidence_failure_reasons(
        result,
        result.text,
        audio_ms=1200,
        policy=FinalRescuePolicy(enabled=True),
    )

    assert "empty_text" in reasons


def test_candidate_selector_prefers_better_rescue_text() -> None:
    current = BackendTranscript(
        text="嗯",
        is_final=True,
        avg_logprob=-1.6,
        max_no_speech_prob=0.7,
        max_compression_ratio=2.5,
    )
    rescued = BackendTranscript(
        text="我們現在開始測試",
        is_final=True,
        avg_logprob=-0.35,
        max_no_speech_prob=0.08,
        max_compression_ratio=1.1,
    )

    assert choose_better_candidate(current, rescued) is rescued


def test_candidate_selector_keeps_longer_current_when_rescue_truncates() -> None:
    current = BackendTranscript(
        text="這是一段已經相當完整的辨識內容",
        is_final=True,
        avg_logprob=-0.8,
        max_no_speech_prob=0.08,
        max_compression_ratio=1.1,
    )
    rescued = BackendTranscript(
        text="完整內容",
        is_final=True,
        avg_logprob=-0.7,
        max_no_speech_prob=0.05,
        max_compression_ratio=1.0,
    )

    assert choose_better_candidate(current, rescued) is current


def test_chinese_model_router_uses_alternate_model() -> None:
    assert alternate_chinese_model("large-v3-turbo").endswith("belle-zh-ct2")
    assert alternate_chinese_model(r".\runtimes\models\belle-zh-ct2") == "large-v3-turbo"


def test_build_chinese_fallback_profile_disabled_for_low_latency() -> None:
    cfg = AppConfig()
    cfg.runtime.asr_accuracy_mode = "low_latency"
    base = cfg.asr_channels.local
    base.model = "large-v3-turbo"

    assert build_chinese_fallback_profile(cfg, base, language="zh-TW") is None


def test_build_chinese_fallback_profile_for_zh() -> None:
    cfg = AppConfig()
    cfg.runtime.asr_accuracy_mode = "balanced"
    base = cfg.asr_channels.local
    base.model = "large-v3-turbo"

    fallback = build_chinese_fallback_profile(cfg, base, language="zh-TW")

    assert fallback is not None
    assert fallback.model != base.model
    assert fallback.final_beam_size >= 5


def test_worker_rescue_selects_better_candidate() -> None:
    runtime = SourceRuntimeV2(
        source="local",
        partial_backend=object(),
        final_backend=_RescueBackend("我們現在開始測試", avg_logprob=-0.2),
        endpointing=_Endpointing(),
        partial_interval_ms=500,
        partial_history_seconds=2,
        final_history_seconds=10,
        soft_final_audio_ms=3000,
        pre_roll_ms=300,
        min_partial_audio_ms=300,
        queue_maxsize=16,
        final_rescue_policy=FinalRescuePolicy(enabled=True, max_attempts=1),
    )
    original = BackendTranscript(
        text="嗯",
        is_final=True,
        avg_logprob=-1.5,
        max_no_speech_prob=0.8,
        max_compression_ratio=2.8,
    )

    selected = runtime._maybe_rescue_final(
        result=original,
        audio=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        audio_ms=1000,
        frontend_stats={},
    )

    assert selected.text == "我們現在開始測試"
    assert runtime.stats().final_rescue_count == 1
    assert "rescue" in runtime.stats().last_final_rescue_reason
