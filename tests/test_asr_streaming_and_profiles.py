"""Tests for StreamingPolicy and EndpointProfiles (Phase 2)."""
from __future__ import annotations

import pytest

from app.infra.asr.streaming_policy import (
    DEGRADATION_CONGESTED,
    DEGRADATION_DEGRADED,
    DEGRADATION_NORMAL,
    StreamingContext,
    StreamingDecision,
    StreamingPolicy,
)
from app.infra.asr.endpoint_profiles import (
    PROFILES,
    EndpointProfile,
    get_endpoint_profile,
    list_profiles,
)
from app.infra.asr.worker_v2 import _scaled_finalize_thresholds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(
    speech_active: bool = True,
    speech_started: bool = False,
    speech_ended: bool = False,
    soft_endpoint: bool = False,
    hard_endpoint: bool = False,
    pause_ms: float = 0.0,
):
    from app.infra.asr.endpointing_v2 import EndpointSignal
    return EndpointSignal(
        speech_active=speech_active,
        speech_started=speech_started,
        speech_ended=speech_ended,
        soft_endpoint=soft_endpoint,
        hard_endpoint=hard_endpoint,
        pause_ms=pause_ms,
    )


def _make_ctx(**overrides) -> StreamingContext:
    defaults = dict(
        signal=_make_signal(speech_active=True),
        segment_audio_ms=500,
        now_ms=10000,
        last_partial_emit_ms=9000,
        backlog=0,
        drop_partial_until_final=False,
        partial_cooldown_until_ms=0,
        dropped_chunks_total=0,
        partial_interval_ms=500,
        min_partial_audio_ms=300,
        soft_endpoint_finalize_audio_ms=1200,
        speech_end_finalize_audio_ms=1000,
        adaptive_length_limit_ms=4000,
        adaptive_length_ceiling_ms=12000,
        force_final_queue_size=8,
        force_final_audio_ms=1800,
    )
    defaults.update(overrides)
    return StreamingContext(**defaults)


# ---------------------------------------------------------------------------
# StreamingPolicy — partial emission
# ---------------------------------------------------------------------------

class TestStreamingPolicyPartial:
    def test_emits_partial_when_conditions_met(self):
        policy = StreamingPolicy()
        ctx = _make_ctx()
        decision = policy.decide(ctx)
        assert decision.emit_partial is True
        assert decision.degradation_level == DEGRADATION_NORMAL

    def test_no_partial_when_interval_not_elapsed(self):
        policy = StreamingPolicy()
        ctx = _make_ctx(last_partial_emit_ms=9800)  # only 200ms ago
        decision = policy.decide(ctx)
        assert decision.emit_partial is False

    def test_no_partial_when_drop_flag_set(self):
        policy = StreamingPolicy()
        ctx = _make_ctx(drop_partial_until_final=True)
        decision = policy.decide(ctx)
        assert decision.emit_partial is False

    def test_no_partial_when_cooldown_active(self):
        policy = StreamingPolicy()
        ctx = _make_ctx(partial_cooldown_until_ms=20000)  # far future
        decision = policy.decide(ctx)
        assert decision.emit_partial is False

    def test_no_partial_when_speech_not_active(self):
        policy = StreamingPolicy()
        ctx = _make_ctx(signal=_make_signal(speech_active=False))
        decision = policy.decide(ctx)
        assert decision.emit_partial is False

    def test_no_partial_when_audio_too_short(self):
        policy = StreamingPolicy()
        ctx = _make_ctx(segment_audio_ms=100, min_partial_audio_ms=300)
        decision = policy.decide(ctx)
        assert decision.emit_partial is False

    def test_no_partial_when_backlog_high(self):
        policy = StreamingPolicy()
        ctx = _make_ctx(backlog=3)  # > 1
        decision = policy.decide(ctx)
        assert decision.emit_partial is False


# ---------------------------------------------------------------------------
# StreamingPolicy — finalization
# ---------------------------------------------------------------------------

class TestStreamingPolicyFinalization:
    def test_hard_endpoint_triggers_final(self):
        policy = StreamingPolicy()
        ctx = _make_ctx(signal=_make_signal(hard_endpoint=True, speech_active=False))
        decision = policy.decide(ctx)
        assert decision.emit_final is True
        assert decision.is_early_final is False
        assert decision.reason == "hard_endpoint"

    def test_soft_endpoint_triggers_final_when_long_enough(self):
        policy = StreamingPolicy()
        ctx = _make_ctx(
            signal=_make_signal(soft_endpoint=True, speech_active=False),
            segment_audio_ms=1500,
        )
        decision = policy.decide(ctx)
        assert decision.emit_final is True
        assert decision.is_early_final is True
        assert decision.reset_endpointing_after_final is True

    def test_soft_endpoint_no_final_when_too_short(self):
        policy = StreamingPolicy()
        ctx = _make_ctx(
            signal=_make_signal(soft_endpoint=True, speech_active=False),
            segment_audio_ms=400,
        )
        decision = policy.decide(ctx)
        assert decision.emit_final is False

    def test_speech_end_triggers_final(self):
        policy = StreamingPolicy()
        ctx = _make_ctx(
            signal=_make_signal(speech_ended=True, speech_active=False),
            segment_audio_ms=1200,
        )
        decision = policy.decide(ctx)
        assert decision.emit_final is True
        assert decision.is_early_final is True

    def test_ceiling_triggers_final(self):
        policy = StreamingPolicy()
        ctx = _make_ctx(segment_audio_ms=13000)
        decision = policy.decide(ctx)
        assert decision.emit_final is True
        assert decision.reason == "ceiling"

    def test_force_final_on_queue_pressure(self):
        policy = StreamingPolicy()
        ctx = _make_ctx(
            backlog=10,
            force_final_queue_size=8,
            segment_audio_ms=2000,
            force_final_audio_ms=1800,
        )
        decision = policy.decide(ctx)
        assert decision.emit_final is True
        assert "force_final" in decision.reason

    def test_adaptive_length_triggers_final(self):
        policy = StreamingPolicy()
        ctx = _make_ctx(
            signal=_make_signal(speech_active=True, pause_ms=250.0),
            segment_audio_ms=4500,
            adaptive_length_limit_ms=4000,
        )
        decision = policy.decide(ctx)
        assert decision.emit_final is True


# ---------------------------------------------------------------------------
# StreamingPolicy — degradation levels
# ---------------------------------------------------------------------------

class TestStreamingPolicyDegradation:
    def test_normal_level(self):
        policy = StreamingPolicy()
        ctx = _make_ctx(backlog=0, dropped_chunks_total=0)
        decision = policy.decide(ctx)
        assert decision.degradation_level == DEGRADATION_NORMAL

    def test_congested_level(self):
        policy = StreamingPolicy()
        ctx = _make_ctx(backlog=4, force_final_queue_size=8, dropped_chunks_total=0)
        decision = policy.decide(ctx)
        assert decision.degradation_level == DEGRADATION_CONGESTED

    def test_degraded_level_by_queue(self):
        policy = StreamingPolicy()
        ctx = _make_ctx(backlog=7, force_final_queue_size=8, dropped_chunks_total=0)
        decision = policy.decide(ctx)
        assert decision.degradation_level == DEGRADATION_DEGRADED

    def test_degraded_level_by_drops(self):
        policy = StreamingPolicy()
        ctx = _make_ctx(dropped_chunks_total=25)
        decision = policy.decide(ctx)
        assert decision.degradation_level == DEGRADATION_DEGRADED

    def test_degraded_suppresses_partial(self):
        policy = StreamingPolicy()
        ctx = _make_ctx(backlog=7, force_final_queue_size=8)
        decision = policy.decide(ctx)
        assert decision.suppress_partial is True
        assert decision.emit_partial is False

    def test_congested_uses_longer_interval(self):
        """In congested mode, interval is multiplied so partial may be suppressed."""
        policy = StreamingPolicy(congested_partial_interval_multiplier=2.0)
        # last emitted 600ms ago, interval=500ms → normally would emit
        # but with 2x multiplier effective interval = 1000ms → should NOT emit
        ctx = _make_ctx(
            backlog=4,
            force_final_queue_size=8,
            last_partial_emit_ms=9400,  # 600ms ago
        )
        decision = policy.decide(ctx)
        assert decision.degradation_level == DEGRADATION_CONGESTED
        assert decision.emit_partial is False

    def test_degradation_disabled(self):
        policy = StreamingPolicy(degradation_enabled=False)
        ctx = _make_ctx(backlog=7, force_final_queue_size=8)
        decision = policy.decide(ctx)
        assert decision.degradation_level == DEGRADATION_NORMAL


# ---------------------------------------------------------------------------
# EndpointProfiles
# ---------------------------------------------------------------------------

class TestEndpointProfiles:
    def test_all_builtin_profiles_exist(self):
        for name in ["default", "meeting_room", "headset", "noisy_environment", "max_accuracy", "low_latency", "turn_taking"]:
            p = get_endpoint_profile(name)
            assert p.name == name

    def test_fallback_to_default_on_unknown(self):
        p = get_endpoint_profile("nonexistent_profile")
        assert p.name == "default"

    def test_fallback_on_none(self):
        p = get_endpoint_profile(None)
        assert p.name == "default"

    def test_list_profiles_returns_all(self):
        names = list_profiles()
        assert "default" in names
        assert "low_latency" in names
        assert len(names) == len(PROFILES)

    def test_to_worker_kwargs_keys(self):
        p = get_endpoint_profile("headset")
        kwargs = p.to_worker_kwargs()
        assert "partial_interval_ms" in kwargs
        assert "min_partial_audio_ms" in kwargs
        assert "soft_final_audio_ms" in kwargs
        assert "pre_roll_ms" in kwargs

    def test_low_latency_shorter_than_max_accuracy(self):
        low = get_endpoint_profile("low_latency")
        acc = get_endpoint_profile("max_accuracy")
        assert low.partial_interval_ms < acc.partial_interval_ms
        assert low.soft_final_audio_ms < acc.soft_final_audio_ms

    def test_turn_taking_is_more_aggressive_than_default(self):
        turn = get_endpoint_profile("turn_taking")
        default = get_endpoint_profile("default")
        assert turn.partial_interval_ms <= default.partial_interval_ms
        assert turn.soft_final_audio_ms < default.soft_final_audio_ms

    def test_noisy_longer_thresholds(self):
        noisy = get_endpoint_profile("noisy_environment")
        default = get_endpoint_profile("default")
        assert noisy.soft_final_audio_ms > default.soft_final_audio_ms


class TestWorkerFinalizeThresholds:
    def test_thresholds_scale_with_soft_final_window(self):
        soft_ms, speech_end_ms = _scaled_finalize_thresholds(
            soft_final_audio_ms=6000,
            min_partial_audio_ms=1000,
            force_final_audio_ms=1800,
        )
        assert soft_ms == 3900
        assert speech_end_ms == 3300

    def test_thresholds_respect_force_final_floor(self):
        soft_ms, speech_end_ms = _scaled_finalize_thresholds(
            soft_final_audio_ms=2000,
            min_partial_audio_ms=300,
            force_final_audio_ms=1800,
        )
        assert soft_ms == 1800
        assert speech_end_ms == 1100
        assert speech_end_ms <= soft_ms
