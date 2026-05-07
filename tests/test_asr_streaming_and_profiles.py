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
from app.infra.asr.worker_v2 import _pressure_force_final_audio_ms
from app.infra.asr.worker_v2 import _drain_limit_for_backlog
from app.infra.asr.worker_v2 import SourceRuntimeV2


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
    is_speech_frame: bool = False,
):
    from app.infra.asr.endpointing_v2 import EndpointSignal
    return EndpointSignal(
        speech_active=speech_active,
        speech_started=speech_started,
        speech_ended=speech_ended,
        soft_endpoint=soft_endpoint,
        hard_endpoint=hard_endpoint,
        pause_ms=pause_ms,
        is_speech_frame=is_speech_frame,
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
        assert decision.is_early_final is False

    def test_pause_turn_triggers_final_before_hard_endpoint(self):
        policy = StreamingPolicy()
        ctx = _make_ctx(
            signal=_make_signal(speech_active=True, pause_ms=260.0),
            segment_audio_ms=1100,
            min_partial_audio_ms=300,
        )
        decision = policy.decide(ctx)
        assert decision.emit_final is True
        assert decision.reason == "pause_turn"
        assert decision.reset_endpointing_after_final is True

    def test_ceiling_triggers_final(self):
        policy = StreamingPolicy()
        ctx = _make_ctx(segment_audio_ms=13000)
        decision = policy.decide(ctx)
        assert decision.emit_final is True
        assert decision.is_early_final is False
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
        assert decision.is_early_final is False
        assert "force_final" in decision.reason

    def test_worker_uses_conservative_pressure_final_threshold(self):
        assert _pressure_force_final_audio_ms(1800) == 3200
        assert _pressure_force_final_audio_ms(4600) == 4600
        assert _pressure_force_final_audio_ms(9000) == 6000

    def test_worker_coalesces_more_chunks_under_queue_pressure(self):
        assert _drain_limit_for_backlog(0, 256) == 3
        assert _drain_limit_for_backlog(60, 256) == 5
        assert _drain_limit_for_backlog(130, 256) == 9
        assert _drain_limit_for_backlog(220, 256) == 15

    def test_adaptive_length_triggers_final(self):
        policy = StreamingPolicy()
        ctx = _make_ctx(
            signal=_make_signal(speech_active=True, pause_ms=190.0),
            segment_audio_ms=4500,
            adaptive_length_limit_ms=4000,
        )
        decision = policy.decide(ctx)
        assert decision.emit_final is True
        assert decision.is_early_final is False

    def test_soft_endpoint_final_does_not_restart_runtime_segment(self):
        decision = StreamingDecision(
            emit_final=True,
            is_early_final=True,
            reset_endpointing_after_final=True,
            reason="soft_endpoint",
        )
        signal = _make_signal(
            speech_active=True,
            soft_endpoint=True,
            pause_ms=260.0,
            is_speech_frame=False,
        )

        assert SourceRuntimeV2._should_restart_after_final(decision=decision, signal=signal) is False

    def test_non_boundary_final_restarts_when_endpointing_still_active(self):
        decision = StreamingDecision(emit_final=True, reason="ceiling")

        assert SourceRuntimeV2._should_restart_after_final(
            decision=decision,
            signal=_make_signal(speech_active=True, is_speech_frame=True),
        ) is True
        assert SourceRuntimeV2._should_restart_after_final(
            decision=decision,
            signal=_make_signal(speech_active=True, is_speech_frame=False),
        ) is True

    def test_adaptive_length_final_restarts_even_on_pause_frame(self):
        decision = StreamingDecision(emit_final=True, reason="adaptive_length")

        assert SourceRuntimeV2._should_restart_after_final(
            decision=decision,
            signal=_make_signal(speech_active=True, pause_ms=190.0, is_speech_frame=False),
        ) is True
        assert SourceRuntimeV2._should_restart_after_final(
            decision=decision,
            signal=_make_signal(speech_active=False, pause_ms=190.0, is_speech_frame=False),
        ) is False


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

    def test_congested_suppresses_partial_at_high_backlog(self):
        policy = StreamingPolicy()
        ctx = _make_ctx(backlog=5, force_final_queue_size=8)
        decision = policy.decide(ctx)
        assert decision.degradation_level == DEGRADATION_CONGESTED
        assert decision.suppress_partial is True
        assert decision.reason == "congested:suppress_partial"

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

    def test_meeting_room_is_more_conservative_than_default(self):
        meeting = get_endpoint_profile("meeting_room")
        default = get_endpoint_profile("default")
        assert meeting.partial_interval_ms > default.partial_interval_ms
        assert meeting.min_partial_audio_ms > default.min_partial_audio_ms
        assert meeting.soft_final_audio_ms > default.soft_final_audio_ms
        assert meeting.speech_end_finalize_audio_ms > default.speech_end_finalize_audio_ms

    def test_turn_taking_stays_faster_than_meeting_room(self):
        turn = get_endpoint_profile("turn_taking")
        meeting = get_endpoint_profile("meeting_room")
        assert turn.partial_interval_ms < meeting.partial_interval_ms
        assert turn.min_partial_audio_ms < meeting.min_partial_audio_ms
        assert turn.soft_final_audio_ms < meeting.soft_final_audio_ms


# ---------------------------------------------------------------------------
# StreamingPolicy — final priority
# ---------------------------------------------------------------------------

class TestStreamingPolicyFinalPriority:
    def test_queue_ratio_triggers_final_priority(self):
        policy = StreamingPolicy(final_priority_queue_ratio=0.45)
        ctx = _make_ctx(backlog=46, queue_maxsize=100)  # ratio=0.46
        decision = policy.decide(ctx)
        assert decision.final_priority_active is True
        assert "queue_ratio" in decision.final_priority_reason

    def test_below_queue_ratio_does_not_trigger(self):
        policy = StreamingPolicy(final_priority_queue_ratio=0.45)
        ctx = _make_ctx(backlog=44, queue_maxsize=100)  # ratio=0.44
        decision = policy.decide(ctx)
        assert decision.final_priority_active is False

    def test_final_priority_suppresses_partial(self):
        policy = StreamingPolicy(final_priority_queue_ratio=0.45)
        ctx = _make_ctx(backlog=46, queue_maxsize=100, last_partial_emit_ms=9000)
        decision = policy.decide(ctx)
        assert decision.final_priority_active is True
        assert decision.emit_partial is False
        assert decision.reason == "final_priority:suppress_partial"

    def test_final_priority_still_emits_final_on_hard_endpoint(self):
        policy = StreamingPolicy(final_priority_queue_ratio=0.45)
        ctx = _make_ctx(
            backlog=46,
            queue_maxsize=100,
            signal=_make_signal(speech_active=True, speech_ended=True, hard_endpoint=True),
            segment_audio_ms=1200,
        )
        decision = policy.decide(ctx)
        assert decision.final_priority_active is True
        assert decision.emit_final is True

    def test_final_priority_recovers_when_queue_clears(self):
        policy = StreamingPolicy(
            final_priority_queue_ratio=0.45,
            final_priority_recover_queue_ratio=0.15,
            final_priority_recover_after_ms=8000,
        )
        # Trigger FP at t=1000ms
        policy.decide(_make_ctx(backlog=46, queue_maxsize=100, now_ms=1000))
        assert policy.final_priority_active is True
        # Queue clears but not enough time yet (only 3s elapsed, need >= 2s min + 8s recover)
        policy.decide(_make_ctx(backlog=10, queue_maxsize=100, now_ms=4000))
        assert policy.final_priority_active is True
        # Now enough time has passed — queue low and 8s recover met
        decision = policy.decide(_make_ctx(backlog=10, queue_maxsize=100, now_ms=12000))
        assert decision.final_priority_active is False

    def test_final_priority_disabled_by_config(self):
        policy = StreamingPolicy(final_priority_enabled=False)
        ctx = _make_ctx(backlog=90, queue_maxsize=100)
        decision = policy.decide(ctx)
        assert decision.final_priority_active is False

    def test_high_final_latency_triggers_final_priority(self):
        policy = StreamingPolicy(final_priority_latency_ms=1800)
        ctx = _make_ctx(backlog=0, queue_maxsize=100, recent_final_latency_ms=2000)
        decision = policy.decide(ctx)
        assert decision.final_priority_active is True
        assert "final_latency" in decision.final_priority_reason

    def test_degraded_plus_drops_triggers_final_priority(self):
        policy = StreamingPolicy(final_priority_queue_ratio=0.45)
        # backlog=7/8 → DEGRADED, plus drops → should enter FP via degraded+drops path
        ctx = _make_ctx(
            backlog=7,
            force_final_queue_size=8,
            queue_maxsize=100,
            dropped_chunks_total=5,
        )
        decision = policy.decide(ctx)
        assert decision.final_priority_active is True


class TestWorkerFinalizeThresholds:
    def test_thresholds_scale_with_soft_final_window(self):
        soft_ms, speech_end_ms = _scaled_finalize_thresholds(
            soft_final_audio_ms=6000,
            min_partial_audio_ms=1000,
            force_final_audio_ms=1800,
        )
        assert soft_ms == 2400
        assert speech_end_ms == 1560

    def test_thresholds_respect_force_final_floor(self):
        soft_ms, speech_end_ms = _scaled_finalize_thresholds(
            soft_final_audio_ms=2000,
            min_partial_audio_ms=300,
            force_final_audio_ms=1800,
        )
        assert soft_ms == 1800
        assert speech_end_ms == 720
        assert speech_end_ms <= soft_ms

    def test_merge_final_with_last_partial_prefers_partial_when_final_drops_prefix(self):
        merged = SourceRuntimeV2._merge_final_with_last_partial(
            final_text="後半句",
            last_partial_text="這是完整的後半句",
        )
        assert merged == "這是完整的後半句"

    def test_merge_final_with_last_partial_keeps_final_when_it_is_not_shorter(self):
        merged = SourceRuntimeV2._merge_final_with_last_partial(
            final_text="這是完整 final",
            last_partial_text="這是完整",
        )
        assert merged == "這是完整 final"
