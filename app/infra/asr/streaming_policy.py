"""StreamingPolicy — 將 SourceRuntimeV2 的 streaming 決策邏輯抽離。

提供 StreamingDecision dataclass 與 StreamingPolicy 類別，
支援 normal / congested / degraded 三階降級策略，
以及 final-priority 負載保護模式。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.infra.asr.endpointing_v2 import EndpointSignal


# ---------------------------------------------------------------------------
# Degradation levels
# ---------------------------------------------------------------------------

DEGRADATION_NORMAL = "normal"
DEGRADATION_CONGESTED = "congested"
DEGRADATION_DEGRADED = "degraded"


@dataclass(slots=True)
class StreamingDecision:
    """單次 chunk 處理後的 streaming 決策。"""
    emit_partial: bool = False
    emit_final: bool = False
    is_early_final: bool = False
    suppress_partial: bool = False
    reset_endpointing_after_final: bool = False
    degradation_level: str = DEGRADATION_NORMAL
    reason: str = ""
    final_priority_active: bool = False
    final_priority_reason: str = ""


@dataclass(slots=True)
class StreamingContext:
    """傳給 StreamingPolicy.decide() 的上下文資訊。"""
    signal: "EndpointSignal"
    segment_audio_ms: int
    now_ms: int
    last_partial_emit_ms: int
    backlog: int
    drop_partial_until_final: bool
    partial_cooldown_until_ms: int
    dropped_chunks_total: int
    # tunable thresholds (從 SourceRuntimeV2 傳入)
    partial_interval_ms: int
    min_partial_audio_ms: int
    soft_endpoint_finalize_audio_ms: int
    speech_end_finalize_audio_ms: int
    adaptive_length_limit_ms: int
    adaptive_length_ceiling_ms: int
    force_final_queue_size: int
    force_final_audio_ms: int
    # Phase 2: final priority inputs
    queue_maxsize: int = 256
    recent_final_latency_ms: int = 0


class StreamingPolicy:
    """決定每個音訊 chunk 是否要 emit partial / final，以及降級策略。

    Parameters
    ----------
    degradation_enabled:
        是否啟用三階降級策略；False 時只用 normal 模式。
    congested_partial_interval_multiplier:
        congested 狀態下 partial interval 倍率（加長間隔）。
    degraded_partial_interval_multiplier:
        degraded 狀態下 partial interval 倍率。
    final_priority_enabled:
        是否啟用 final-priority 模式。啟用後，queue 比例超過門檻或 final
        latency 過高時，停止所有 partial decode 並積極縮短 segment。
    final_priority_queue_ratio:
        觸發 final-priority 的 queue 使用比例（0~1）。
    final_priority_latency_ms:
        觸發 final-priority 的 recent final latency 門檻（毫秒）。
    final_priority_recover_queue_ratio:
        queue 使用比例降至此值以下才可恢復 partial。
    final_priority_recover_after_ms:
        在 queue 比例低於 recover 門檻後，還需額外等待此時間才完全恢復。
    """

    def __init__(
        self,
        *,
        degradation_enabled: bool = True,
        congested_partial_interval_multiplier: float = 1.5,
        degraded_partial_interval_multiplier: float = 3.0,
        final_priority_enabled: bool = True,
        final_priority_queue_ratio: float = 0.45,
        final_priority_latency_ms: int = 1800,
        final_priority_recover_queue_ratio: float = 0.15,
        final_priority_recover_after_ms: int = 8000,
    ) -> None:
        self._degradation_enabled = degradation_enabled
        self._congested_multiplier = max(1.0, congested_partial_interval_multiplier)
        self._degraded_multiplier = max(1.0, degraded_partial_interval_multiplier)
        self._fp_enabled = final_priority_enabled
        self._fp_queue_ratio = max(0.05, min(0.95, float(final_priority_queue_ratio)))
        self._fp_latency_ms = max(500, int(final_priority_latency_ms))
        self._fp_recover_queue_ratio = max(0.0, min(self._fp_queue_ratio - 0.01, float(final_priority_recover_queue_ratio)))
        self._fp_recover_after_ms = max(1000, int(final_priority_recover_after_ms))

        # Mutable final-priority state (updated on every decide() call).
        self._fp_active: bool = False
        self._fp_reason: str = ""
        self._fp_since_ms: int = 0
        # Last observed dropped_chunks_total for "新增 drop" detection.
        self._fp_last_dropped: int = 0
        # Monotonic ms of the last time we saw a new chunk drop.
        self._fp_last_new_drop_ms: int = 0
        # Monotonic ms when queue first dropped below recover_queue_ratio.
        self._fp_recover_eligible_ms: int = 0

    # ------------------------------------------------------------------
    # Public read-only properties (for stats / UI)
    # ------------------------------------------------------------------

    @property
    def final_priority_active(self) -> bool:
        return self._fp_active

    @property
    def final_priority_reason(self) -> str:
        return self._fp_reason

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def decide(self, ctx: StreamingContext) -> StreamingDecision:
        """依當前上下文回傳 StreamingDecision。"""
        degradation = self._compute_degradation(ctx)
        self._update_final_priority(ctx, degradation)

        decision = StreamingDecision(
            degradation_level=degradation,
            final_priority_active=self._fp_active,
            final_priority_reason=self._fp_reason,
        )

        # Compute effective thresholds – final priority applies aggressive reductions.
        if self._fp_active:
            eff_force_final_audio_ms = max(800, ctx.force_final_audio_ms // 2)
            eff_force_final_queue_size = max(2, ctx.force_final_queue_size // 2)
            eff_soft_ep_ms = max(600, ctx.soft_endpoint_finalize_audio_ms // 2)
            eff_speech_end_ms = max(400, ctx.speech_end_finalize_audio_ms // 2)
            eff_adaptive_limit_ms = max(1200, ctx.adaptive_length_limit_ms // 2)
            eff_ceiling_ms = max(2400, ctx.adaptive_length_ceiling_ms // 2)
        else:
            eff_force_final_audio_ms = ctx.force_final_audio_ms
            eff_force_final_queue_size = ctx.force_final_queue_size
            eff_soft_ep_ms = ctx.soft_endpoint_finalize_audio_ms
            eff_speech_end_ms = ctx.speech_end_finalize_audio_ms
            eff_adaptive_limit_ms = ctx.adaptive_length_limit_ms
            eff_ceiling_ms = ctx.adaptive_length_ceiling_ms

        # --- finalization logic ---
        should_force_final = (
            ctx.backlog >= eff_force_final_queue_size
            and ctx.segment_audio_ms >= eff_force_final_audio_ms
        )

        finalize_on_soft = (
            ctx.signal.soft_endpoint
            and ctx.segment_audio_ms >= eff_soft_ep_ms
        )
        finalize_on_speech_end = (
            ctx.signal.speech_ended
            and ctx.segment_audio_ms >= eff_speech_end_ms
        )
        finalize_on_pause_turn = (
            ctx.signal.pause_ms >= 240.0
            and ctx.segment_audio_ms >= max(900, ctx.min_partial_audio_ms + 180)
        )
        finalize_on_length = (
            ctx.segment_audio_ms >= eff_adaptive_limit_ms
            and ctx.signal.pause_ms >= 180.0
        )
        finalize_on_ceiling = ctx.segment_audio_ms >= eff_ceiling_ms

        # degraded mode: lower finalize threshold to avoid unbounded growth
        if degradation == DEGRADATION_DEGRADED:
            finalize_on_ceiling = ctx.segment_audio_ms >= max(4000, eff_ceiling_ms // 2)

        should_finalize = (
            ctx.signal.hard_endpoint
            or finalize_on_speech_end
            or finalize_on_soft
            or finalize_on_pause_turn
            or finalize_on_length
            or finalize_on_ceiling
            or should_force_final
        )

        if should_finalize:
            decision.emit_final = True
            decision.reset_endpointing_after_final = (
                (finalize_on_soft or finalize_on_pause_turn) and not ctx.signal.hard_endpoint
            )

            if should_force_final:
                decision.reason = "force_final(queue_pressure)"
                decision.is_early_final = False
            elif ctx.signal.hard_endpoint:
                decision.reason = "hard_endpoint"
                decision.is_early_final = False
            elif finalize_on_speech_end:
                decision.reason = "speech_ended"
                decision.is_early_final = False
            elif finalize_on_soft:
                decision.reason = "soft_endpoint"
                decision.is_early_final = True
            elif finalize_on_pause_turn:
                decision.reason = "pause_turn"
                decision.is_early_final = True
            elif finalize_on_ceiling:
                decision.reason = "ceiling"
                decision.is_early_final = False
            else:
                decision.reason = "adaptive_length"
                decision.is_early_final = False
            return decision

        # --- partial logic ---
        # Final-priority and degraded both suppress partials entirely.
        if self._fp_active:
            decision.suppress_partial = True
            decision.reason = "final_priority:suppress_partial"
            return decision

        if degradation == DEGRADATION_DEGRADED:
            decision.suppress_partial = True
            decision.reason = "degraded:suppress_partial"
            return decision

        effective_interval = ctx.partial_interval_ms
        if degradation == DEGRADATION_CONGESTED:
            effective_interval = int(ctx.partial_interval_ms * self._congested_multiplier)
            if ctx.backlog >= max(2, ctx.force_final_queue_size // 2):
                decision.suppress_partial = True
                decision.reason = "congested:suppress_partial"
                return decision

        can_emit_partial = (
            ctx.signal.speech_active
            and ctx.segment_audio_ms >= ctx.min_partial_audio_ms
            and ctx.backlog <= 1
            and not ctx.drop_partial_until_final
            and ctx.now_ms >= ctx.partial_cooldown_until_ms
        )
        if can_emit_partial and (ctx.now_ms - ctx.last_partial_emit_ms) >= effective_interval:
            decision.emit_partial = True
            decision.reason = f"partial({degradation})"

        return decision

    # ------------------------------------------------------------------
    # Degradation computation
    # ------------------------------------------------------------------

    def _compute_degradation(self, ctx: StreamingContext) -> str:
        if not self._degradation_enabled:
            return DEGRADATION_NORMAL

        # queue 壓力分數 (0~1)
        queue_pressure = min(1.0, ctx.backlog / max(1, ctx.force_final_queue_size))

        if queue_pressure >= 0.75 or ctx.dropped_chunks_total >= 20:
            return DEGRADATION_DEGRADED
        if queue_pressure >= 0.4 or ctx.dropped_chunks_total >= 5:
            return DEGRADATION_CONGESTED
        return DEGRADATION_NORMAL

    # ------------------------------------------------------------------
    # Final-priority state machine
    # ------------------------------------------------------------------

    def _update_final_priority(self, ctx: StreamingContext, degradation: str) -> None:
        """Update _fp_active based on queue ratio, latency, and recovery conditions."""
        if not self._fp_enabled:
            return

        # Track new chunk drops (for recovery eligibility).
        if ctx.dropped_chunks_total > self._fp_last_dropped:
            self._fp_last_dropped = ctx.dropped_chunks_total
            self._fp_last_new_drop_ms = ctx.now_ms

        queue_ratio = ctx.backlog / max(1, ctx.queue_maxsize)

        if not self._fp_active:
            self._check_enter(ctx, queue_ratio, degradation)
        else:
            self._check_exit(ctx, queue_ratio)

    def _check_enter(
        self,
        ctx: StreamingContext,
        queue_ratio: float,
        degradation: str,
    ) -> None:
        """Evaluate whether final-priority mode should be activated."""
        reason = ""
        if queue_ratio >= self._fp_queue_ratio:
            reason = f"queue_ratio={queue_ratio:.2f}>={self._fp_queue_ratio:.2f}"
        elif (
            ctx.recent_final_latency_ms > 0
            and ctx.recent_final_latency_ms >= self._fp_latency_ms
        ):
            reason = f"final_latency={ctx.recent_final_latency_ms}ms>={self._fp_latency_ms}ms"
        elif ctx.dropped_chunks_total > 0 and degradation == DEGRADATION_DEGRADED:
            # When the degradation engine reaches DEGRADED and drops are occurring,
            # upgrade to final-priority to protect remaining final decode slots.
            reason = f"degraded+drops={ctx.dropped_chunks_total}"

        if reason:
            self._fp_active = True
            self._fp_reason = reason
            self._fp_since_ms = ctx.now_ms
            self._fp_recover_eligible_ms = 0

    def _check_exit(self, ctx: StreamingContext, queue_ratio: float) -> None:
        """Evaluate whether final-priority mode should be deactivated."""
        # Must be running for at least 2 seconds before recovery is considered.
        if ctx.now_ms - self._fp_since_ms < 2000:
            return

        if queue_ratio > self._fp_recover_queue_ratio:
            # Still under pressure – reset the recovery eligibility timer.
            self._fp_recover_eligible_ms = 0
            return

        # Queue has relaxed – start the recovery countdown on first observation.
        if self._fp_recover_eligible_ms == 0:
            self._fp_recover_eligible_ms = ctx.now_ms

        no_recent_drops = (ctx.now_ms - self._fp_last_new_drop_ms) >= self._fp_recover_after_ms
        elapsed_since_eligible = ctx.now_ms - self._fp_recover_eligible_ms
        recovered = (
            elapsed_since_eligible >= self._fp_recover_after_ms
            and no_recent_drops
        )
        if recovered:
            self._fp_active = False
            self._fp_reason = ""
            self._fp_since_ms = 0
            self._fp_recover_eligible_ms = 0


__all__ = [
    "DEGRADATION_NORMAL",
    "DEGRADATION_CONGESTED",
    "DEGRADATION_DEGRADED",
    "StreamingDecision",
    "StreamingContext",
    "StreamingPolicy",
]
