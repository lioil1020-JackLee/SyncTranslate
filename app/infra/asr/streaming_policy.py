"""StreamingPolicy — 將 SourceRuntimeV2 的 streaming 決策邏輯抽離。

提供 StreamingDecision dataclass 與 StreamingPolicy 類別，
支援 normal / congested / degraded 三階降級策略。
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
    """

    def __init__(
        self,
        *,
        degradation_enabled: bool = True,
        congested_partial_interval_multiplier: float = 1.5,
        degraded_partial_interval_multiplier: float = 3.0,
    ) -> None:
        self._degradation_enabled = degradation_enabled
        self._congested_multiplier = max(1.0, congested_partial_interval_multiplier)
        self._degraded_multiplier = max(1.0, degraded_partial_interval_multiplier)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def decide(self, ctx: StreamingContext) -> StreamingDecision:
        """依當前上下文回傳 StreamingDecision。"""
        degradation = self._compute_degradation(ctx)

        decision = StreamingDecision(degradation_level=degradation)

        # --- finalization logic ---
        should_force_final = (
            ctx.backlog >= ctx.force_final_queue_size
            and ctx.segment_audio_ms >= ctx.force_final_audio_ms
        )

        finalize_on_soft = (
            ctx.signal.soft_endpoint
            and ctx.segment_audio_ms >= ctx.soft_endpoint_finalize_audio_ms
        )
        finalize_on_speech_end = (
            ctx.signal.speech_ended
            and ctx.segment_audio_ms >= ctx.speech_end_finalize_audio_ms
        )
        finalize_on_pause_turn = (
            ctx.signal.pause_ms >= 240.0
            and ctx.segment_audio_ms >= max(900, ctx.min_partial_audio_ms + 180)
        )
        finalize_on_length = (
            ctx.segment_audio_ms >= ctx.adaptive_length_limit_ms
            and ctx.signal.pause_ms >= 180.0
        )
        finalize_on_ceiling = ctx.segment_audio_ms >= ctx.adaptive_length_ceiling_ms

        # degraded mode: lower finalize threshold to avoid unbounded growth
        if degradation == DEGRADATION_DEGRADED:
            finalize_on_ceiling = ctx.segment_audio_ms >= max(4000, ctx.adaptive_length_ceiling_ms // 2)

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
            decision.is_early_final = not ctx.signal.hard_endpoint
            decision.reset_endpointing_after_final = (
                (finalize_on_soft or finalize_on_pause_turn) and not ctx.signal.hard_endpoint
            )

            if should_force_final:
                decision.reason = "force_final(queue_pressure)"
            elif ctx.signal.hard_endpoint:
                decision.reason = "hard_endpoint"
            elif finalize_on_speech_end:
                decision.reason = "speech_ended"
            elif finalize_on_soft:
                decision.reason = "soft_endpoint"
            elif finalize_on_pause_turn:
                decision.reason = "pause_turn"
            elif finalize_on_ceiling:
                decision.reason = "ceiling"
            else:
                decision.reason = "adaptive_length"
            return decision

        # --- partial logic ---
        if degradation == DEGRADATION_DEGRADED:
            decision.suppress_partial = True
            decision.reason = "degraded:suppress_partial"
            return decision

        effective_interval = ctx.partial_interval_ms
        if degradation == DEGRADATION_CONGESTED:
            effective_interval = int(ctx.partial_interval_ms * self._congested_multiplier)

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


__all__ = [
    "DEGRADATION_NORMAL",
    "DEGRADATION_CONGESTED",
    "DEGRADATION_DEGRADED",
    "StreamingDecision",
    "StreamingContext",
    "StreamingPolicy",
]
