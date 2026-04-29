"""Adaptive tuning mixin for SourceRuntimeV2.

Provides ``_adaptive_length_limit_ms`` and ``_recompute_adaptive_tuning`` as a
Mixin so the methods can be maintained and tested separately from the main
worker body, without copying any of the instance-state definitions.

Usage::

    class SourceRuntimeV2(_AdaptiveTunerMixin, ...):
        ...

The Mixin only references ``self._*`` attributes that are defined by the host
class (``SourceRuntimeV2``).  No additional ``__init__`` is required.
"""
from __future__ import annotations

from app.infra.asr.endpointing_v2 import EndpointSignal
from app.infra.asr.streaming_policy import DEGRADATION_CONGESTED, DEGRADATION_DEGRADED


class _AdaptiveTunerMixin:
    """Mixin that adds adaptive timing methods to SourceRuntimeV2.

    Accesses the following ``self._*`` attributes (all defined by the host):
    - ``_adaptive_enabled`` (bool)
    - ``_adaptive_partial_latencies`` (deque[int])
    - ``_adaptive_final_latencies`` (deque[int])
    - ``_adaptive_recent_audio_ms`` (deque[int])
    - ``_adaptive_length_floor_ms`` (int)
    - ``_adaptive_length_ceiling_ms`` (int)
    - ``_partial_interval_floor_ms`` (int)
    - ``_base_partial_interval_ms`` (int)
    - ``_base_soft_final_audio_ms`` (int)
    - ``_base_soft_endpoint_finalize_audio_ms`` (int)
    - ``_base_speech_end_finalize_audio_ms`` (int)
    - ``_force_final_audio_ms`` (int)
    - ``_last_degradation_level`` (str)
    - ``_partial_interval_ms`` (int) — written
    - ``_soft_final_audio_ms`` (int) — written
    - ``_soft_endpoint_finalize_audio_ms`` (int) — written
    - ``_speech_end_finalize_audio_ms`` (int) — written
    - ``_debug`` (callable) — method on host
    """

    def _adaptive_length_limit_ms(self, *, signal: EndpointSignal) -> int:
        """Return the maximum segment length given current endpoint pause."""
        if signal.pause_ms >= 280.0:
            return max(self._adaptive_length_floor_ms, 2800)
        if signal.pause_ms >= 180.0:
            return max(self._adaptive_length_floor_ms, 3800)
        if signal.pause_ms >= 90.0:
            return max(self._adaptive_length_floor_ms, 5200)
        return self._adaptive_length_ceiling_ms

    def _recompute_adaptive_tuning(
        self, *, now_ms: int, segment_audio_ms: int, signal: EndpointSignal
    ) -> None:
        """Adjust timing parameters in response to observed latency and queue health."""
        if not self._adaptive_enabled:
            return

        avg_partial_latency_ms = (
            sum(self._adaptive_partial_latencies) / len(self._adaptive_partial_latencies)
            if self._adaptive_partial_latencies
            else 0.0
        )
        avg_final_latency_ms = (
            sum(self._adaptive_final_latencies) / len(self._adaptive_final_latencies)
            if self._adaptive_final_latencies
            else 0.0
        )
        avg_final_audio_ms = (
            sum(self._adaptive_recent_audio_ms) / len(self._adaptive_recent_audio_ms)
            if self._adaptive_recent_audio_ms
            else 0.0
        )

        next_partial_interval_ms = self._base_partial_interval_ms
        next_soft_final_audio_ms = self._base_soft_final_audio_ms
        next_soft_endpoint_finalize_audio_ms = self._base_soft_endpoint_finalize_audio_ms
        next_speech_end_finalize_audio_ms = self._base_speech_end_finalize_audio_ms

        if avg_final_audio_ms and avg_final_audio_ms <= 1600:
            next_partial_interval_ms = max(
                self._partial_interval_floor_ms, self._base_partial_interval_ms - 140
            )
            next_soft_final_audio_ms = max(
                self._force_final_audio_ms,
                int(round(self._base_soft_final_audio_ms * 0.76)),
            )
        elif avg_final_audio_ms >= 3200:
            next_partial_interval_ms = max(
                self._partial_interval_floor_ms, self._base_partial_interval_ms - 40
            )
            next_soft_final_audio_ms = max(
                self._force_final_audio_ms,
                int(round(self._base_soft_final_audio_ms * 0.70)),
            )
        elif avg_final_audio_ms >= 2400:
            next_soft_final_audio_ms = max(
                self._force_final_audio_ms,
                int(round(self._base_soft_final_audio_ms * 0.80)),
            )

        if avg_partial_latency_ms >= 850 or avg_final_latency_ms >= 1400:
            next_partial_interval_ms = max(
                next_partial_interval_ms, self._base_partial_interval_ms + 120
            )
            next_soft_final_audio_ms = max(
                self._force_final_audio_ms,
                int(round(self._base_soft_final_audio_ms * 0.72)),
            )

        if self._last_degradation_level == DEGRADATION_CONGESTED:
            next_partial_interval_ms = max(
                next_partial_interval_ms, self._base_partial_interval_ms + 120
            )
        elif self._last_degradation_level == DEGRADATION_DEGRADED:
            next_partial_interval_ms = max(
                next_partial_interval_ms, self._base_partial_interval_ms + 200
            )
            next_soft_final_audio_ms = max(
                self._force_final_audio_ms,
                int(round(self._base_soft_final_audio_ms * 0.68)),
            )

        if signal.pause_ms >= 160.0 and segment_audio_ms >= max(
            1800, int(self._base_soft_final_audio_ms * 0.7)
        ):
            next_soft_final_audio_ms = min(
                next_soft_final_audio_ms,
                max(
                    self._force_final_audio_ms,
                    int(round(self._base_soft_final_audio_ms * 0.78)),
                ),
            )

        next_partial_interval_ms = max(self._partial_interval_floor_ms, int(next_partial_interval_ms))
        next_soft_final_audio_ms = max(self._force_final_audio_ms, int(next_soft_final_audio_ms))
        next_soft_endpoint_finalize_audio_ms = min(
            next_soft_final_audio_ms,
            max(
                self._force_final_audio_ms,
                int(
                    round(
                        self._base_soft_endpoint_finalize_audio_ms
                        * (next_soft_final_audio_ms / max(1, self._base_soft_final_audio_ms))
                    )
                ),
            ),
        )
        next_speech_end_finalize_audio_ms = min(
            next_soft_endpoint_finalize_audio_ms,
            max(
                900,
                int(
                    round(
                        self._base_speech_end_finalize_audio_ms
                        * (next_soft_final_audio_ms / max(1, self._base_soft_final_audio_ms))
                    )
                ),
            ),
        )

        changed = any(
            (
                next_partial_interval_ms != self._partial_interval_ms,
                next_soft_final_audio_ms != self._soft_final_audio_ms,
                next_soft_endpoint_finalize_audio_ms != self._soft_endpoint_finalize_audio_ms,
                next_speech_end_finalize_audio_ms != self._speech_end_finalize_audio_ms,
            )
        )
        self._partial_interval_ms = next_partial_interval_ms
        self._soft_final_audio_ms = next_soft_final_audio_ms
        self._soft_endpoint_finalize_audio_ms = next_soft_endpoint_finalize_audio_ms
        self._speech_end_finalize_audio_ms = next_speech_end_finalize_audio_ms
        if changed:
            self._debug(
                "v2 adaptive "
                f"partial_ms={self._partial_interval_ms} "
                f"soft_final_ms={self._soft_final_audio_ms} "
                f"soft_finalize_ms={self._soft_endpoint_finalize_audio_ms} "
                f"speech_end_finalize_ms={self._speech_end_finalize_audio_ms}"
            )


__all__ = ["_AdaptiveTunerMixin"]
