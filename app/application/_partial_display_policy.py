"""Partial-display policy helpers extracted from AudioRouter.

Decides whether a streaming ASR partial transcript should be shown to the user,
based on the configured display strategy and stability of the text over consecutive
updates.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.domain.constants import (
    DISPLAY_PARTIAL_ALL,
    DISPLAY_PARTIAL_NONE,
    DISPLAY_PARTIAL_STABLE_ONLY,
)


@dataclass(slots=True)
class PartialDisplayState:
    """Per-channel mutable state for the stable-partial progression detector."""

    utterance_id: str = ""
    last_text: str = ""
    repeat_count: int = 0
    last_displayed_text: str = ""


class PartialDisplayPolicy:
    """Stateful policy that decides whether a partial transcript should be displayed.

    Parameters
    ----------
    runtime_config:
        The live ``AppConfig`` instance (or ``None`` before the first config is
        applied).  The policy reads ``runtime.display_partial_strategy``,
        ``runtime.stable_partial_min_repeats``, and
        ``runtime.partial_stability_max_delta_chars`` on every call so that
        hot-applied config changes take effect immediately.
    """

    def __init__(self, runtime_config=None) -> None:
        self._runtime_config = runtime_config
        self._state: dict[str, PartialDisplayState] = {}

    def update_config(self, runtime_config) -> None:
        self._runtime_config = runtime_config

    def reset(self) -> None:
        self._state.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_display(
        self,
        *,
        channel: str,
        utterance_id: str | None,
        text: str,
        is_final: bool,
    ) -> tuple[bool, bool]:
        """Return ``(should_display, is_stable_partial)``.

        Parameters
        ----------
        channel:
            Source channel identifier ("local" / "remote").
        utterance_id:
            Current utterance ID; ``None`` means unknown.
        text:
            The partial or final transcript text.
        is_final:
            ``True`` for final transcripts — always displayed, state cleared.
        """
        if is_final:
            self._state.pop(channel, None)
            return True, False
        strategy = self._display_partial_strategy()
        if strategy == DISPLAY_PARTIAL_ALL:
            return True, False
        if strategy == DISPLAY_PARTIAL_NONE:
            return False, False
        normalized = text.strip()
        if not normalized or not utterance_id:
            return False, False
        state = self._state.get(channel)
        if state is None or state.utterance_id != utterance_id:
            self._state[channel] = PartialDisplayState(
                utterance_id=utterance_id,
                last_text=normalized,
                repeat_count=1,
            )
            return False, False
        if self.is_stable_progression(state.last_text, normalized):
            state.repeat_count += 1
        else:
            state.repeat_count = 1
        state.last_text = normalized
        is_stable = state.repeat_count >= self._stable_partial_min_repeats()
        if not is_stable:
            return False, False
        if normalized == state.last_displayed_text:
            return False, True
        state.last_displayed_text = normalized
        return True, True

    # ------------------------------------------------------------------
    # Stability helpers
    # ------------------------------------------------------------------

    def is_stable_progression(self, previous: str, current: str) -> bool:
        """Return True when *current* is a stable continuation of *previous*.

        A progression is stable when the text grows/shrinks within the configured
        delta-char threshold or the shared prefix ratio is high enough.
        """
        if previous == current:
            return True
        delta_limit = self._partial_stability_max_delta_chars()
        if not previous or not current:
            return False

        min_shared_prefix = min(6, len(previous), len(current))
        shared_prefix = 0
        for prev_char, curr_char in zip(previous, current):
            if prev_char != curr_char:
                break
            shared_prefix += 1

        if current.startswith(previous):
            growth = len(current) - len(previous)
            if growth <= delta_limit:
                return True
            return shared_prefix >= max(min_shared_prefix, int(len(previous) * 0.75))
        if previous.startswith(current):
            shrink = len(previous) - len(current)
            if shrink <= delta_limit:
                return True
            return shared_prefix >= max(min_shared_prefix, int(len(current) * 0.85))
        if shared_prefix <= 0:
            return False
        previous_tail = len(previous) - shared_prefix
        current_tail = len(current) - shared_prefix
        shorter_len = min(len(previous), len(current))
        shared_ratio = shared_prefix / max(1, shorter_len)
        if previous_tail <= delta_limit and current_tail <= delta_limit:
            return True
        return shared_ratio >= 0.72 and previous_tail <= delta_limit * 2 and current_tail <= delta_limit * 2

    # ------------------------------------------------------------------
    # Config accessors
    # ------------------------------------------------------------------

    def _display_partial_strategy(self) -> str:
        runtime = getattr(self._runtime_config, "runtime", None)
        value = str(
            getattr(runtime, "display_partial_strategy", DISPLAY_PARTIAL_STABLE_ONLY) or DISPLAY_PARTIAL_STABLE_ONLY
        ).strip().lower()
        if value in {DISPLAY_PARTIAL_ALL, DISPLAY_PARTIAL_NONE, DISPLAY_PARTIAL_STABLE_ONLY}:
            return value
        return DISPLAY_PARTIAL_STABLE_ONLY

    def _stable_partial_min_repeats(self) -> int:
        runtime = getattr(self._runtime_config, "runtime", None)
        value = int(getattr(runtime, "stable_partial_min_repeats", 2)) if runtime is not None else 2
        return max(1, value)

    def _partial_stability_max_delta_chars(self) -> int:
        runtime = getattr(self._runtime_config, "runtime", None)
        value = int(getattr(runtime, "partial_stability_max_delta_chars", 8)) if runtime is not None else 8
        return max(1, value)


__all__ = ["PartialDisplayPolicy", "PartialDisplayState"]
