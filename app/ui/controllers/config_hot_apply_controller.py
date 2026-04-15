"""ConfigHotApplyController — manages live config apply logic.

Extracted from MainWindow to decouple config debounce and hot-apply from UI.
Supports:
- Pending apply: schedule a config change to apply after a short debounce
- Hot switch: immediately apply a config change to the running router

Usage in MainWindow:

    self._config_hot_apply = ConfigHotApplyController(
        on_apply=self._do_apply_live_config,
        on_error=self._report_error,
        debounce_ms=150,
    )
    # In a QTimer connected handler:
    self._config_hot_apply.tick()
    # When config changes:
    self._config_hot_apply.mark_pending()
"""
from __future__ import annotations

import time
from typing import Callable


class ConfigHotApplyController:
    """Debounced config hot-apply controller.

    Parameters
    ----------
    on_apply:
        Callable invoked when a pending config apply fires.
    on_error:
        Optional error callback.
    debounce_ms:
        Milliseconds to wait after ``mark_pending()`` before calling ``on_apply``.
    """

    def __init__(
        self,
        *,
        on_apply: Callable[[], None],
        on_error: Callable[[str], None] | None = None,
        debounce_ms: int = 150,
    ) -> None:
        self._on_apply = on_apply
        self._on_error = on_error
        self._debounce_ms = max(0, int(debounce_ms))
        self._pending: bool = False
        self._pending_at: float = 0.0
        self._suspend: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mark_pending(self) -> None:
        """Schedule a config apply after the debounce delay."""
        self._pending = True
        self._pending_at = time.monotonic()

    def suspend(self) -> None:
        """Temporarily prevent applies (e.g. while a session is starting)."""
        self._suspend = True

    def resume(self) -> None:
        """Re-enable applies."""
        self._suspend = False

    def apply_immediately(self) -> None:
        """Fire an apply right now, bypassing the debounce."""
        self._pending = False
        self._do_apply()

    def tick(self) -> None:
        """Call this regularly (e.g. from a QTimer) to check if an apply is due."""
        if not self._pending or self._suspend:
            return
        elapsed_ms = (time.monotonic() - self._pending_at) * 1000
        if elapsed_ms >= self._debounce_ms:
            self._pending = False
            self._do_apply()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _do_apply(self) -> None:
        try:
            self._on_apply()
        except Exception as exc:  # noqa: BLE001
            if self._on_error:
                self._on_error(f"config apply failed: {exc}")


__all__ = ["ConfigHotApplyController"]
