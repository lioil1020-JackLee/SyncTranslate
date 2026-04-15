"""HealthcheckController — manages healthcheck start, poll, and status update.

Extracted from MainWindow to decouple healthcheck UI logic from rendering.

Usage:

    self._healthcheck_ctrl = HealthcheckController(
        healthcheck_service=self._healthcheck_service,
        on_status_update=self._on_healthcheck_status,
    )
    # Kick off a check:
    self._healthcheck_ctrl.run()
    # In a QTimer tick:
    self._healthcheck_ctrl.tick()
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class HealthcheckStatus:
    """Snapshot of the latest healthcheck result."""
    running: bool = False
    ok: bool = False
    message: str = ""
    details: dict[str, Any] | None = None


class HealthcheckController:
    """Polls a HealthCheckService and reports status to the UI.

    Parameters
    ----------
    healthcheck_service:
        The service object that performs the actual check.
        Must have: ``run_async()`` or ``start()``, ``is_running() -> bool``,
        ``result() -> dict | None``.
    on_status_update:
        Called with a HealthcheckStatus whenever the status changes.
    on_error:
        Optional error callback.
    """

    def __init__(
        self,
        *,
        healthcheck_service: Any,
        on_status_update: Callable[[HealthcheckStatus], None],
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self._service = healthcheck_service
        self._on_status = on_status_update
        self._on_error = on_error
        self._last_status: HealthcheckStatus = HealthcheckStatus()
        self._polling: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start a healthcheck in the background."""
        try:
            start_fn = getattr(self._service, "run_async", None) or getattr(self._service, "start", None)
            if callable(start_fn):
                start_fn()
            self._polling = True
            self._on_status(HealthcheckStatus(running=True, message="健康檢查中…"))
        except Exception as exc:  # noqa: BLE001
            if self._on_error:
                self._on_error(f"healthcheck start failed: {exc}")

    def tick(self) -> None:
        """Poll the service for completion and update status if changed.

        Call this from a QTimer handler.
        """
        if not self._polling:
            return
        try:
            is_running_fn = getattr(self._service, "is_running", None)
            if callable(is_running_fn) and is_running_fn():
                return  # still in progress
            self._polling = False
            result_fn = getattr(self._service, "result", None)
            result = result_fn() if callable(result_fn) else None
            if result is None:
                status = HealthcheckStatus(running=False, ok=False, message="健康檢查未回傳結果")
            else:
                ok = bool(result.get("ok", False))
                msg = str(result.get("message", "完成"))
                status = HealthcheckStatus(running=False, ok=ok, message=msg, details=result)
            self._last_status = status
            self._on_status(status)
        except Exception as exc:  # noqa: BLE001
            self._polling = False
            if self._on_error:
                self._on_error(f"healthcheck poll failed: {exc}")

    @property
    def last_status(self) -> HealthcheckStatus:
        return self._last_status


__all__ = ["HealthcheckController", "HealthcheckStatus"]
