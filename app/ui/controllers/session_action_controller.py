"""SessionActionController — handles start/stop session lifecycle.

Extracted from MainWindow to decouple session lifecycle logic from UI rendering.
MainWindow keeps its existing timer/QThread infrastructure; this controller
provides a cleaner interface for triggering and monitoring session state.

Design note:
    MainWindow can use this controller directly, or keep its existing code.
    This module exists so the session logic can be unit-tested without a
    running Qt application.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import Thread
from typing import Any, Callable


@dataclass
class SessionActionResult:
    """Result of a session start or stop attempt."""
    success: bool
    action: str
    error: str = ""


class SessionActionController:
    """Coordinates session start/stop with status reporting.

    Parameters
    ----------
    on_start:
        Coroutine/callable that initiates the ASR session.  Called with
        ``route``, ``sample_rate``, and ``chunk_ms`` keyword arguments.
    on_stop:
        Callable that tears down the ASR session.
    on_status_changed:
        Optional UI callback called with a human-readable status string.
    on_error:
        Optional error callback.
    """

    def __init__(
        self,
        *,
        on_start: Callable[..., None],
        on_stop: Callable[[], None],
        on_status_changed: Callable[[str], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self._on_start = on_start
        self._on_stop = on_stop
        self._on_status_changed = on_status_changed
        self._on_error = on_error
        self._running: bool = False
        self._action_running: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def action_in_progress(self) -> bool:
        return self._action_running

    def request_start(self, *, route: Any, sample_rate: int, chunk_ms: int = 100) -> None:
        """Request a session start in the background."""
        if self._action_running:
            return
        self._action_running = True
        self._notify_status("啟動中…")
        t = Thread(
            target=self._do_start,
            kwargs={"route": route, "sample_rate": sample_rate, "chunk_ms": chunk_ms},
            daemon=True,
        )
        t.start()

    def request_stop(self) -> None:
        """Request a session stop in the background."""
        if self._action_running:
            return
        self._action_running = True
        self._notify_status("停止中…")
        t = Thread(target=self._do_stop, daemon=True)
        t.start()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _do_start(self, *, route: Any, sample_rate: int, chunk_ms: int) -> None:
        try:
            self._on_start(route=route, sample_rate=sample_rate, chunk_ms=chunk_ms)
            self._running = True
            self._notify_status("執行中")
        except Exception as exc:  # noqa: BLE001
            self._running = False
            if self._on_error:
                self._on_error(f"start_session failed: {exc}")
            self._notify_status("啟動失敗")
        finally:
            self._action_running = False

    def _do_stop(self) -> None:
        try:
            self._on_stop()
            self._running = False
            self._notify_status("已停止")
        except Exception as exc:  # noqa: BLE001
            if self._on_error:
                self._on_error(f"stop_session failed: {exc}")
            self._notify_status("停止失敗")
        finally:
            self._action_running = False

    def _notify_status(self, message: str) -> None:
        if self._on_status_changed:
            try:
                self._on_status_changed(message)
            except Exception:  # noqa: BLE001
                pass


__all__ = ["SessionActionController", "SessionActionResult"]
