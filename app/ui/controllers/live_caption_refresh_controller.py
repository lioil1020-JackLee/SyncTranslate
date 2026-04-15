"""LiveCaptionRefreshController — manages live caption panel refresh.

Extracted from MainWindow to decouple transcript polling from UI layout.
This controller owns:
- transcript buffer polling
- line cache comparison
- panel status update scheduling

MainWindow retains its QTimer but can delegate the refresh logic here.
"""
from __future__ import annotations

from typing import Any, Callable


class LiveCaptionRefreshController:
    """Polls the transcript buffer and updates the live caption panel.

    Parameters
    ----------
    get_transcript_lines:
        Callable returning (local_lines, local_translated, remote_lines, remote_translated)
        — all as list[str].  This is a snapshot of the current transcript buffer.
    on_remote_update:
        Called with (lines, translated_lines) when remote panel needs a refresh.
    on_local_update:
        Called with (lines, translated_lines) when local panel needs a refresh.
    on_panel_status_update:
        Called with a status dict when panel statuses change.
    """

    def __init__(
        self,
        *,
        get_transcript_lines: Callable[[], tuple[list[str], list[str], list[str], list[str]]],
        on_remote_update: Callable[[list[str], list[str]], None],
        on_local_update: Callable[[list[str], list[str]], None],
        on_panel_status_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._get_lines = get_transcript_lines
        self._on_remote = on_remote_update
        self._on_local = on_local_update
        self._on_status = on_panel_status_update

        self._remote_cache: list[str] = []
        self._remote_translated_cache: list[str] = []
        self._local_cache: list[str] = []
        self._local_translated_cache: list[str] = []

    # ------------------------------------------------------------------
    # Refresh (called by QTimer)
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Check for transcript changes and fire update callbacks if needed."""
        local, local_t, remote, remote_t = self._get_lines()

        if remote != self._remote_cache or remote_t != self._remote_translated_cache:
            self._remote_cache = list(remote)
            self._remote_translated_cache = list(remote_t)
            self._on_remote(remote, remote_t)

        if local != self._local_cache or local_t != self._local_translated_cache:
            self._local_cache = list(local)
            self._local_translated_cache = list(local_t)
            self._on_local(local, local_t)

    def reset(self) -> None:
        """Clear caches (call on session start/clear)."""
        self._remote_cache = []
        self._remote_translated_cache = []
        self._local_cache = []
        self._local_translated_cache = []


__all__ = ["LiveCaptionRefreshController"]
