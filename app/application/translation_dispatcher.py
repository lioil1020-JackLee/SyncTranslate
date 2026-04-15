"""TranslationDispatcher — manages async translation queue and overflow policy.

Extracted from AudioRouter to make translation queue logic independently testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Callable

from app.infra.asr.contracts import ASREventWithSource
from app.infra.translation.engine import TranslatorManager


@dataclass
class TranslationDispatcherStats:
    overflow_count: int = 0
    queue_size: int = 0
    worker_alive: bool = False


class TranslationDispatcher:
    """Async translation queue with configurable overflow policy.

    When async translation is enabled, ASR events are pushed onto a bounded
    queue and processed by a background worker.  If the queue is full, the
    oldest event is dropped (tail-drop policy) to avoid head-of-line blocking.

    Parameters
    ----------
    translator_manager:
        The translation engine to call for each event.
    on_translated:
        Callback invoked with the translation result after each successful translation.
    on_skipped:
        Optional callback invoked when translation was skipped.
    queue_maxsize:
        Maximum number of pending translation events.
    """

    def __init__(
        self,
        *,
        translator_manager: TranslatorManager,
        on_translated: Callable[[object], None],
        on_skipped: Callable[[str, str], None] | None = None,
        queue_maxsize: int = 32,
    ) -> None:
        self._translator = translator_manager
        self._on_translated = on_translated
        self._on_skipped = on_skipped
        self._queue: Queue[ASREventWithSource | None] = Queue(maxsize=max(4, queue_maxsize))
        self._stop_event = Event()
        self._worker: Thread | None = None
        self._overflow_count: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self.stop()
        self._stop_event.clear()
        self._worker = Thread(target=self._run, daemon=True, name="translation-dispatcher")
        self._worker.start()

    def stop(self) -> None:
        self._stop_event.set()
        # Unblock worker with sentinel
        try:
            self._queue.put_nowait(None)  # sentinel
        except Full:
            pass
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=6.0)
        self._worker = None
        # Drain any leftover sentinel so a subsequent start() gets a clean queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Empty:
                break
        self._stop_event.clear()

    # ------------------------------------------------------------------
    # Enqueue
    # ------------------------------------------------------------------

    def enqueue(self, event: ASREventWithSource) -> None:
        """Push an ASR event onto the translation queue.

        If the queue is full, the oldest pending event is discarded first
        (tail-drop) to prevent unbounded latency buildup.
        """
        try:
            self._queue.put_nowait(event)
        except Full:
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            try:
                self._queue.put_nowait(event)
            except Full:
                pass
            self._overflow_count += 1

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> TranslationDispatcherStats:
        return TranslationDispatcherStats(
            overflow_count=self._overflow_count,
            queue_size=self._queue.qsize(),
            worker_alive=bool(self._worker and self._worker.is_alive()),
        )

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                event = self._queue.get(timeout=0.3)
            except Empty:
                continue
            if event is None:  # sentinel
                break
            self._process(event)

    def _process(self, event: ASREventWithSource) -> None:
        try:
            correct_event = getattr(self._translator, "correct_asr_event", lambda v: v)
            corrected = correct_event(event)
            translated = self._translator.process(corrected)
            if not translated:
                skip_reason = ""
                try:
                    skip_reason = str(
                        getattr(self._translator, "last_skip_reason", lambda _s: "")(event.source) or ""
                    )
                except Exception:  # noqa: BLE001
                    pass
                if self._on_skipped:
                    self._on_skipped(event.source, skip_reason)
                return
            self._on_translated(translated)
        except Exception:  # noqa: BLE001
            pass  # translation errors must not crash the worker


__all__ = ["TranslationDispatcher", "TranslationDispatcherStats"]
