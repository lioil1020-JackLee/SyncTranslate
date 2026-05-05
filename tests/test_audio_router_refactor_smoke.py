"""Smoke tests for active AudioRouter-adjacent services."""
from __future__ import annotations

from app.application.translation_dispatcher import TranslationDispatcher, TranslationDispatcherStats


# ---------------------------------------------------------------------------
# TranslationDispatcher
# ---------------------------------------------------------------------------

class TestTranslationDispatcher:
    def _make_mock_translator(self, result=None):
        from unittest.mock import MagicMock
        t = MagicMock()
        t.process.return_value = result
        t.correct_asr_event.side_effect = lambda v: v
        return t

    def _make_event(self, source="remote", text="hello", is_final=True):
        from unittest.mock import MagicMock
        e = MagicMock()
        e.source = source
        e.text = text
        e.is_final = is_final
        e.utterance_id = "u1"
        e.revision = 1
        return e

    def test_start_stop(self):
        import time
        translator = self._make_mock_translator()
        d = TranslationDispatcher(
            translator_manager=translator,
            on_translated=lambda _r: None,
            queue_maxsize=4,
        )
        d.start()
        time.sleep(0.1)  # give thread time to start
        assert d.stats().worker_alive
        d.stop()

    def test_overflow_increments_counter(self):
        translator = self._make_mock_translator()
        d = TranslationDispatcher(
            translator_manager=translator,
            on_translated=lambda _r: None,
            queue_maxsize=1,
        )
        event = self._make_event()
        # Fill queue
        for _ in range(5):
            d.enqueue(event)
        assert d.stats().overflow_count > 0

    def test_translated_callback_called(self):
        import time
        results = []
        fake_result = object()
        translator = self._make_mock_translator(result=fake_result)
        d = TranslationDispatcher(
            translator_manager=translator,
            on_translated=results.append,
            queue_maxsize=4,
        )
        d.start()
        time.sleep(0.05)  # let worker start
        d.enqueue(self._make_event())
        time.sleep(0.5)
        d.stop()
        assert len(results) >= 1

    def test_skipped_callback_when_translation_none(self):
        import time
        skipped = []
        translator = self._make_mock_translator(result=None)
        # Patch last_skip_reason to be a proper callable
        translator.last_skip_reason = lambda _src: ""
        d = TranslationDispatcher(
            translator_manager=translator,
            on_translated=lambda _r: None,
            on_skipped=lambda src, reason: skipped.append(src),
            queue_maxsize=4,
        )
        d.start()
        time.sleep(0.05)  # let worker start
        d.enqueue(self._make_event())
        time.sleep(0.5)
        d.stop()
        assert skipped == ["remote"]


# ---------------------------------------------------------------------------
# ASRManagerV2 — Event fast-path (P2-4)
# ---------------------------------------------------------------------------

class TestASRManagerV2FastPath:
    """Verify that set_enabled / submit use the Event fast path correctly."""

    def _make_manager(self):
        from unittest.mock import MagicMock
        from app.infra.asr.manager_v2 import ASRManagerV2
        from app.infra.config.schema import AppConfig

        cfg = AppConfig()
        mgr = ASRManagerV2(cfg)
        return mgr

    def test_initially_both_channels_enabled(self):
        mgr = self._make_manager()
        assert mgr._enabled_events["local"].is_set()
        assert mgr._enabled_events["remote"].is_set()

    def test_set_enabled_false_clears_event(self):
        mgr = self._make_manager()
        mgr.set_enabled("local", False)
        assert not mgr._enabled_events["local"].is_set()
        assert not mgr.is_enabled("local")

    def test_set_enabled_true_sets_event(self):
        mgr = self._make_manager()
        mgr.set_enabled("local", False)
        mgr.set_enabled("local", True)
        assert mgr._enabled_events["local"].is_set()
        assert mgr.is_enabled("local")

    def test_remote_event_independent_of_local(self):
        mgr = self._make_manager()
        mgr.set_enabled("local", False)
        assert not mgr._enabled_events["local"].is_set()
        assert mgr._enabled_events["remote"].is_set()

