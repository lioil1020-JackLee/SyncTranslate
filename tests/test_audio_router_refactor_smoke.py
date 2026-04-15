"""Smoke tests for AudioRouter refactor sub-modules (Phase 4)."""
from __future__ import annotations

import pytest

from app.application.translation_dispatcher import TranslationDispatcher, TranslationDispatcherStats
from app.application.tts_dispatcher import TTSDispatcher, TTSEnqueueRequest
from app.application.pipeline_metrics import PipelineMetricsCollector
from app.ui.controllers.session_action_controller import SessionActionController
from app.ui.controllers.live_caption_refresh_controller import LiveCaptionRefreshController
from app.ui.controllers.config_hot_apply_controller import ConfigHotApplyController
from app.ui.controllers.healthcheck_controller import HealthcheckController, HealthcheckStatus


# ---------------------------------------------------------------------------
# PipelineMetricsCollector
# ---------------------------------------------------------------------------

class TestPipelineMetricsCollector:
    def test_records_asr_partial(self):
        col = PipelineMetricsCollector()
        col.record_asr_partial(source="remote", utterance_id="u1")
        assert col.in_flight_count() == 1

    def test_finalize_moves_to_recent(self):
        col = PipelineMetricsCollector()
        col.record_asr_partial(source="remote", utterance_id="u2")
        col.record_asr_final(source="remote", utterance_id="u2")
        col.finalize_utterance(source="remote", utterance_id="u2")
        assert col.in_flight_count() == 0
        recent = col.recent_latencies()
        assert len(recent) == 1
        assert recent[0]["utterance_id"] == "u2"
        assert recent[0]["asr_final_ms"] is not None

    def test_playback_retires_utterance(self):
        col = PipelineMetricsCollector()
        col.record_asr_partial(source="local", utterance_id="u3")
        col.record_tts_playback(source="local", utterance_id="u3")
        assert col.in_flight_count() == 0
        assert len(col.recent_latencies()) == 1

    def test_reset(self):
        col = PipelineMetricsCollector()
        col.record_asr_partial(source="local", utterance_id="u4")
        col.reset()
        assert col.in_flight_count() == 0
        assert len(col.recent_latencies()) == 0

    def test_window_size_respected(self):
        col = PipelineMetricsCollector(window_size=4)  # min window is 4
        for i in range(6):
            uid = f"u{i}"
            col.record_asr_partial(source="remote", utterance_id=uid)
            col.finalize_utterance(source="remote", utterance_id=uid)
        assert len(col.recent_latencies()) == 4

    def test_unknown_utterance_id_no_crash(self):
        col = PipelineMetricsCollector()
        col.record_tts_playback(source="remote", utterance_id="never-seen")
        # Should not crash, recent should be empty
        assert col.in_flight_count() == 0


# ---------------------------------------------------------------------------
# TTSDispatcher
# ---------------------------------------------------------------------------

class TestTTSDispatcher:
    def _make_mock_tts(self, mode: str = "tts"):
        from unittest.mock import MagicMock
        tts = MagicMock()
        tts.output_mode.return_value = mode
        return tts

    def test_enqueues_when_mode_is_tts(self):
        tts = self._make_mock_tts("tts")
        dispatcher = TTSDispatcher(tts_manager=tts)
        req = TTSEnqueueRequest(
            channel="remote", text="Hello world",
            utterance_id="u1", revision=1,
            is_final=True, is_stable_partial=False, is_early_final=False,
        )
        result = dispatcher.maybe_enqueue(req)
        assert result is True
        tts.enqueue.assert_called_once()
        assert dispatcher.stats()["enqueue_count"] == 1

    def test_skips_when_mode_is_passthrough(self):
        tts = self._make_mock_tts("passthrough")
        dispatcher = TTSDispatcher(tts_manager=tts)
        req = TTSEnqueueRequest(
            channel="remote", text="Hello",
            utterance_id="u1", revision=1,
            is_final=True, is_stable_partial=False, is_early_final=False,
        )
        result = dispatcher.maybe_enqueue(req)
        assert result is False
        tts.enqueue.assert_not_called()
        assert dispatcher.stats()["skipped_count"] == 1

    def test_skips_empty_text(self):
        tts = self._make_mock_tts("tts")
        dispatcher = TTSDispatcher(tts_manager=tts)
        req = TTSEnqueueRequest(
            channel="remote", text="   ",
            utterance_id="u1", revision=1,
            is_final=True, is_stable_partial=False, is_early_final=False,
        )
        result = dispatcher.maybe_enqueue(req)
        assert result is False

    def test_on_tts_request_called(self):
        tts = self._make_mock_tts("tts")
        requests = []
        dispatcher = TTSDispatcher(tts_manager=tts, on_tts_request=lambda c, t: requests.append((c, t)))
        req = TTSEnqueueRequest(
            channel="remote", text="Hello",
            utterance_id="u1", revision=1,
            is_final=True, is_stable_partial=False, is_early_final=False,
        )
        dispatcher.maybe_enqueue(req)
        assert requests == [("remote", "Hello")]


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
# SessionActionController
# ---------------------------------------------------------------------------

class TestSessionActionController:
    def test_request_start_calls_on_start(self):
        import time
        called = []

        def on_start(**kwargs):
            called.append(kwargs)

        ctrl = SessionActionController(on_start=on_start, on_stop=lambda: None)
        ctrl.request_start(route="r", sample_rate=16000)
        time.sleep(0.2)
        assert len(called) == 1
        assert ctrl.is_running

    def test_request_stop_calls_on_stop(self):
        import time
        stopped = []
        ctrl = SessionActionController(on_start=lambda **k: None, on_stop=lambda: stopped.append(True))
        ctrl.request_start(route="r", sample_rate=16000)
        time.sleep(0.2)
        ctrl.request_stop()
        time.sleep(0.2)
        assert stopped

    def test_status_changed_callback(self):
        import time
        statuses = []
        ctrl = SessionActionController(
            on_start=lambda **k: None,
            on_stop=lambda: None,
            on_status_changed=statuses.append,
        )
        ctrl.request_start(route="r", sample_rate=16000)
        time.sleep(0.2)
        assert len(statuses) >= 1

    def test_error_callback(self):
        import time
        errors = []

        def bad_start(**k):
            raise RuntimeError("boom")

        ctrl = SessionActionController(
            on_start=bad_start,
            on_stop=lambda: None,
            on_error=errors.append,
        )
        ctrl.request_start(route="r", sample_rate=16000)
        time.sleep(0.2)
        assert errors


# ---------------------------------------------------------------------------
# ConfigHotApplyController
# ---------------------------------------------------------------------------

class TestConfigHotApplyController:
    def test_applies_after_debounce(self):
        import time
        applied = []
        ctrl = ConfigHotApplyController(on_apply=lambda: applied.append(True), debounce_ms=10)
        ctrl.mark_pending()
        time.sleep(0.05)
        ctrl.tick()
        assert applied

    def test_does_not_apply_before_debounce(self):
        applied = []
        ctrl = ConfigHotApplyController(on_apply=lambda: applied.append(True), debounce_ms=10000)
        ctrl.mark_pending()
        ctrl.tick()
        assert not applied

    def test_suspend_prevents_apply(self):
        import time
        applied = []
        ctrl = ConfigHotApplyController(on_apply=lambda: applied.append(True), debounce_ms=0)
        ctrl.suspend()
        ctrl.mark_pending()
        time.sleep(0.05)
        ctrl.tick()
        assert not applied

    def test_resume_allows_apply(self):
        import time
        applied = []
        ctrl = ConfigHotApplyController(on_apply=lambda: applied.append(True), debounce_ms=0)
        ctrl.suspend()
        ctrl.mark_pending()
        ctrl.resume()
        time.sleep(0.05)
        ctrl.tick()
        assert applied

    def test_apply_immediately(self):
        applied = []
        ctrl = ConfigHotApplyController(on_apply=lambda: applied.append(True))
        ctrl.apply_immediately()
        assert applied


# ---------------------------------------------------------------------------
# LiveCaptionRefreshController
# ---------------------------------------------------------------------------

class TestLiveCaptionRefreshController:
    def test_triggers_update_on_change(self):
        remote_updates = []
        local_updates = []

        lines_store = {
            "local": ["hello"],
            "local_t": [],
            "remote": ["world"],
            "remote_t": [],
        }

        ctrl = LiveCaptionRefreshController(
            get_transcript_lines=lambda: (
                lines_store["local"],
                lines_store["local_t"],
                lines_store["remote"],
                lines_store["remote_t"],
            ),
            on_remote_update=lambda l, t: remote_updates.append((l, t)),
            on_local_update=lambda l, t: local_updates.append((l, t)),
        )
        ctrl.tick()
        assert len(remote_updates) == 1
        assert len(local_updates) == 1

    def test_no_update_when_unchanged(self):
        remote_updates = []
        lines = ["hello"]
        ctrl = LiveCaptionRefreshController(
            get_transcript_lines=lambda: ([], [], lines, []),
            on_remote_update=lambda l, t: remote_updates.append((l, t)),
            on_local_update=lambda l, t: None,
        )
        ctrl.tick()
        ctrl.tick()  # second call should not trigger again
        assert len(remote_updates) == 1

    def test_reset_clears_cache(self):
        remote_updates = []
        lines = ["hello"]
        ctrl = LiveCaptionRefreshController(
            get_transcript_lines=lambda: ([], [], lines, []),
            on_remote_update=lambda l, t: remote_updates.append((l, t)),
            on_local_update=lambda l, t: None,
        )
        ctrl.tick()
        ctrl.reset()
        ctrl.tick()  # should trigger again after reset
        assert len(remote_updates) == 2


# ---------------------------------------------------------------------------
# HealthcheckController
# ---------------------------------------------------------------------------

class TestHealthcheckController:
    def _make_service(self, ok: bool = True):
        from unittest.mock import MagicMock
        svc = MagicMock()
        svc.is_running.return_value = False
        svc.result.return_value = {"ok": ok, "message": "done"}
        return svc

    def test_run_emits_running_status(self):
        statuses = []
        ctrl = HealthcheckController(
            healthcheck_service=self._make_service(),
            on_status_update=statuses.append,
        )
        ctrl.run()
        assert statuses[0].running is True

    def test_tick_completes(self):
        statuses = []
        svc = self._make_service(ok=True)
        ctrl = HealthcheckController(
            healthcheck_service=svc,
            on_status_update=statuses.append,
        )
        ctrl.run()
        ctrl.tick()
        assert statuses[-1].ok is True
        assert statuses[-1].running is False

    def test_tick_failed_healthcheck(self):
        statuses = []
        svc = self._make_service(ok=False)
        ctrl = HealthcheckController(
            healthcheck_service=svc,
            on_status_update=statuses.append,
        )
        ctrl.run()
        ctrl.tick()
        assert statuses[-1].ok is False
