from __future__ import annotations

import unittest

from app.application.audio_router import RouterStats
from app.application.session_service import SessionController, SessionState
from app.infra.config.schema import AudioRouteConfig


class _FakeAudioRouter:
    def __init__(self) -> None:
        self.fail_start = False
        self.fail_stop = False
        self.start_calls: list[tuple[str, int, int]] = []
        self.stop_calls = 0
        self._running = False

    def start(self, mode: str, routes: AudioRouteConfig, sample_rate: int, chunk_ms: int = 100) -> None:
        if self.fail_start:
            raise ValueError("start failed")
        self.start_calls.append((mode, sample_rate, chunk_ms))
        self._running = True

    def stop(self) -> None:
        self.stop_calls += 1
        if self.fail_stop:
            raise ValueError("stop failed")
        self._running = False

    def stats(self) -> RouterStats:
        return RouterStats(
            running=self._running,
            active_sources=["remote"] if self._running else [],
            state={
                "local_asr_enabled": True,
                "remote_asr_enabled": True,
                "local_tts_busy": False,
                "remote_tts_busy": False,
                "local_resume_in_ms": 0,
                "remote_resume_in_ms": 0,
            },
            capture={},
            asr={},
            tts={},
            latency=[],
            translation_overflow={"local": 0, "remote": 0},
        )


class SessionControllerTests(unittest.TestCase):
    def test_start_and_stop_success_path(self) -> None:
        router = _FakeAudioRouter()
        controller = SessionController(router)  # type: ignore[arg-type]
        routes = AudioRouteConfig(meeting_in="m", microphone_in="l", speaker_out="s", meeting_out="o")

        start_result = controller.start(routes, sample_rate=24000, chunk_ms=40, mode="meeting_to_local")
        self.assertTrue(start_result.ok)
        self.assertEqual(controller.current_state(), SessionState.RUNNING)
        self.assertEqual(router.start_calls[-1], ("meeting_to_local", 24000, 40))

        stop_result = controller.stop()
        self.assertTrue(stop_result.ok)
        self.assertEqual(controller.current_state(), SessionState.IDLE)
        self.assertIsNotNone(stop_result.payload)
        assert stop_result.payload is not None
        self.assertIn("stats_before_stop", stop_result.payload)
        self.assertIn("session_meta", stop_result.payload)

    def test_start_failure_sets_failed_state_and_invokes_stop_cleanup(self) -> None:
        router = _FakeAudioRouter()
        router.fail_start = True
        controller = SessionController(router)  # type: ignore[arg-type]
        routes = AudioRouteConfig()

        result = controller.start(routes, sample_rate=16000)
        self.assertFalse(result.ok)
        self.assertEqual(controller.current_state(), SessionState.FAILED)
        self.assertEqual(router.stop_calls, 1)
        self.assertIn("start failed", controller.last_failure())

    def test_stop_when_idle_returns_already_stopped(self) -> None:
        router = _FakeAudioRouter()
        controller = SessionController(router)  # type: ignore[arg-type]

        result = controller.stop()
        self.assertTrue(result.ok)
        self.assertEqual(result.message, "Session already stopped")
        self.assertEqual(controller.current_state(), SessionState.IDLE)

    def test_stop_failure_sets_failed_state(self) -> None:
        router = _FakeAudioRouter()
        controller = SessionController(router)  # type: ignore[arg-type]
        routes = AudioRouteConfig()
        controller.start(routes, sample_rate=24000)
        router.fail_stop = True

        result = controller.stop()
        self.assertFalse(result.ok)
        self.assertEqual(controller.current_state(), SessionState.FAILED)
        self.assertIn("Session stop failed", result.message)

    def test_running_start_result_reports_requested_mode(self) -> None:
        router = _FakeAudioRouter()
        controller = SessionController(router)  # type: ignore[arg-type]
        routes = AudioRouteConfig()
        controller.start(routes, sample_rate=24000, mode="local_to_meeting")

        result = controller.start(routes, sample_rate=24000, mode="local_to_meeting")

        self.assertTrue(result.ok)
        self.assertEqual(result.payload, {"mode": "local_to_meeting"})


if __name__ == "__main__":
    unittest.main()
