from __future__ import annotations

import unittest

from app.domain.runtime_state import StateManager


class RuntimeStateTests(unittest.TestCase):
    def test_remote_tts_no_longer_disables_remote_asr(self) -> None:
        state = StateManager()
        state.start_session()
        self.assertTrue(state.can_accept_asr("remote"))

        state.on_tts_start("remote")
        self.assertTrue(state.can_accept_asr("remote"))

        state.on_tts_end("remote", resume_delay_ms=0)
        state.tick()
        self.assertTrue(state.can_accept_asr("remote"))

    def test_local_tts_no_longer_blocks_local_asr(self) -> None:
        state = StateManager()
        state.start_session()
        state.on_tts_start("local")
        self.assertTrue(state.can_accept_asr("local"))

    def test_local_tts_echo_guard_blocks_local_asr_when_enabled(self) -> None:
        state = StateManager(local_echo_guard_enabled=True, local_resume_delay_ms=0)
        state.start_session()
        state.on_tts_start("local")
        self.assertFalse(state.can_accept_asr("local"))
        state.on_tts_end("local", resume_delay_ms=0)
        state.tick()
        self.assertTrue(state.can_accept_asr("local"))


if __name__ == "__main__":
    unittest.main()
