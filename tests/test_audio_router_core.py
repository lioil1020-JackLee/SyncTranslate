from __future__ import annotations

from datetime import datetime
import unittest

from app.application.audio_router import AudioRouter
from app.application.transcript_service import TranscriptBuffer
from app.domain.runtime_state import StateManager
from app.infra.asr.contracts import ASREventWithSource
from app.domain.models import ErrorEvent


class _FakeInputManager:
    def stop_all(self) -> None:
        pass

    def remove_consumer(self, source: str, consumer) -> None:
        pass

    def stats(self) -> dict[str, object]:
        return {"local": {}, "remote": {}}


class _FakeAsrManager:
    def stop_all(self) -> None:
        pass

    def stats(self) -> dict[str, dict[str, object]]:
        return {"local": {}, "remote": {}}


class _FakeTranslatorManager:
    def translation_enabled(self, source: str | None = None) -> bool:
        return False

    def correct_asr_event(self, event: ASREventWithSource) -> ASREventWithSource:
        return event


class _FakeTtsManager:
    def __init__(self) -> None:
        self._mode = {"local": "subtitle_only", "remote": "subtitle_only"}
        self.enqueued: list[tuple[str, str]] = []

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def output_mode(self, channel: str) -> str:
        return self._mode.get(channel, "subtitle_only")

    def set_output_mode(self, channel: str, mode: str) -> None:
        self._mode[channel] = mode

    def enqueue(
        self,
        channel: str,
        text: str,
        *,
        utterance_id: str,
        revision: int,
        is_final: bool,
        is_stable_partial: bool = False,
        is_early_final: bool = False,
    ) -> None:
        self.enqueued.append((channel, text))

    def is_passthrough_enabled(self, channel: str) -> bool:
        return False

    def submit_passthrough(self, channel: str, chunk, sample_rate: float) -> None:
        pass

    def stats(self) -> dict[str, object]:
        return {
            "queue_depth": 0,
            "queue_depth_local": 0,
            "queue_depth_remote": 0,
            "drop_count_local": 0,
            "drop_count_remote": 0,
            "oldest_age_ms": 0.0,
        }


class AudioRouterCoreTests(unittest.TestCase):
    def _build_router(self, on_error=None):
        transcript = TranscriptBuffer()
        tts = _FakeTtsManager()
        router = AudioRouter(
            transcript_buffer=transcript,
            input_manager=_FakeInputManager(),  # type: ignore[arg-type]
            asr_manager=_FakeAsrManager(),  # type: ignore[arg-type]
            translator_manager=_FakeTranslatorManager(),  # type: ignore[arg-type]
            tts_manager=tts,  # type: ignore[arg-type]
            state_manager=StateManager(local_echo_guard_enabled=True),
            on_error=on_error,
        )
        return router, transcript, tts

    @staticmethod
    def _event(*, source: str = "remote", text: str = "hello", is_final: bool = True) -> ASREventWithSource:
        return ASREventWithSource(
            source=source,
            utterance_id="u-core",
            revision=1,
            pipeline_revision=1,
            config_fingerprint="fp",
            created_at=0.0,
            text=text,
            is_final=is_final,
            is_early_final=False,
            start_ms=0,
            end_ms=800,
            latency_ms=50,
            detected_language="en",
        )

    def test_handle_asr_event_no_translation_mirrors_to_translated_channel_and_tts(self) -> None:
        router, transcript, tts = self._build_router()
        tts.set_output_mode("local", "tts")
        event = self._event(source="remote", text="No translation", is_final=True)

        router._handle_asr_event_no_translation(
            event=event,
            translated_channel="meeting_translated",
            tts_channel="local",
        )

        translated_items = transcript.latest("meeting_translated", limit=5)
        self.assertGreaterEqual(len(translated_items), 1)
        self.assertEqual(translated_items[-1].text, "No translation")
        self.assertEqual(tts.enqueued, [("local", "No translation")])

    def test_maybe_store_transcript_respects_partial_display_policy(self) -> None:
        router, transcript, _ = self._build_router()
        router._partial_display_policy.should_display = lambda **kwargs: (False, False)  # type: ignore[method-assign]

        router._maybe_store_transcript(
            source="meeting_original",
            channel="meeting_original",
            kind="original",
            text="skip me",
            is_final=False,
            is_stable_partial=True,
            utterance_id="u-skip",
            revision=1,
            latency_ms=20,
            created_at=datetime.fromtimestamp(0),
            speaker_label="",
        )

        self.assertEqual(transcript.latest("meeting_original", limit=5), [])

    def test_on_asr_event_wraps_exceptions_and_calls_on_error(self) -> None:
        errors: list[ErrorEvent] = []
        router, _, _ = self._build_router(on_error=errors.append)
        router._handle_asr_event_payload = lambda _event: (_ for _ in ()).throw(RuntimeError("boom"))  # type: ignore[method-assign]

        router._on_asr_event(self._event(source="local", text="hello", is_final=True))

        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].code, "asr_event_failed")
        self.assertEqual(errors[0].module, "audio_router")


if __name__ == "__main__":
    unittest.main()
