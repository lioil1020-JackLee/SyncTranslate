from __future__ import annotations

import math
import unittest

import numpy as np

from app.infra.asr.speaker_diarizer import OnlineSpeakerDiarizer
from app.application.transcript_service import TranscriptService
from app.ui.main_window import MainWindow


def _sine_wave(freq_hz: float, *, seconds: float = 1.3, sample_rate: int = 16000) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / float(sample_rate)
    return (0.12 * np.sin(2.0 * math.pi * freq_hz * t)).astype(np.float32)


class SpeakerDiarizationTests(unittest.TestCase):
    def test_online_speaker_diarizer_reuses_label_for_similar_voice(self) -> None:
        diarizer = OnlineSpeakerDiarizer(enabled=True, min_audio_ms=900, max_speakers=3, similarity_threshold=0.8)

        speaker_a_first = diarizer.assign(audio=_sine_wave(130.0), sample_rate=16000, now_ms=1000)
        speaker_a_second = diarizer.assign(audio=_sine_wave(136.0), sample_rate=16000, now_ms=2000)

        self.assertEqual(speaker_a_first, "Speaker A")
        self.assertEqual(speaker_a_second, "Speaker A")

    def test_online_speaker_diarizer_creates_new_label_for_different_voice(self) -> None:
        diarizer = OnlineSpeakerDiarizer(enabled=True, min_audio_ms=900, max_speakers=3, similarity_threshold=0.84)

        speaker_a = diarizer.assign(audio=_sine_wave(125.0), sample_rate=16000, now_ms=1000)
        first_probe = diarizer.assign(audio=_sine_wave(235.0), sample_rate=16000, now_ms=2000)
        speaker_b = diarizer.assign(audio=_sine_wave(235.0), sample_rate=16000, now_ms=3000)

        self.assertEqual(speaker_a, "Speaker A")
        self.assertEqual(first_probe, "Speaker A")
        self.assertEqual(speaker_b, "Speaker B")

    def test_online_speaker_diarizer_requires_repeat_before_switching_existing_speaker(self) -> None:
        diarizer = OnlineSpeakerDiarizer(enabled=True, min_audio_ms=900, max_speakers=2, similarity_threshold=0.84)

        speaker_a = diarizer.assign(audio=_sine_wave(125.0), sample_rate=16000, now_ms=1000)
        diarizer.assign(audio=_sine_wave(235.0), sample_rate=16000, now_ms=2000)
        speaker_b = diarizer.assign(audio=_sine_wave(235.0), sample_rate=16000, now_ms=3000)
        first_return = diarizer.assign(audio=_sine_wave(125.0), sample_rate=16000, now_ms=4000)
        second_return = diarizer.assign(audio=_sine_wave(125.0), sample_rate=16000, now_ms=5000)

        self.assertEqual(speaker_a, "Speaker A")
        self.assertEqual(speaker_b, "Speaker B")
        self.assertEqual(first_return, "Speaker B")
        self.assertEqual(second_return, "Speaker A")

    def test_short_audio_does_not_force_unreliable_speaker_label(self) -> None:
        diarizer = OnlineSpeakerDiarizer(enabled=True, min_audio_ms=900, max_speakers=3)
        short = _sine_wave(180.0, seconds=0.35)

        self.assertEqual(diarizer.assign(audio=short, sample_rate=16000, now_ms=1000), "")

    def test_transcript_service_does_not_merge_adjacent_finals_with_different_speakers(self) -> None:
        svc = TranscriptService()
        svc.upsert_event(
            source="meeting_original",
            channel="meeting_original",
            kind="original",
            text="先說一句",
            is_final=True,
            speaker_label="Speaker A",
        )
        svc.upsert_event(
            source="meeting_original",
            channel="meeting_original",
            kind="original",
            text="再接一句",
            is_final=True,
            speaker_label="Speaker B",
        )

        items = svc.latest("meeting_original", limit=10)
        self.assertEqual(len(items), 2)

    def test_transcript_line_format_includes_speaker_label(self) -> None:
        line = MainWindow._format_transcript_line("測試內容", True, "Speaker A")
        self.assertEqual(line, "[final] Speaker A: 測試內容")


if __name__ == "__main__":
    unittest.main()
