from __future__ import annotations

import unittest

from app.transcript_buffer import TranscriptBuffer


class TranscriptBufferTests(unittest.TestCase):
    def test_upsert_by_utterance_and_revision(self) -> None:
        buf = TranscriptBuffer(max_items=20)
        buf.append("meeting_translated", "hello", False, utterance_id="u1", revision=1)
        buf.append("meeting_translated", "hello world", False, utterance_id="u1", revision=2)
        items = buf.latest("meeting_translated", limit=5)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].text, "hello world")

    def test_ignore_older_revision(self) -> None:
        buf = TranscriptBuffer(max_items=20)
        buf.append("meeting_translated", "new", False, utterance_id="u1", revision=3)
        buf.append("meeting_translated", "old", False, utterance_id="u1", revision=2)
        items = buf.latest("meeting_translated", limit=5)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].text, "new")

    def test_final_blocks_later_partial_for_same_utterance(self) -> None:
        buf = TranscriptBuffer(max_items=20)
        buf.append("meeting_translated", "final", True, utterance_id="u1", revision=4)
        buf.append("meeting_translated", "partial backflow", False, utterance_id="u1", revision=5)
        items = buf.latest("meeting_translated", limit=5)
        self.assertEqual(len(items), 1)
        self.assertTrue(items[0].is_final)
        self.assertEqual(items[0].text, "final")


if __name__ == "__main__":
    unittest.main()
