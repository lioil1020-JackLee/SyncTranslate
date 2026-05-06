from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from app.application.transcript_service import TranscriptService


class TranscriptServiceTests(unittest.TestCase):
    def test_upsert_replaces_same_utterance_with_newer_revision(self) -> None:
        svc = TranscriptService()
        svc.upsert_event(
            source="meeting_original",
            channel="meeting_original",
            kind="original",
            text="hello",
            is_final=False,
            utterance_id="u1",
            revision=1,
        )
        svc.upsert_event(
            source="meeting_original",
            channel="meeting_original",
            kind="original",
            text="hello world",
            is_final=False,
            utterance_id="u1",
            revision=2,
        )

        items = svc.latest("meeting_original", limit=10)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].text, "hello world")
        self.assertEqual(items[0].revision, 2)

    def test_partial_is_replaced_by_following_final_without_utterance(self) -> None:
        svc = TranscriptService()
        svc.append(source="meeting_original", text="part", is_final=False)
        svc.append(source="meeting_original", text="final text", is_final=True)

        items = svc.latest("meeting_original", limit=10)
        self.assertEqual(len(items), 1)
        self.assertTrue(items[0].is_final)
        self.assertEqual(items[0].text, "final text")

    def test_final_event_is_not_downgraded_by_later_partial_same_utterance(self) -> None:
        svc = TranscriptService()
        svc.upsert_event(
            source="meeting_original",
            channel="meeting_original",
            kind="original",
            text="done",
            is_final=True,
            utterance_id="u2",
            revision=2,
        )
        svc.upsert_event(
            source="meeting_original",
            channel="meeting_original",
            kind="original",
            text="older partial",
            is_final=False,
            utterance_id="u2",
            revision=3,
        )

        items = svc.latest("meeting_original", limit=10)
        self.assertEqual(len(items), 1)
        self.assertTrue(items[0].is_final)
        self.assertEqual(items[0].text, "done")

    def test_adjacent_short_final_fragments_are_merged(self) -> None:
        svc = TranscriptService()
        now = datetime.now()
        svc.upsert_event(
            source="meeting_original",
            channel="meeting_original",
            kind="original",
            text="不用理會待會兒入水",
            is_final=True,
            created_at=now,
        )
        svc.upsert_event(
            source="meeting_original",
            channel="meeting_original",
            kind="original",
            text="甩掉他們",
            is_final=True,
            created_at=now + timedelta(milliseconds=320),
        )

        items = svc.latest("meeting_original", limit=10)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].text, "不用理會待會兒入水甩掉他們")

    def test_sentence_finished_final_is_not_merged_with_next_one(self) -> None:
        svc = TranscriptService()
        now = datetime.now()
        svc.upsert_event(
            source="meeting_original",
            channel="meeting_original",
            kind="original",
            text="大家休整一下。",
            is_final=True,
            created_at=now,
        )
        svc.upsert_event(
            source="meeting_original",
            channel="meeting_original",
            kind="original",
            text="天亮我們繼續趕路",
            is_final=True,
            created_at=now + timedelta(milliseconds=300),
        )

        items = svc.latest("meeting_original", limit=10)
        self.assertEqual(len(items), 2)

    def test_finals_with_different_utterance_ids_are_not_merged(self) -> None:
        svc = TranscriptService()
        now = datetime.now()
        svc.upsert_event(
            source="meeting_original",
            channel="meeting_original",
            kind="original",
            text="first sentence",
            is_final=True,
            utterance_id="utt-1",
            created_at=now,
        )
        svc.upsert_event(
            source="meeting_original",
            channel="meeting_original",
            kind="original",
            text="second sentence",
            is_final=True,
            utterance_id="utt-2",
            created_at=now + timedelta(milliseconds=180),
        )

        items = svc.latest("meeting_original", limit=10)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].text, "first sentence")
        self.assertEqual(items[1].text, "second sentence")

    def test_overlapping_final_prefix_is_trimmed_from_next_item(self) -> None:
        svc = TranscriptService()
        now = datetime.now()
        svc.upsert_event(
            source="meeting_original",
            channel="meeting_original",
            kind="original",
            text="去年舉發違反道路交通管理事件一共有一千六百三十一萬",
            is_final=True,
            created_at=now,
        )
        svc.upsert_event(
            source="meeting_original",
            channel="meeting_original",
            kind="original",
            text="一千六百三十一萬九千五百四十六件舉發件數以違規停車最多",
            is_final=True,
            created_at=now + timedelta(milliseconds=1900),
        )

        items = svc.latest("meeting_original", limit=10)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[1].text, "九千五百四十六件舉發件數以違規停車最多")

    def test_fully_duplicated_final_is_dropped(self) -> None:
        svc = TranscriptService()
        now = datetime.now()
        text = "According to the latest traffic management statistics"
        svc.upsert_event(
            source="meeting_original",
            channel="meeting_original",
            kind="original",
            text=text,
            is_final=True,
            created_at=now,
        )
        svc.upsert_event(
            source="meeting_original",
            channel="meeting_original",
            kind="original",
            text=text,
            is_final=True,
            created_at=now + timedelta(milliseconds=2100),
        )

        items = svc.latest("meeting_original", limit=10)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].text, text)


if __name__ == "__main__":
    unittest.main()
