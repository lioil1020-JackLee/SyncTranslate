from __future__ import annotations

import unittest

from app.settings import migrate_legacy_config


class SettingsMigrationTests(unittest.TestCase):
    def test_runtime_new_fields_are_present(self) -> None:
        migrated = migrate_legacy_config({"model": "legacy"})
        runtime = migrated["runtime"]
        self.assertIn("translation_exact_cache_size", runtime)
        self.assertIn("translation_prefix_min_delta_chars", runtime)
        self.assertIn("tts_cancel_pending_on_new_final", runtime)
        self.assertIn("tts_drop_backlog_threshold", runtime)

    def test_llm_profile_fields_are_present(self) -> None:
        migrated = migrate_legacy_config({"model": "legacy"})
        llm = migrated["llm"]
        self.assertIn("profiles", llm)
        self.assertIn("caption_profile", llm)
        self.assertIn("speech_profile", llm)
        self.assertIn("live_caption_fast", llm["profiles"])


if __name__ == "__main__":
    unittest.main()
