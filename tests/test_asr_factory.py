from __future__ import annotations

import unittest

from app.infra.asr.factory import create_asr_manager, normalize_asr_pipeline_mode
from app.infra.asr.manager_v2 import ASRManagerV2
from app.infra.asr.streaming_pipeline import ASRManager
from app.infra.config.schema import AppConfig


class AsrFactoryTests(unittest.TestCase):
    def test_normalize_pipeline_mode_defaults_to_legacy(self) -> None:
        self.assertEqual(normalize_asr_pipeline_mode(""), "legacy")
        self.assertEqual(normalize_asr_pipeline_mode("whisper_legacy"), "legacy")
        self.assertEqual(normalize_asr_pipeline_mode("v2"), "v2")

    def test_factory_builds_v2_manager_by_default(self) -> None:
        cfg = AppConfig()

        manager = create_asr_manager(cfg, pipeline_revision=3)

        self.assertIsInstance(manager, ASRManagerV2)

    def test_factory_builds_v2_manager_when_requested(self) -> None:
        cfg = AppConfig()
        cfg.runtime.asr_pipeline = "v2"

        manager = create_asr_manager(cfg, pipeline_revision=5)

        self.assertIsInstance(manager, ASRManagerV2)
        self.assertEqual(manager.pipeline_spec.pipeline_name, "asr_v2")
        self.assertEqual(manager.pipeline_spec.partial_backend.name, "faster_whisper_v2:partial")
        self.assertEqual(manager.pipeline_spec.endpointing.name, "silero_vad")
        stats = manager.stats()
        self.assertEqual(stats["local"]["pipeline_mode"], "v2")
        self.assertEqual(stats["remote"]["execution_mode"], "native_v2")
        self.assertEqual(stats["local"]["resolved_backend"], "faster_whisper_v2")

    def test_factory_pipeline_spec_keeps_faster_whisper_for_explicit_chinese(self) -> None:
        cfg = AppConfig()
        cfg.runtime.asr_pipeline = "v2"
        cfg.runtime.local_asr_language = "zh-TW"

        manager = create_asr_manager(cfg, pipeline_revision=6)

        self.assertEqual(manager.pipeline_spec.partial_backend.name, "faster_whisper_v2:partial")
        self.assertEqual(manager.pipeline_spec.endpointing.name, "silero_vad")


if __name__ == "__main__":
    unittest.main()
