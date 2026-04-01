from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import yaml

from app.infra.config.settings_store import (
    _normalize_external_config_keys,
    _present_external_config_keys,
    load_config,
    save_config,
    is_legacy_config,
    migrate_legacy_config,
)
from app.infra.config.schema import AppConfig


class ConfigMigrationTests(unittest.TestCase):
    def test_runtime_defaults_prefer_auto_asr_language_mode(self) -> None:
        cfg = AppConfig()
        self.assertEqual(cfg.runtime.asr_language_mode, "auto")

    def test_default_remote_asr_profile_matches_local_asr_defaults(self) -> None:
        cfg = AppConfig()
        self.assertEqual(cfg.asr_channels.remote.beam_size, 1)
        self.assertEqual(cfg.asr_channels.local.beam_size, 1)
        self.assertEqual(cfg.asr_channels.remote.streaming.partial_history_seconds, 2)
        self.assertEqual(cfg.asr_channels.local.streaming.partial_history_seconds, 2)
        self.assertTrue(cfg.asr_channels.remote.condition_on_previous_text)
        self.assertAlmostEqual(cfg.asr_channels.remote.vad.rms_threshold, 0.02, places=3)
        self.assertTrue(cfg.asr_channels.local.condition_on_previous_text)
        self.assertAlmostEqual(cfg.asr_channels.local.vad.rms_threshold, 0.02, places=3)

    def test_is_legacy_when_missing_required_sections(self) -> None:
        self.assertTrue(is_legacy_config({"audio": {}}))

    def test_migrate_legacy_maps_core_fields(self) -> None:
        legacy = {
            "session_mode": "remote_only",
            "audio": {
                "remote_in": "REMOTE_IN",
                "local_mic_in": "LOCAL_IN",
                "local_tts_out": "LOCAL_OUT",
                "meeting_tts_out": "REMOTE_OUT",
            },
            "openai": {
                "translate_model": "legacy-model",
                "base_url": "http://127.0.0.1:1234",
            },
            "runtime": {
                "sample_rate": 16000,
                "chunk_ms": 30,
            },
            "health_last_success": {"asr": "ok"},
        }

        migrated = migrate_legacy_config(legacy)
        self.assertEqual(migrated["direction"]["mode"], "bidirectional")
        self.assertEqual(migrated["audio"]["meeting_in"], "REMOTE_IN")
        self.assertEqual(migrated["audio"]["microphone_in"], "LOCAL_IN")
        self.assertEqual(migrated["audio"]["speaker_out"], "LOCAL_OUT")
        self.assertEqual(migrated["audio"]["meeting_out"], "REMOTE_OUT")
        self.assertEqual(migrated["runtime"]["sample_rate"], 16000)
        self.assertEqual(migrated["runtime"]["chunk_ms"], 30)
        self.assertEqual(migrated["runtime"]["config_schema_version"], 5)
        self.assertIn("local", migrated["asr_channels"])
        self.assertIn("remote", migrated["llm_channels"])

    def test_migrate_legacy_keeps_channel_defaults_language_neutral(self) -> None:
        legacy = {
            "session_mode": "bidirectional",
            "audio": {},
            "llm": {
                "max_output_tokens": 111,
                "stop_tokens": "</target>,TRANSLATE:",
            },
            "asr": {
                "beam_size": 2,
                "no_speech_threshold": 0.33,
                "streaming": {"soft_final_audio_ms": 5100},
            },
            "runtime": {},
        }

        migrated = migrate_legacy_config(legacy)

        self.assertEqual(migrated["llm_channels"]["local"]["max_output_tokens"], 111)
        self.assertEqual(migrated["llm_channels"]["remote"]["max_output_tokens"], 111)
        self.assertEqual(migrated["llm_channels"]["local"]["stop_tokens"], "</target>,TRANSLATE:")
        self.assertEqual(migrated["llm_channels"]["remote"]["stop_tokens"], "</target>,TRANSLATE:")

        self.assertEqual(migrated["asr_channels"]["local"]["no_speech_threshold"], 0.33)
        self.assertEqual(migrated["asr_channels"]["remote"]["no_speech_threshold"], 0.33)
        self.assertEqual(migrated["asr_channels"]["local"]["streaming"]["soft_final_audio_ms"], 5100)
        self.assertEqual(migrated["asr_channels"]["remote"]["streaming"]["soft_final_audio_ms"], 5100)

    def test_normalize_external_alias_keys(self) -> None:
        raw = {
            "asr_channels": {"chinese": {"model": "a"}, "english": {"model": "b"}},
            "llm_channels": {"zh_to_en": {"model": "x"}, "en_to_zh": {"model": "y"}},
            "tts_channels": {"chinese": {"voice_name": "zh"}, "english": {"voice_name": "en"}},
            "chinese_tts": {"voice_name": "zh-voice"},
            "english_tts": {"voice_name": "en-voice"},
            "runtime": {
                "asr_queue_maxsize_chinese": 11,
                "asr_queue_maxsize_english": 12,
                "llm_queue_maxsize_zh_to_en": 21,
                "llm_queue_maxsize_en_to_zh": 22,
                "tts_queue_maxsize_chinese": 31,
                "tts_queue_maxsize_english": 32,
                "remote_tts_enabled": True,
                "local_tts_enabled": False,
            },
        }

        normalized = _normalize_external_config_keys(raw)
        self.assertIn("local", normalized["asr_channels"])
        self.assertIn("remote", normalized["asr_channels"])
        self.assertIn("local", normalized["llm_channels"])
        self.assertIn("remote", normalized["llm_channels"])
        self.assertIn("local", normalized["tts_channels"])
        self.assertIn("remote", normalized["tts_channels"])
        self.assertEqual(normalized["runtime"]["asr_queue_maxsize_local"], 11)
        self.assertEqual(normalized["runtime"]["asr_queue_maxsize_remote"], 12)
        self.assertEqual(normalized["runtime"]["llm_queue_maxsize_local"], 21)
        self.assertEqual(normalized["runtime"]["llm_queue_maxsize_remote"], 22)
        self.assertEqual(normalized["runtime"]["tts_queue_maxsize_local"], 31)
        self.assertEqual(normalized["runtime"]["tts_queue_maxsize_remote"], 32)
        self.assertEqual(normalized["runtime"]["tts_output_mode"], "tts")
        self.assertTrue(normalized["runtime"]["remote_translation_enabled"])
        self.assertTrue(normalized["runtime"]["local_translation_enabled"])
        self.assertEqual(normalized["runtime"]["asr_language_mode"], "auto")

    def test_present_external_keeps_canonical_and_drops_aliases(self) -> None:
        raw = {
            "asr_channels": {
                "local": {"model": "l"},
                "remote": {"model": "r"},
                "chinese": {"model": "old-c"},
                "english": {"model": "old-e"},
            },
            "llm_channels": {
                "local": {"model": "l"},
                "remote": {"model": "r"},
                "zh_to_en": {"model": "old-z2e"},
                "en_to_zh": {"model": "old-e2z"},
            },
            "tts_channels": {
                "local": {"voice_name": "l"},
                "remote": {"voice_name": "r"},
                "chinese": {"voice_name": "old-c"},
                "english": {"voice_name": "old-e"},
            },
            "meeting_tts": {"voice_name": "meeting"},
            "local_tts": {"voice_name": "local"},
            "chinese_tts": {"voice_name": "old-c"},
            "english_tts": {"voice_name": "old-e"},
            "runtime": {
                "asr_queue_maxsize_local": 10,
                "asr_queue_maxsize_remote": 20,
                "llm_queue_maxsize_local": 30,
                "llm_queue_maxsize_remote": 40,
                "tts_queue_maxsize_local": 50,
                "tts_queue_maxsize_remote": 60,
                "asr_queue_maxsize_chinese": 11,
                "asr_queue_maxsize_english": 21,
                "llm_queue_maxsize_zh_to_en": 31,
                "llm_queue_maxsize_en_to_zh": 41,
                "tts_queue_maxsize_chinese": 51,
                "tts_queue_maxsize_english": 61,
            },
        }

        presented = _present_external_config_keys(raw)

        self.assertIn("local", presented["asr_channels"])
        self.assertIn("remote", presented["asr_channels"])
        self.assertNotIn("chinese", presented["asr_channels"])
        self.assertNotIn("english", presented["asr_channels"])

        self.assertIn("local", presented["llm_channels"])
        self.assertIn("remote", presented["llm_channels"])
        self.assertNotIn("zh_to_en", presented["llm_channels"])
        self.assertNotIn("en_to_zh", presented["llm_channels"])

        self.assertIn("local", presented["tts_channels"])
        self.assertIn("remote", presented["tts_channels"])
        self.assertNotIn("chinese", presented["tts_channels"])
        self.assertNotIn("english", presented["tts_channels"])

        self.assertIn("meeting_tts", presented)
        self.assertIn("local_tts", presented)
        self.assertNotIn("chinese_tts", presented)
        self.assertNotIn("english_tts", presented)

        self.assertEqual(presented["runtime"]["asr_queue_maxsize_local"], 10)
        self.assertEqual(presented["runtime"]["asr_queue_maxsize_remote"], 20)
        self.assertEqual(presented["runtime"]["llm_queue_maxsize_local"], 30)
        self.assertEqual(presented["runtime"]["llm_queue_maxsize_remote"], 40)
        self.assertEqual(presented["runtime"]["tts_queue_maxsize_local"], 50)
        self.assertEqual(presented["runtime"]["tts_queue_maxsize_remote"], 60)
        self.assertNotIn("asr_queue_maxsize_chinese", presented["runtime"])
        self.assertNotIn("asr_queue_maxsize_english", presented["runtime"])
        self.assertNotIn("llm_queue_maxsize_zh_to_en", presented["runtime"])
        self.assertNotIn("llm_queue_maxsize_en_to_zh", presented["runtime"])
        self.assertNotIn("tts_queue_maxsize_chinese", presented["runtime"])
        self.assertNotIn("tts_queue_maxsize_english", presented["runtime"])

    def test_save_config_writes_canonical_keys_only(self) -> None:
        cfg = AppConfig()
        cfg.asr_channels.local.model = "local-asr"
        cfg.asr_channels.remote.model = "remote-asr"
        cfg.llm_channels.local.model = "local-llm"
        cfg.llm_channels.remote.model = "remote-llm"
        cfg.runtime.asr_queue_maxsize_local = 111
        cfg.runtime.asr_queue_maxsize_remote = 222

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            save_config(cfg, path)
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))

        language = payload.get("language", {})
        asr_channels = payload.get("asr_channels", {})
        llm_channels = payload.get("llm_channels", {})
        tts_channels = payload.get("tts_channels", {})
        runtime = payload.get("runtime", {})

        self.assertNotIn("meeting_source", language)
        self.assertNotIn("local_source", language)
        self.assertIn("remote_translation_target", language)
        self.assertIn("local_translation_target", language)

        self.assertIn("local", asr_channels)
        self.assertIn("remote", asr_channels)
        self.assertNotIn("chinese", asr_channels)
        self.assertNotIn("english", asr_channels)

        self.assertIn("local", llm_channels)
        self.assertIn("remote", llm_channels)
        self.assertNotIn("zh_to_en", llm_channels)
        self.assertNotIn("en_to_zh", llm_channels)

        self.assertIn("local", tts_channels)
        self.assertIn("remote", tts_channels)
        self.assertNotIn("chinese", tts_channels)
        self.assertNotIn("english", tts_channels)

        self.assertIn("meeting_tts", payload)
        self.assertIn("local_tts", payload)
        self.assertNotIn("chinese_tts", payload)
        self.assertNotIn("english_tts", payload)

        self.assertEqual(runtime.get("asr_queue_maxsize_local"), 111)
        self.assertEqual(runtime.get("asr_queue_maxsize_remote"), 222)
        self.assertNotIn("asr_queue_maxsize_chinese", runtime)
        self.assertNotIn("asr_queue_maxsize_english", runtime)

    def test_load_config_creates_missing_file_with_canonical_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fresh_config.yaml"
            missing_fallback = Path(tmp) / "missing_fallback.yaml"
            cfg = load_config(path, missing_fallback)
            self.assertIsNotNone(cfg)
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))

        asr_channels = payload.get("asr_channels", {})
        llm_channels = payload.get("llm_channels", {})
        runtime = payload.get("runtime", {})

        self.assertIn("local", asr_channels)
        self.assertIn("remote", asr_channels)
        self.assertNotIn("chinese", asr_channels)
        self.assertNotIn("english", asr_channels)

        self.assertIn("local", llm_channels)
        self.assertIn("remote", llm_channels)
        self.assertNotIn("zh_to_en", llm_channels)
        self.assertNotIn("en_to_zh", llm_channels)

        self.assertNotIn("asr_queue_maxsize_chinese", runtime)
        self.assertNotIn("asr_queue_maxsize_english", runtime)
        self.assertNotIn("llm_queue_maxsize_zh_to_en", runtime)
        self.assertNotIn("llm_queue_maxsize_en_to_zh", runtime)

    def test_load_config_uses_fallback_when_primary_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            primary = Path(tmp) / "missing_config.yaml"
            fallback = Path(tmp) / "config.example.yaml"
            fallback_payload = {
                "audio": {},
                "direction": {"mode": "bidirectional"},
                "language": {
                    "remote_translation_target": "de",
                    "local_translation_target": "it",
                },
                "asr": {},
                "llm": {},
                "tts": {},
                "runtime": {},
            }
            fallback.write_text(yaml.safe_dump(fallback_payload, sort_keys=False), encoding="utf-8")

            cfg = load_config(primary, fallback)

            self.assertFalse(primary.exists())
            self.assertEqual(cfg.language.meeting_target, "de")
            self.assertEqual(cfg.language.local_target, "it")

    def test_five_language_round_trip_and_runtime_modes(self) -> None:
        cfg = AppConfig()
        cfg.language.meeting_target = "ko"
        cfg.language.local_target = "zh-TW"
        cfg.tts.style_preset = "broadcast_clear"
        cfg.runtime.remote_translation_enabled = False
        cfg.runtime.local_translation_enabled = True
        cfg.runtime.tts_output_mode = "passthrough"
        cfg.runtime.asr_language_mode = "auto"
        cfg.runtime.local_echo_guard_enabled = True
        cfg.runtime.local_echo_guard_resume_delay_ms = 420
        cfg.runtime.remote_echo_guard_resume_delay_ms = 520

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "roundtrip.yaml"
            save_config(cfg, path)
            loaded = load_config(path)

        self.assertEqual(loaded.language.meeting_target, "ko")
        self.assertEqual(loaded.language.local_target, "zh-TW")
        self.assertEqual(loaded.tts.style_preset, "broadcast_clear")
        self.assertFalse(loaded.runtime.remote_translation_enabled)
        self.assertTrue(loaded.runtime.local_translation_enabled)
        self.assertEqual(loaded.runtime.tts_output_mode, "passthrough")
        self.assertEqual(loaded.runtime.asr_language_mode, "auto")
        self.assertTrue(loaded.runtime.local_echo_guard_enabled)
        self.assertEqual(loaded.runtime.local_echo_guard_resume_delay_ms, 420)
        self.assertEqual(loaded.runtime.remote_echo_guard_resume_delay_ms, 520)


if __name__ == "__main__":
    unittest.main()
