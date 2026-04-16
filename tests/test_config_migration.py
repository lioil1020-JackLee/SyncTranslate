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
    def test_runtime_defaults_start_with_legacy_pipeline_and_v2_placeholders(self) -> None:
        cfg = AppConfig()
        self.assertEqual(cfg.runtime.asr_pipeline, "v2")
        self.assertEqual(cfg.runtime.asr_v2_backend, "faster_whisper_v2")
        self.assertEqual(cfg.runtime.asr_v2_endpointing, "neural_endpoint")

    def test_normalize_legacy_funasr_fields_to_new_defaults(self) -> None:
        raw = {
            "asr_channels": {
                "local": {
                    "engine": "funasr",
                    "funasr_online_mode": True,
                    "funasr": {"model": "iic/SenseVoiceSmall"},
                    "vad": {"backend": "fsmn_vad"},
                }
            },
            "runtime": {"asr_v2_backend": "funasr_v2"},
        }

        normalized = _normalize_external_config_keys(raw)
        local = normalized["asr_channels"]["local"]

        self.assertEqual(local["engine"], "faster_whisper")
        self.assertEqual(local["vad"]["backend"], "silero_vad")
        self.assertNotIn("funasr_online_mode", local)
        self.assertNotIn("funasr", local)
        self.assertEqual(normalized["runtime"]["asr_v2_backend"], "faster_whisper_v2")

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
            "direction": {"mode": "bidirectional"},
            "asr": {"model": "shared"},
            "asr_channels": {
                "local": {"model": "l"},
                "remote": {"model": "r"},
                "chinese": {"model": "old-c"},
                "english": {"model": "old-e"},
            },
            "llm": {"model": "shared"},
            "llm_channels": {
                "local": {"model": "l"},
                "remote": {"model": "r"},
                "zh_to_en": {"model": "old-z2e"},
                "en_to_zh": {"model": "old-e2z"},
            },
            "tts": {"voice_name": "base", "sample_rate": 24000, "noise_w": 0.65},
            "tts_channels": {
                "local": {"voice_name": "base", "sample_rate": 24000, "noise_w": 0.65},
                "remote": {"voice_name": "r", "sample_rate": 24000, "noise_w": 0.65},
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
                "translation_enabled": True,
                "remote_translation_enabled": True,
                "local_translation_enabled": False,
                "passthrough_gain": 1.0,
                "tts_gain": 1.0,
                "config_schema_version": 5,
                "last_migration_note": "",
            },
            "health_last_success": {"asr": "", "llm": "", "tts": ""},
        }

        presented = _present_external_config_keys(raw)

        self.assertNotIn("direction", presented)
        self.assertNotIn("asr", presented)
        self.assertNotIn("llm", presented)

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
        self.assertEqual(presented["tts_channels"]["local"], {})
        self.assertEqual(presented["tts_channels"]["remote"], {"voice_name": "r"})

        self.assertNotIn("meeting_tts", presented)
        self.assertNotIn("local_tts", presented)
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
        self.assertNotIn("asr_queue_maxsize", presented["runtime"])
        self.assertNotIn("llm_queue_maxsize", presented["runtime"])
        self.assertNotIn("tts_queue_maxsize", presented["runtime"])
        self.assertNotIn("translation_enabled", presented["runtime"])
        self.assertNotIn("remote_translation_enabled", presented["runtime"])
        self.assertNotIn("local_translation_enabled", presented["runtime"])
        self.assertNotIn("passthrough_gain", presented["runtime"])
        self.assertNotIn("tts_gain", presented["runtime"])
        self.assertNotIn("config_schema_version", presented["runtime"])
        self.assertNotIn("last_migration_note", presented["runtime"])
        self.assertNotIn("health_last_success", presented)

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

        self.assertNotIn("direction", payload)
        self.assertNotIn("asr", payload)
        self.assertNotIn("llm", payload)
        self.assertNotIn("meeting_tts", payload)
        self.assertNotIn("local_tts", payload)
        self.assertNotIn("chinese_tts", payload)
        self.assertNotIn("english_tts", payload)

        self.assertEqual(runtime.get("asr_queue_maxsize_local"), 111)
        self.assertEqual(runtime.get("asr_queue_maxsize_remote"), 222)
        self.assertNotIn("asr_queue_maxsize_chinese", runtime)
        self.assertNotIn("asr_queue_maxsize_english", runtime)
        self.assertNotIn("asr_queue_maxsize", runtime)
        self.assertNotIn("llm_queue_maxsize", runtime)
        self.assertNotIn("tts_queue_maxsize", runtime)
        self.assertNotIn("remote_translation_enabled", runtime)
        self.assertNotIn("local_translation_enabled", runtime)
        self.assertNotIn("translation_enabled", runtime)
        self.assertNotIn("config_schema_version", runtime)
        self.assertNotIn("last_migration_note", runtime)

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
        self.assertNotIn("direction", payload)
        self.assertNotIn("asr", payload)
        self.assertNotIn("llm", payload)
        self.assertNotIn("meeting_tts", payload)
        self.assertNotIn("local_tts", payload)

    def test_load_config_supports_compact_external_shape(self) -> None:
        compact = {
            "audio": {},
            "language": {
                "remote_translation_target": "de",
                "local_translation_target": "it",
            },
            "asr_channels": {
                "local": {"model": "local-asr"},
                "remote": {"model": "remote-asr"},
            },
            "llm_channels": {
                "local": {"base_url": "http://127.0.0.1:1234", "caption_profile": "technical_meeting"},
                "remote": {"base_url": "http://127.0.0.1:1234", "request_timeout_sec": 12},
            },
            "tts": {
                "engine": "edge_tts",
                "voice_name": "zh-TW-HsiaoChenNeural",
                "sample_rate": 24000,
            },
            "tts_channels": {
                "remote": {"voice_name": "en-US-JennyNeural"},
            },
            "runtime": {
                "sample_rate": 48000,
                "chunk_ms": 40,
                "remote_asr_language": "zh-TW",
                "local_asr_language": "en",
                "remote_translation_target": "de",
                "local_translation_target": "it",
            },
        }

        cfg = AppConfig.from_dict(_normalize_external_config_keys(compact))

        self.assertEqual(cfg.asr.model, "local-asr")
        self.assertEqual(cfg.asr_channels.remote.model, "remote-asr")
        self.assertEqual(cfg.llm.caption_profile, "technical_meeting")
        self.assertEqual(cfg.llm_channels.remote.request_timeout_sec, 12)
        self.assertEqual(cfg.meeting_tts.voice_name, "zh-TW-HsiaoChenNeural")
        self.assertEqual(cfg.local_tts.voice_name, "en-US-JennyNeural")

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
