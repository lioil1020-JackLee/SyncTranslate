from __future__ import annotations

import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from app.infra.asr.backend_resolution import resolve_backend_for_language
from app.infra.asr.backend_v2 import _prepare_audio16k, _sanitize_funasr_text, _should_drop_funasr_text
from app.infra.asr.funasr_registry import FunASRModelHandle, FunASRModelRegistry
from app.infra.asr.manager_v2 import ASRManagerV2
from app.infra.config.schema import AppConfig
from app.ui.main_window import MainWindow


class _QtTestCase(unittest.TestCase):
    _app: QApplication | None = None

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._app = QApplication.instance() or QApplication([])


class BackendRoutingTests(unittest.TestCase):
    def test_chinese_language_routes_to_funasr(self) -> None:
        resolution = resolve_backend_for_language("zh-TW")
        self.assertEqual(resolution.backend_name, "funasr_v2")
        self.assertEqual(resolution.language_family, "chinese")

    def test_non_chinese_language_routes_to_faster_whisper(self) -> None:
        resolution = resolve_backend_for_language("ja")
        self.assertEqual(resolution.backend_name, "faster_whisper_v2")
        self.assertEqual(resolution.language_family, "non_chinese")

    def test_auto_routes_to_faster_whisper(self) -> None:
        resolution = resolve_backend_for_language("auto")
        self.assertEqual(resolution.backend_name, "faster_whisper_v2")
        self.assertEqual(resolution.language_family, "auto")

    def test_none_disables_asr(self) -> None:
        cfg = AppConfig()
        cfg.runtime.asr_pipeline = "v2"
        cfg.runtime.local_asr_language = "none"
        manager = ASRManagerV2(cfg, pipeline_revision=11)

        manager.start("local", lambda _event: None)

        self.assertEqual(manager.stats()["local"]["resolved_backend"], "disabled")

    def test_funasr_text_sanitizer_removes_event_emojis(self) -> None:
        cleaned = _sanitize_funasr_text("🎼 三進三出後，衝進西螺福興宮。😊")

        self.assertEqual(cleaned, "三進三出後，衝進西螺福興宮。")

    def test_funasr_audio_prep_applies_mild_gain_compensation(self) -> None:
        t = np.linspace(0.0, 0.1, 1600, endpoint=False, dtype=np.float32)
        audio = 0.005 * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)

        prepared = _prepare_audio16k(audio, 16000)

        self.assertGreater(float(np.sqrt(np.mean(np.square(prepared)))), 0.005)

    def test_funasr_drop_rule_rejects_short_low_speech_ratio_text(self) -> None:
        audio = np.zeros((1600,), dtype=np.float32)
        audio[300:320] = 0.01

        self.assertTrue(_should_drop_funasr_text("嗯。", audio=audio))


class FunASRRegistryTests(unittest.TestCase):
    def test_registry_initializes_same_model_only_once(self) -> None:
        registry = FunASRModelRegistry()
        calls: list[str] = []

        def _fake_load_handle(*, kind: str, key: str, model_path: str, requested_device: str) -> FunASRModelHandle:
            calls.append(key)
            return FunASRModelHandle(
                kind=kind,
                key=key,
                model=object(),
                device_effective="cpu",
                model_path=model_path,
                init_mode="lazy",
            )

        registry._load_handle = _fake_load_handle  # type: ignore[method-assign]

        first = registry.get_vad(requested_device="cuda")
        second = registry.get_vad(requested_device="cuda")

        self.assertEqual(len(calls), 1)
        self.assertEqual(first.device_effective, "cpu")
        self.assertEqual(second.init_mode, "warm")

    def test_registry_keeps_vad_sessions_isolated_by_session_key(self) -> None:
        registry = FunASRModelRegistry()
        calls: list[str] = []

        def _fake_load_handle(*, kind: str, key: str, model_path: str, requested_device: str) -> FunASRModelHandle:
            calls.append(key)
            return FunASRModelHandle(
                kind=kind,
                key=key,
                model=object(),
                device_effective="cpu",
                model_path=model_path,
                init_mode="lazy",
            )

        registry._load_handle = _fake_load_handle  # type: ignore[method-assign]

        first = registry.get_vad(requested_device="cuda", session_key="local")
        second = registry.get_vad(requested_device="cuda", session_key="remote")

        self.assertEqual(len(calls), 2)
        self.assertNotEqual(first.key, second.key)


class ErrorDedupTests(_QtTestCase):
    def test_main_window_deduplicates_repeated_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = os.path.join(tmp, "config.yaml")
            window = MainWindow(config_path)

            window._report_error("passthrough_playback_failed sample error")
            window._report_error("passthrough_playback_failed sample error")
            window._report_error("passthrough_playback_failed sample error")
            errors = window._get_recent_errors()

            self.assertEqual(sum(1 for item in errors if "passthrough_playback_failed sample error" in item), 2)
            self.assertTrue(any("[dedup]" in item for item in errors))


if __name__ == "__main__":
    unittest.main()
