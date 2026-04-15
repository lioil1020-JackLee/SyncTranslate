from __future__ import annotations

import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.infra.asr.backend_resolution import resolve_backend_for_language
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
