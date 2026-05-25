from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from app.bootstrap.app_factory import _default_config_path, create_cli_parser, run_from_cli
from app.infra.config.schema import AppConfig
from tools.validation.common import ValidationItem, ValidationReport


class AppFactoryTests(TestCase):
    def test_parser_defaults_to_repo_config_in_dev_mode(self) -> None:
        args = create_cli_parser().parse_args([])

        self.assertEqual(args.config, _default_config_path())
        self.assertTrue(Path(args.config).is_absolute())
        self.assertEqual(Path(args.config).name, "config.yaml")

    def test_parser_accepts_packaged_runtime_checks(self) -> None:
        args = create_cli_parser().parse_args(["--llm-runtime-check", "--translation-smoke"])

        self.assertTrue(args.llm_runtime_check)
        self.assertTrue(args.translation_smoke)

    def test_default_config_path_uses_executable_dir_when_frozen(self) -> None:
        fake_exe = Path(r"E:\dist\SyncTranslate\SyncTranslate.exe")
        with patch.object(sys, "frozen", True, create=True):
            with patch.object(sys, "executable", str(fake_exe)):
                self.assertEqual(_default_config_path(), str((fake_exe.parent / "config.yaml").resolve()))

    def test_default_config_path_copies_bundled_config_next_to_frozen_exe(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            exe_dir = root / "dist" / "SyncTranslate"
            bundle_dir = root / "bundle"
            exe_dir.mkdir(parents=True)
            bundle_dir.mkdir(parents=True)
            bundled = bundle_dir / "config.yaml"
            bundled.write_text("runtime:\n  sample_rate: 48000\n", encoding="utf-8")
            fake_exe = exe_dir / "SyncTranslate.exe"
            target = exe_dir / "config.yaml"

            with patch.object(sys, "frozen", True, create=True):
                with patch.object(sys, "executable", str(fake_exe)):
                    with patch.object(sys, "_MEIPASS", str(bundle_dir), create=True):
                        resolved = Path(_default_config_path())

            self.assertEqual(resolved, target.resolve())
            self.assertTrue(target.exists())
            self.assertEqual(target.read_text(encoding="utf-8"), bundled.read_text(encoding="utf-8"))

    def test_check_output_includes_v2_health_summary(self) -> None:
        cfg = AppConfig()
        report = ValidationReport(
            "check",
            "WARN",
            [
                ValidationItem("bridge_status", "WARN", "Bridge is not required in meeting mode."),
                ValidationItem("driver_status", "WARN", "Driver missing; meeting mode still works."),
                ValidationItem("audio_devices", "PASS", "devices", {"input_count": 2, "output_count": 3}),
            ],
            {
                "config_schema_version": 7,
                "session_mode": "meeting",
                "meeting_audio_source": "system_input",
                "selected_asr_backend": "faster-whisper",
                "virtual_audio_required": False,
                "bridge_required": False,
                "config_migration_status": "current",
                "validation_warnings": [],
            },
        )
        with patch("app.bootstrap.app_factory.load_config", return_value=cfg):
            with patch("app.bootstrap.app_factory.save_config"):
                with patch("app.bootstrap.app_factory.AutoPopulateDevicesService"):
                    with patch("app.bootstrap.app_factory.DeviceManager") as manager_cls:
                        with patch("app.bootstrap.app_factory.build_check_report", return_value=report):
                            manager_cls.return_value.list_all.return_value = [object(), object()]
                            from io import StringIO
                            import contextlib

                            output = StringIO()
                            with contextlib.redirect_stdout(output):
                                exit_code = run_from_cli(["--check", "--config", "config.yaml"])

        text = output.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("config_schema_version=7", text)
        self.assertIn("session_mode=meeting", text)
        self.assertIn("meeting_audio_source=system_input", text)
        self.assertIn("selected_asr_backend=faster-whisper", text)
        self.assertIn("virtual_audio_required=false", text)
        self.assertIn("bridge_required=false", text)
        self.assertIn("input_device_count=2", text)
        self.assertIn("output_device_count=3", text)
