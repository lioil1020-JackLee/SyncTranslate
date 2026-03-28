from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from app.bootstrap.app_factory import _default_config_path, create_cli_parser


class AppFactoryTests(TestCase):
    def test_parser_defaults_to_repo_config_in_dev_mode(self) -> None:
        args = create_cli_parser().parse_args([])

        self.assertEqual(args.config, _default_config_path())
        self.assertTrue(Path(args.config).is_absolute())
        self.assertEqual(Path(args.config).name, "config.yaml")

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
