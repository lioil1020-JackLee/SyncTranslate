from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from app.infra.audio.device_registry import DeviceManager
from app.infra.config.schema import translation_enabled_for_source
from app.infra.config.settings_store import load_config
from app.ui.main_window import MainWindow


def _default_config_path() -> str:
    if getattr(sys, "frozen", False):
        base_dir = Path(sys.executable).resolve().parent
        target = (base_dir / "config.yaml").resolve()
        if target.exists():
            return str(target)
        bundled_candidates = []
        meipass = getattr(sys, "_MEIPASS", "")
        if meipass:
            bundled_candidates.append((Path(meipass) / "config.yaml").resolve())
        bundled_candidates.append((base_dir / "_internal" / "config.yaml").resolve())
        for candidate in bundled_candidates:
            if not candidate.exists():
                continue
            try:
                shutil.copyfile(candidate, target)
                return str(target)
            except Exception:
                return str(candidate)
    else:
        base_dir = Path(__file__).resolve().parents[2]
    return str((base_dir / "config.yaml").resolve())


def create_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=_default_config_path())
    parser.add_argument("--check", action="store_true", help="Run config + device checks without opening the UI")
    parser.add_argument("--healthcheck-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--healthcheck-config", default="", help=argparse.SUPPRESS)
    return parser


def run_from_cli(argv: list[str] | None = None) -> int:
    parser = create_cli_parser()
    args = parser.parse_args(argv)

    if args.healthcheck_worker:
        worker_args: list[str] = []
        if args.healthcheck_config:
            worker_args.append(args.healthcheck_config)
        from app.local_ai.healthcheck_worker import main as healthcheck_worker_main

        return int(healthcheck_worker_main(worker_args))

    config = load_config(args.config)
    devices = DeviceManager().list_all()
    if args.check:
        print(f"Config OK: {args.config}")
        print(f"mode={config.direction.mode}")
        print(
            "runtime_modes="
            f"remote_translation_enabled={translation_enabled_for_source(config.runtime, 'remote')} "
            f"local_translation_enabled={translation_enabled_for_source(config.runtime, 'local')} "
            "asr_language_mode=auto "
            f"tts_output_mode={str(getattr(config.runtime, 'tts_output_mode', 'subtitle_only') or 'subtitle_only')}"
        )
        print(f"sample_rate={config.runtime.sample_rate} chunk_ms={config.runtime.chunk_ms}")
        print(f"asr={config.asr.engine}:{config.asr.model} device={config.asr.device}")
        print(f"llm={config.llm.backend} model={config.llm.model} base_url={config.llm.base_url}")
        print(
            "meeting_tts="
            f"{config.meeting_tts.engine} voice={config.meeting_tts.voice_name or config.meeting_tts.model_path}"
        )
        print(
            "local_tts="
            f"{config.local_tts.engine} voice={config.local_tts.voice_name or config.local_tts.model_path}"
        )
        print(f"devices_found={len(devices)}")
        return 0

    _apply_windows_app_id()
    app = QApplication(sys.argv)
    icon_file = _resolve_icon_path()
    if icon_file:
        app.setWindowIcon(QIcon(icon_file))
    window = MainWindow(args.config)
    if icon_file:
        window.setWindowIcon(QIcon(icon_file))
    window.show()
    try:
        return app.exec()
    except KeyboardInterrupt:
        return 130


def _apply_windows_app_id() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("lioil.synctranslate")
    except Exception:
        return


def _resolve_icon_path() -> str | None:
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", os.path.abspath("."))
    else:
        base = os.path.abspath(".")
    candidate = os.path.join(base, "lioil.ico")
    if os.path.exists(candidate):
        return candidate

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate2 = os.path.join(script_dir, "..", "..", "lioil.ico")
    candidate2 = os.path.normpath(candidate2)
    if os.path.exists(candidate2):
        return candidate2

    repo_root_candidate = Path(__file__).resolve().parent.parent.parent / "lioil.ico"
    if repo_root_candidate.exists():
        return str(repo_root_candidate)
    return None


__all__ = ["create_cli_parser", "run_from_cli"]
