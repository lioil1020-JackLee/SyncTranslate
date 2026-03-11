from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from app.device_manager import DeviceManager
from app.settings import load_config
from app.ui_main import MainWindow


def _apply_windows_app_id() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("lioil.synctranslate")
    except Exception:
        return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--check", action="store_true", help="Run config + device checks without opening the UI")
    args = parser.parse_args()

    config = load_config(args.config)
    devices = DeviceManager().list_all()
    if args.check:
        print(f"Config OK: {args.config}")
        print(f"mode={config.direction.mode}")
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
    icon_path = Path("lioil.ico")
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    window = MainWindow(args.config)
    window.show()
    try:
        return app.exec()
    except KeyboardInterrupt:
        # Allow clean Ctrl+C exit without traceback spam in terminal runs.
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
