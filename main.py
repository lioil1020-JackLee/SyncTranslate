from __future__ import annotations

import argparse
import faulthandler
from pathlib import Path
import sys
import threading
import traceback

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from app.device_manager import DeviceManager
from app.main_window import MainWindow
from app.settings import load_config


_FAULT_LOG_HANDLE = None


def _apply_windows_app_id() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("lioil.synctranslate")
    except Exception:
        return


def _configure_runtime_logging() -> None:
    global _FAULT_LOG_HANDLE
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    fault_path = log_dir / "runtime_crash.log"
    _FAULT_LOG_HANDLE = fault_path.open("a", encoding="utf-8")
    _FAULT_LOG_HANDLE.write(f"\n===== startup {Path.cwd()} =====\n")
    _FAULT_LOG_HANDLE.flush()
    faulthandler.enable(_FAULT_LOG_HANDLE, all_threads=True)

    def _log_exception(prefix: str, exc_type, exc_value, exc_tb) -> None:
        _FAULT_LOG_HANDLE.write(prefix + "\n")
        traceback.print_exception(exc_type, exc_value, exc_tb, file=_FAULT_LOG_HANDLE)
        _FAULT_LOG_HANDLE.flush()

    def _sys_excepthook(exc_type, exc_value, exc_tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            # Ctrl+C is an expected shutdown path in terminal runs.
            return
        _log_exception("Unhandled exception", exc_type, exc_value, exc_tb)
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    def _threading_excepthook(args: threading.ExceptHookArgs) -> None:
        if issubclass(args.exc_type, KeyboardInterrupt):
            return
        _log_exception(f"Unhandled thread exception: {args.thread.name if args.thread else 'unknown'}", args.exc_type, args.exc_value, args.exc_traceback)

    sys.excepthook = _sys_excepthook
    threading.excepthook = _threading_excepthook


def main() -> int:
    _configure_runtime_logging()
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
