from __future__ import annotations

import faulthandler
from pathlib import Path
import sys
import threading
import traceback

from app.bootstrap.app_factory import run_from_cli
from app.bootstrap.runtime_paths import runtime_logs_dir


_FAULT_LOG_HANDLE = None


def _configure_runtime_logging() -> None:
    global _FAULT_LOG_HANDLE
    log_dir = runtime_logs_dir()
    fault_path = log_dir / "runtime_crash.log"
    _FAULT_LOG_HANDLE = fault_path.open("a", encoding="utf-8")
    _FAULT_LOG_HANDLE.write(f"\n===== startup {Path.cwd()} =====\n")
    _FAULT_LOG_HANDLE.flush()
    faulthandler.enable(_FAULT_LOG_HANDLE, all_threads=True)

    def _log_exception(prefix: str, exc_type, exc_value, exc_tb) -> None:
        if _FAULT_LOG_HANDLE is None:
            return
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
    return run_from_cli()


if __name__ == "__main__":
    raise SystemExit(main())
