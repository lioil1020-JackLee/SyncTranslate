from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Run the same healthcheck path used by the UI button.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)

    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication

    from app.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([str(root / "main.py")])
    window = MainWindow(str((root / args.config).resolve()))
    try:
        window.run_system_check()
        deadline = time.monotonic() + max(1.0, float(args.timeout))
        while time.monotonic() < deadline:
            app.processEvents()
            window._drain_health_check_results()
            if not window._healthcheck_service.running:
                break
            time.sleep(0.05)

        app.processEvents()
        window._drain_health_check_results()
        status = window.statusBar().currentMessage()
        diagnostics = window.diagnostics_page.diagnostics_details.toPlainText()
        print(f"status: {status}")
        print(diagnostics)

        if window._healthcheck_service.running:
            print("error: healthcheck did not finish before timeout")
            return 2
        if "健康檢查：正常" in status and "ASR: 正常" in diagnostics:
            return 0
        return 1
    finally:
        window.close()


if __name__ == "__main__":
    raise SystemExit(main())
