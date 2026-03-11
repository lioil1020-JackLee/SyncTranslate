from __future__ import annotations

import argparse
import sys

from PySide6.QtWidgets import QApplication

from app.device_manager import DeviceManager
from app.settings import load_config
from app.ui_main import MainWindow


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--check", action="store_true", help="Run config + device checks without opening the UI")
    args = parser.parse_args()

    config = load_config(args.config)
    devices = DeviceManager().list_all()
    if args.check:
        print(f"Config OK: {args.config}")
        print(f"sample_rate={config.sample_rate} chunk_ms={config.chunk_ms}")
        print(f"devices_found={len(devices)}")
        return 0

    app = QApplication(sys.argv)
    window = MainWindow(args.config)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

