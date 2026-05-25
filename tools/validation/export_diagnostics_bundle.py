from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.application.first_run_readiness import evaluate_first_run_readiness
from app.domain.version import build_metadata
from app.infra.config.settings_store import load_config
from tools.validation.common import VALIDATION_OUTPUT_DIR, list_audio_devices
from tools.validation.preflight_release_check import build_report as build_preflight_report
from tools.validation.validate_windows_audio_runtime import build_report as build_windows_audio_report


SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)(.+)"),
    re.compile(r"(?i)([A-Za-z0-9_-]*token[A-Za-z0-9_-]*\s*[:=]\s*)(.+)"),
    re.compile(r"(?i)(password\s*[:=]\s*)(.+)"),
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a sanitized SyncTranslate diagnostics bundle.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="", help="Output zip path")
    parser.add_argument("--include-wav", action="append", default=[], help="Explicit smoke-test WAV file to include")
    return parser


def _sanitize(text: str) -> str:
    sanitized = text
    for pattern in SECRET_PATTERNS:
        sanitized = pattern.sub(r"\1***REDACTED***", sanitized)
    sanitized = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._\-]+", r"\1***REDACTED***", sanitized)
    return sanitized


def _run_check(config_path: str) -> str:
    try:
        result = subprocess.run(
            [sys.executable, "main.py", "--check", "--config", config_path],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        return _sanitize((result.stdout or "") + ("\nSTDERR:\n" + result.stderr if result.stderr else ""))
    except Exception as exc:
        return f"main.py --check failed to run: {exc}"


def _write_text(zf: zipfile.ZipFile, name: str, text: str) -> None:
    zf.writestr(name, _sanitize(text))


def build_bundle(*, config_path: str = "config.yaml", output: str = "", include_wav: list[str] | None = None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output) if output else VALIDATION_OUTPUT_DIR / f"diagnostics_bundle_{timestamp}.zip"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    metadata = build_metadata(
        config_schema_version=int(getattr(config.runtime, "config_schema_version", 0) or 0),
        runtime_mode=str(getattr(config.runtime, "session_mode", "meeting") or "meeting"),
    )
    windows_report = build_windows_audio_report(config_path, probe_bridge=False, probe_capture=False)
    preflight_report = build_preflight_report(config_path, mode=str(getattr(config.runtime, "session_mode", "meeting") or "meeting"))
    windows_details = getattr(windows_report, "details", {}) or {}
    preflight_details = getattr(preflight_report, "details", {}) or {}
    readiness = evaluate_first_run_readiness(config, probe_bridge=False)
    inputs, outputs, device_error = list_audio_devices()
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        cfg_path = Path(config_path)
        if cfg_path.exists():
            _write_text(zf, "config.sanitized.yaml", cfg_path.read_text(encoding="utf-8", errors="replace"))
        _write_text(zf, "main_check.txt", _run_check(config_path))
        _write_text(zf, "windows_audio_runtime.json", json.dumps(windows_report.to_dict(), ensure_ascii=False, indent=2))
        _write_text(zf, "preflight_release.json", json.dumps(preflight_report.to_dict(), ensure_ascii=False, indent=2))
        _write_text(
            zf,
            "driver_runtime_summary.json",
            json.dumps(
                    {
                        "windows_audio_runtime": {
                            "driver_format_status": windows_details.get("driver_format_status"),
                            "driver_format_expected": windows_details.get("driver_format_expected"),
                            "wdk_environment_status": windows_details.get("wdk_environment_status"),
                            "driver_build_tools_available": windows_details.get("driver_build_tools_available"),
                            "protocol_v2_ready": windows_details.get("protocol_v2_ready"),
                            "pcm16_stereo_boundary_ready": windows_details.get("pcm16_stereo_boundary_ready"),
                        },
                        "preflight": {
                            "driver_format_status": preflight_details.get("driver_format_status"),
                            "driver_format_expected": preflight_details.get("driver_format_expected"),
                            "wdk_environment_status": preflight_details.get("wdk_environment_status"),
                            "driver_build_tools_available": preflight_details.get("driver_build_tools_available"),
                        },
                    },
                ensure_ascii=False,
                indent=2,
            ),
        )
        _write_text(zf, "readiness.json", json.dumps(readiness.to_dict(), ensure_ascii=False, indent=2))
        _write_text(
            zf,
            "devices.json",
            json.dumps({"inputs": inputs, "outputs": outputs, "error": device_error}, ensure_ascii=False, indent=2),
        )
        _write_text(zf, "build_metadata.json", json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2))
        for log_dir in (ROOT / "logs", ROOT / "runtime_logs"):
            if not log_dir.exists():
                continue
            for path in sorted(log_dir.glob("*.log"), key=lambda item: item.stat().st_mtime, reverse=True)[:8]:
                try:
                    _write_text(zf, f"logs/{path.name}", path.read_text(encoding="utf-8", errors="replace")[-200_000:])
                except Exception:
                    continue
        for wav in include_wav or []:
            path = Path(wav)
            if path.exists() and path.suffix.lower() == ".wav":
                zf.write(path, f"explicit_wav/{path.name}")
    return output_path


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    try:
        path = build_bundle(config_path=args.config, output=args.output, include_wav=args.include_wav)
        print(f"PASS diagnostics bundle written: {path}")
        return 0
    except Exception as exc:
        print(f"FAIL diagnostics bundle could not be created: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
