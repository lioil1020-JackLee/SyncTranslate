"""Run a 2-hour soak via tools/runtime_soak_30m.py and normalize output report path."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PRESET_DURATIONS: dict[str, int] = {
    "smoke": 120,
    "quick": 600,
    "medium": 1800,
    "full": 7200,
}


def _latest_soak_report() -> Path | None:
    report_dir = Path("logs/session_reports")
    candidates = sorted(report_dir.glob("runtime_soak_report_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def run_2hr_soak(
    config_path: str,
    out_json: str | None = None,
    sample_interval_sec: float = 5.0,
    duration_sec: int = 7200,
    preset: str = "full",
    local_asr_language: str = "",
    remote_asr_language: str = "",
    tts_output_mode: str = "",
) -> dict[str, object]:
    normalized_preset = str(preset or "full").strip().lower()
    if normalized_preset not in PRESET_DURATIONS:
        normalized_preset = "full"
    effective_duration = max(30, int(duration_sec or PRESET_DURATIONS[normalized_preset]))
    command = [
        sys.executable,
        "tools/runtime_soak_30m.py",
        "--config",
        str(config_path),
        "--duration-sec",
        str(effective_duration),
        "--sample-interval-sec",
        str(sample_interval_sec),
    ]
    if str(local_asr_language or "").strip():
        command.extend(["--local-asr-language", str(local_asr_language).strip()])
    if str(remote_asr_language or "").strip():
        command.extend(["--remote-asr-language", str(remote_asr_language).strip()])
    if str(tts_output_mode or "").strip():
        command.extend(["--tts-output-mode", str(tts_output_mode).strip()])
    proc = subprocess.run(command, capture_output=True, text=True)

    latest = _latest_soak_report()
    report: dict[str, object] = {
        "kind": "runtime_soak_2hr",
        "preset": normalized_preset,
        "duration_sec_effective": int(effective_duration),
        "command": command,
        "return_code": int(proc.returncode),
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
        "source_report": str(latest).replace("\\", "/") if latest else "",
    }

    if latest and latest.exists():
        try:
            src_payload = json.loads(latest.read_text(encoding="utf-8"))
            report["runtime_soak"] = src_payload
        except Exception as exc:
            report["parse_error"] = str(exc)

    out_path = Path(out_json or "logs/session_reports/runtime_soak_2hr.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run soak automation (supports smoke/quick/medium/full presets).")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default=None)
    parser.add_argument("--sample-interval-sec", type=float, default=5.0)
    parser.add_argument("--preset", choices=tuple(PRESET_DURATIONS.keys()), default="full")
    parser.add_argument("--duration-sec", type=int, default=0, help="Override preset duration when > 0")
    parser.add_argument("--local-asr-language", default="")
    parser.add_argument("--remote-asr-language", default="")
    parser.add_argument("--tts-output-mode", default="")
    args = parser.parse_args()

    chosen_duration = int(args.duration_sec) if int(args.duration_sec or 0) > 0 else int(PRESET_DURATIONS[args.preset])

    result = run_2hr_soak(
        args.config,
        args.out,
        args.sample_interval_sec,
        chosen_duration,
        args.preset,
        args.local_asr_language,
        args.remote_asr_language,
        args.tts_output_mode,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result.get("return_code", 1))


if __name__ == "__main__":
    raise SystemExit(main())
