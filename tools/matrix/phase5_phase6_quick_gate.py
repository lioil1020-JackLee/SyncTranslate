"""Quick gate for post-Phase4 validation (Phase5 software matrix + Phase6 soak)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def _run(command: list[str]) -> tuple[int, str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    existing = str(env.get("PYTHONPATH", "") or "").strip()
    env["PYTHONPATH"] = str(repo_root) if not existing else f"{repo_root}{os.pathsep}{existing}"
    proc = subprocess.run(command, capture_output=True, text=True, cwd=str(repo_root), env=env)
    return int(proc.returncode), str(proc.stdout or ""), str(proc.stderr or "")


def run_quick_gate(
    *,
    config_path: str,
    soak_preset: str,
    sample_interval_sec: float,
    out_json: str | None,
    local_asr_language: str,
    tts_output_mode: str,
) -> dict[str, Any]:
    matrix_report = "logs/session_reports/softphone_matrix_test_phase5_quick.json"
    soak_report = "logs/session_reports/runtime_soak_phase6_quick.json"

    matrix_cmd = [
        sys.executable,
        "-m",
        "tools.matrix.softphone_matrix_test",
        "--config",
        str(config_path),
        "--loops",
        "1",
        "--observe-sec",
        "8",
        "--out",
        matrix_report,
    ]
    matrix_rc, matrix_stdout, matrix_stderr = _run(matrix_cmd)

    soak_cmd = [
        sys.executable,
        "-m",
        "tools.matrix.runtime_soak_2hr",
        "--config",
        str(config_path),
        "--preset",
        str(soak_preset),
        "--sample-interval-sec",
        str(sample_interval_sec),
        "--out",
        soak_report,
    ]
    if str(local_asr_language or "").strip():
        soak_cmd.extend(["--local-asr-language", str(local_asr_language).strip()])
    if str(tts_output_mode or "").strip():
        soak_cmd.extend(["--tts-output-mode", str(tts_output_mode).strip()])

    soak_rc, soak_stdout, soak_stderr = _run(soak_cmd)

    matrix_payload: dict[str, Any] = {}
    matrix_path = Path(matrix_report)
    if matrix_path.exists():
        matrix_payload = json.loads(matrix_path.read_text(encoding="utf-8"))

    soak_payload: dict[str, Any] = {}
    soak_path = Path(soak_report)
    if soak_path.exists():
        soak_payload = json.loads(soak_path.read_text(encoding="utf-8"))

    matrix_pass_count = int((matrix_payload.get("summary") or {}).get("pass_count", 0) or 0)
    matrix_total = int((matrix_payload.get("summary") or {}).get("total", 0) or 0)
    matrix_ok = bool(matrix_rc == 0 and matrix_total > 0 and matrix_pass_count == matrix_total)

    runtime_soak = soak_payload.get("runtime_soak") if isinstance(soak_payload.get("runtime_soak"), dict) else {}
    failures = runtime_soak.get("failures") if isinstance(runtime_soak, dict) else []
    if not isinstance(failures, list):
        failures = [str(failures)]
    soak_ok = bool(soak_rc == 0 and len(failures) == 0)

    summary = {
        "matrix_ok": matrix_ok,
        "soak_ok": soak_ok,
        "overall_ok": bool(matrix_ok and soak_ok),
    }

    report: dict[str, Any] = {
        "kind": "phase5_phase6_quick_gate",
        "generated_at": datetime.now().isoformat(),
        "config": str(config_path),
        "summary": summary,
        "matrix": {
            "command": matrix_cmd,
            "return_code": matrix_rc,
            "report": matrix_report,
            "pass_count": matrix_pass_count,
            "total": matrix_total,
            "stdout_tail": matrix_stdout[-1200:],
            "stderr_tail": matrix_stderr[-1200:],
        },
        "soak": {
            "command": soak_cmd,
            "return_code": soak_rc,
            "report": soak_report,
            "preset": str(soak_preset),
            "duration_sec_effective": int(soak_payload.get("duration_sec_effective", 0) or 0),
            "failures": failures,
            "stdout_tail": soak_stdout[-1200:],
            "stderr_tail": soak_stderr[-1200:],
        },
    }

    out_path = Path(out_json or "logs/session_reports/phase5_phase6_quick_gate.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run quick gate for Phase 5/6 with shortened soak test.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--soak-preset", choices=("smoke", "quick", "medium", "full"), default="smoke")
    parser.add_argument("--sample-interval-sec", type=float, default=5.0)
    parser.add_argument("--local-asr-language", default="none")
    parser.add_argument("--tts-output-mode", default="subtitle_only")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    report = run_quick_gate(
        config_path=args.config,
        soak_preset=args.soak_preset,
        sample_interval_sec=args.sample_interval_sec,
        out_json=args.out,
        local_asr_language=args.local_asr_language,
        tts_output_mode=args.tts_output_mode,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if bool((report.get("summary") or {}).get("overall_ok", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
