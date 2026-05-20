"""Execute softphone validation matrix and write run records."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from tools.matrix.multi_route_test import run_multi_route_test


PROFILES = [
    "Zoom",
    "Teams",
    "Google Meet (Chrome)",
    "LINE",
    "WhatsApp Desktop",
    "Discord",
]


def _to_status(ok: bool) -> str:
    return "通過" if ok else "失敗"


def _matrix_table_lines(results: list[dict[str, Any]], report_path: str) -> list[str]:
    lines = [
        "",
        "## 2) 案例矩陣（自動回填）",
        "",
        "| 軟體 | 版本 | D1 雙向翻譯 | D2 30 分鐘穩定 | D3 App crash 後 silence | D4 Bridge crash 後 silence | D5 48k/16k 重採樣 | 結果 | 備註 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in results:
        d1 = _to_status(bool(row.get("pass_d1", False)))
        d2 = "未測"
        d3 = "未測"
        d4 = "未測"
        d5 = "未測"
        overall = "通過" if d1 == "通過" else "失敗"
        note = f"report={report_path}"
        lines.append(
            f"| {row.get('profile', '')} | auto | {d1} | {d2} | {d3} | {d4} | {d5} | {overall} | {note} |"
        )
    return lines


def _execution_log_lines(results: list[dict[str, Any]], report_path: str) -> list[str]:
    today = datetime.now().strftime("%Y-%m-%d")
    lines = [
        "",
        "## 3) 測試執行紀錄（逐次）",
        "",
        "| 日期 | 測試者 | 軟體 | 測試項目 | 報告檔案 | 結果 | 問題摘要 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in results:
        d1_ok = bool(row.get("pass_d1", False))
        result_text = _to_status(d1_ok)
        issue = "" if d1_ok else f"remote_final={row.get('remote_final', 0)}"
        lines.append(f"| {today} | auto-script | {row.get('profile', '')} | D1 | {report_path} | {result_text} | {issue} |")
    return lines


def _rewrite_template(matrix_file: Path, results: list[dict[str, Any]], report_path: str) -> None:
    matrix_file.parent.mkdir(parents=True, exist_ok=True)
    original = matrix_file.read_text(encoding="utf-8") if matrix_file.exists() else "# 通話軟體矩陣驗證結果\n"
    marker = "## 2) 案例矩陣（自動回填）"
    if marker in original:
        original = original.split(marker)[0].rstrip() + "\n"
    new_text = "\n".join(
        [original.rstrip()] + _matrix_table_lines(results, report_path) + _execution_log_lines(results, report_path)
    ).rstrip() + "\n"
    matrix_file.write_text(new_text, encoding="utf-8")


def run_matrix_test(
    matrix_path: str,
    config_path: str,
    wav_path: str,
    loops: int,
    observe_sec: float,
    out_json: str | None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for profile in PROFILES:
        row = run_multi_route_test(
            config_path=config_path,
            wav_path=wav_path,
            loops=loops,
            observe_sec=observe_sec,
            out_json=None,
            profile_name=profile,
        )
        rows.append(row)

    report = {
        "kind": "softphone_matrix",
        "generated_at": datetime.now().isoformat(),
        "matrix": matrix_path,
        "profiles": rows,
        "summary": {
            "pass_count": sum(1 for r in rows if bool(r.get("pass_d1", False))),
            "total": len(rows),
        },
    }

    out_report = out_json or "logs/session_reports/softphone_matrix_test.json"
    out_path = Path(out_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    _rewrite_template(Path(matrix_path), rows, str(out_path).replace("\\", "/"))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run softphone matrix automation.")
    parser.add_argument("--matrix", default="logs/session_reports/softphone_matrix_results.md")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--wav", default="artifacts/driver/synctranslate_virtual_audio/virtual_mic_recording.wav")
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument("--observe-sec", type=float, default=8.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    result = run_matrix_test(
        matrix_path=args.matrix,
        config_path=args.config,
        wav_path=args.wav,
        loops=args.loops,
        observe_sec=args.observe_sec,
        out_json=args.out,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    passed = int(result["summary"]["pass_count"])
    total = int(result["summary"]["total"])
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
