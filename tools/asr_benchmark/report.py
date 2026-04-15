"""ASR benchmark report generator.

Reads benchmark_results.jsonl and produces a summary table comparing
different profiles / backends.

Usage
-----
uv run python tools/asr_benchmark/report.py results/benchmark_results.jsonl
uv run python tools/asr_benchmark/report.py results/ --format csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_results(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if path.is_dir():
        path = path / "benchmark_results.jsonl"
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def _avg(values: list[float | None]) -> str:
    nums = [v for v in values if v is not None]
    if not nums:
        return "N/A"
    return f"{sum(nums) / len(nums):.1f}"


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

COLUMNS = [
    "file",
    "profile",
    "source",
    "duration_ms",
    "first_partial_latency_ms",
    "final_latency_ms",
    "partial_count",
    "final_count",
    "endpoint_count",
    "cer",
    "wer",
]


def _print_table(records: list[dict[str, Any]]) -> None:
    if not records:
        print("No records found.")
        return

    # Dynamic column widths
    widths = {col: len(col) for col in COLUMNS}
    for r in records:
        for col in COLUMNS:
            v = str(r.get(col, ""))
            widths[col] = max(widths[col], len(v))

    header = "  ".join(col.ljust(widths[col]) for col in COLUMNS)
    sep = "  ".join("-" * widths[col] for col in COLUMNS)
    print(header)
    print(sep)
    for r in records:
        row = "  ".join(str(r.get(col, "")).ljust(widths[col]) for col in COLUMNS)
        print(row)

    # Summary
    print()
    print("Summary:")
    print(f"  Files processed:       {len(records)}")
    print(f"  Avg first partial (ms): {_avg([r.get('first_partial_latency_ms') for r in records])}")
    print(f"  Avg final latency (ms): {_avg([r.get('final_latency_ms') for r in records])}")
    cer_vals = [r.get("cer") for r in records if r.get("cer") is not None]
    wer_vals = [r.get("wer") for r in records if r.get("wer") is not None]
    if cer_vals:
        print(f"  Avg CER:               {_avg(cer_vals)}")
    if wer_vals:
        print(f"  Avg WER:               {_avg(wer_vals)}")


def _print_csv(records: list[dict[str, Any]], out_path: Path | None = None) -> None:
    import io
    sink: Any = io.StringIO() if out_path is None else out_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(sink, fieldnames=COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for r in records:
        writer.writerow({col: r.get(col, "") for col in COLUMNS})
    if out_path is None:
        print(sink.getvalue())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SyncTranslate ASR benchmark report",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", help="Path to benchmark_results.jsonl or directory containing it")
    p.add_argument("--format", choices=["table", "csv", "json"], default="table")
    p.add_argument("--output", default=None, help="Write output to file instead of stdout")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    records = _load_results(Path(args.input))
    if not records:
        print("[report] No records found.", file=sys.stderr)
        return 1

    out_path = Path(args.output) if args.output else None

    if args.format == "table":
        _print_table(records)
    elif args.format == "csv":
        _print_csv(records, out_path)
    elif args.format == "json":
        text = json.dumps(records, ensure_ascii=False, indent=2)
        if out_path:
            out_path.write_text(text, encoding="utf-8")
        else:
            print(text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
