"""run_regression_corpus.py — ASR regression corpus runner.

Reads a manifest.yaml that defines audio samples and accuracy thresholds,
runs ASR benchmarks for each model × mode combination, and writes
summary.json and summary.md reports.

Exit code 0 = all thresholds met; 1 = one or more failures or errors.

Usage:
    uv run python tools/asr_benchmark/run_regression_corpus.py \\
        --manifest downloads/asr_regression/manifest.yaml \\
        [--models belle,turbo] [--modes meeting,dialogue] \\
        [--speed 8] [--output-dir <path>] [--quick] [--skip-missing]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml  # PyYAML
except ImportError:
    yaml = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SampleConfig:
    id: str
    language: str
    category: str
    audio: str
    reference: str
    description: str = ""
    min_accuracy: dict[str, float] = field(default_factory=dict)
    max_dropped_chunks: int = 0
    max_repetition_ratio: float = 0.05


@dataclass
class SampleResult:
    sample_id: str
    model: str
    mode: str
    accuracy: float | None = None
    dropped_chunks: int = 0
    repetition_ratio: float = 0.0
    passed: bool = False
    failure_reasons: list[str] = field(default_factory=list)
    error: str = ""


@dataclass
class RunSummary:
    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    results: list[SampleResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Manifest loading and validation
# ---------------------------------------------------------------------------

def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load and return manifest dict from a YAML file."""
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to load the regression manifest.\n"
            "Install it with:  pip install pyyaml"
        )
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(
            f"Manifest must be a YAML mapping, got {type(data).__name__}"
        )
    return data


def parse_samples(manifest: dict[str, Any]) -> list[SampleConfig]:
    """Parse and validate sample entries from a manifest dict."""
    raw_samples = manifest.get("samples")
    if not isinstance(raw_samples, list):
        raise ValueError("Manifest must have a 'samples' list")
    samples: list[SampleConfig] = []
    for i, raw in enumerate(raw_samples):
        if not isinstance(raw, dict):
            raise ValueError(f"samples[{i}] must be a mapping, got {type(raw).__name__}")
        for required in ("id", "language", "audio", "reference"):
            if required not in raw:
                raise ValueError(
                    f"samples[{i}] is missing required field '{required}'"
                )
        samples.append(SampleConfig(
            id=str(raw["id"]),
            language=str(raw["language"]),
            category=str(raw.get("category", "unknown")),
            audio=str(raw["audio"]),
            reference=str(raw["reference"]),
            description=str(raw.get("description", "")),
            min_accuracy={
                k: float(v)
                for k, v in (raw.get("min_accuracy") or {}).items()
            },
            max_dropped_chunks=int(raw.get("max_dropped_chunks", 0)),
            max_repetition_ratio=float(raw.get("max_repetition_ratio", 0.05)),
        ))
    return samples


def validate_sample_paths(
    samples: list[SampleConfig], base_dir: Path
) -> list[str]:
    """Return a list of error strings for missing audio/reference files."""
    errors: list[str] = []
    for s in samples:
        audio_path = base_dir / s.audio
        if not audio_path.is_file():
            errors.append(f"[{s.id}] audio file not found: {audio_path}")
        ref_path = base_dir / s.reference
        if not ref_path.is_file():
            errors.append(f"[{s.id}] reference file not found: {ref_path}")
    return errors


# ---------------------------------------------------------------------------
# Threshold checking
# ---------------------------------------------------------------------------

def check_thresholds(
    result: SampleResult,
    sample: SampleConfig,
) -> list[str]:
    """Return a list of failure reasons; empty means all thresholds met."""
    failures: list[str] = []
    # Accuracy: try category_model key first, then bare model name
    key = f"{sample.category}_{result.model}"
    min_acc = sample.min_accuracy.get(key) or sample.min_accuracy.get(result.model)
    if min_acc is not None and result.accuracy is not None:
        if result.accuracy < min_acc:
            failures.append(
                f"accuracy {result.accuracy:.3f} < min {min_acc:.3f} (key={key})"
            )
    if result.dropped_chunks > sample.max_dropped_chunks:
        failures.append(
            f"dropped_chunks {result.dropped_chunks} > max {sample.max_dropped_chunks}"
        )
    if result.repetition_ratio > sample.max_repetition_ratio:
        failures.append(
            f"repetition_ratio {result.repetition_ratio:.4f}"
            f" > max {sample.max_repetition_ratio:.4f}"
        )
    return failures


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

def aggregate_summary(results: list[SampleResult]) -> RunSummary:
    """Aggregate individual results into a RunSummary."""
    summary = RunSummary(total=len(results), results=list(results))
    for r in results:
        if r.error:
            summary.errors += 1
        elif r.passed:
            summary.passed += 1
        else:
            summary.failed += 1
    return summary


def write_summary_json(summary: RunSummary, output_dir: Path) -> Path:
    """Write summary.json and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "summary.json"
    data: dict[str, Any] = {
        "total": summary.total,
        "passed": summary.passed,
        "failed": summary.failed,
        "errors": summary.errors,
        "results": [
            {
                "sample_id": r.sample_id,
                "model": r.model,
                "mode": r.mode,
                "accuracy": r.accuracy,
                "dropped_chunks": r.dropped_chunks,
                "repetition_ratio": r.repetition_ratio,
                "passed": r.passed,
                "failure_reasons": r.failure_reasons,
                "error": r.error,
            }
            for r in summary.results
        ],
    }
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def write_summary_md(summary: RunSummary, output_dir: Path) -> Path:
    """Write summary.md and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "summary.md"
    lines: list[str] = [
        "# ASR Regression Summary",
        "",
        f"**Total:** {summary.total}  "
        f"**Passed:** {summary.passed}  "
        f"**Failed:** {summary.failed}  "
        f"**Errors:** {summary.errors}",
        "",
        "| Sample | Model | Mode | Accuracy | Drop | RepRatio | Status |",
        "|--------|-------|------|----------|------|----------|--------|",
    ]
    for r in summary.results:
        acc = f"{r.accuracy:.3f}" if r.accuracy is not None else "n/a"
        if r.error:
            status = "ERR"
        elif r.passed:
            status = "✓"
        else:
            status = "✗"
        lines.append(
            f"| {r.sample_id} | {r.model} | {r.mode} | {acc} | "
            f"{r.dropped_chunks} | {r.repetition_ratio:.4f} | {status} |"
        )
    failures = [r for r in summary.results if not r.passed and not r.error and r.failure_reasons]
    if failures:
        lines += ["", "## Failures", ""]
        for r in failures:
            lines.append(
                f"- **{r.sample_id} ({r.model}/{r.mode})**: "
                + "; ".join(r.failure_reasons)
            )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# ASR invocation (delegates to streaming_sim.py via subprocess)
# ---------------------------------------------------------------------------

def _run_sample(
    sample: SampleConfig,
    model: str,
    mode: str,
    speed: float,
    manifest_base: Path,
    quick: bool,
) -> SampleResult:
    """Run ASR on one sample and return a SampleResult."""
    result = SampleResult(sample_id=sample.id, model=model, mode=mode)
    audio_path = manifest_base / sample.audio
    ref_path = manifest_base / sample.reference
    sim_script = Path(__file__).parent / "streaming_sim.py"
    cmd = [
        sys.executable, str(sim_script),
        "--audio", str(audio_path),
        "--reference", str(ref_path),
        "--language", sample.language,
        "--model", model,
        "--mode", mode,
        "--speed", str(speed),
        "--output-format", "json",
    ]
    if quick:
        cmd.append("--quick")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            result.error = (
                f"streaming_sim exited {proc.returncode}: "
                + proc.stderr[:300].strip()
            )
            return result
        output = json.loads(proc.stdout)
        result.accuracy = float(output.get("accuracy", 0.0))
        result.dropped_chunks = int(output.get("dropped_chunks", 0))
        result.repetition_ratio = float(output.get("repetition_ratio", 0.0))
    except subprocess.TimeoutExpired:
        result.error = "timeout"
        return result
    except Exception as exc:
        result.error = str(exc)
        return result
    result.failure_reasons = check_thresholds(result, sample)
    result.passed = len(result.failure_reasons) == 0
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run ASR regression corpus and compare results against thresholds."
    )
    parser.add_argument(
        "--manifest",
        default="downloads/asr_regression/manifest.yaml",
        help="Path to manifest.yaml",
    )
    parser.add_argument(
        "--models",
        default="turbo",
        help="Comma-separated model identifiers (e.g. belle,turbo)",
    )
    parser.add_argument(
        "--modes",
        default="meeting,dialogue",
        help="Comma-separated modes (e.g. meeting,dialogue)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=8.0,
        help="Audio playback speed multiplier for streaming simulation",
    )
    parser.add_argument(
        "--output-dir",
        default="downloads/benchmark_results/asr_regression",
        help="Directory for summary.json and summary.md",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: shorter audio segments for faster local validation",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip samples whose audio/reference files are not found",
    )
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    try:
        manifest = load_manifest(manifest_path)
        samples = parse_samples(manifest)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    manifest_base = manifest_path.parent
    path_errors = validate_sample_paths(samples, manifest_base)
    if path_errors:
        if args.skip_missing:
            missing_ids = {e.split("]")[0].lstrip("[").strip() for e in path_errors}
            for err in path_errors:
                print(f"SKIP: {err}", file=sys.stderr)
            samples = [s for s in samples if s.id not in missing_ids]
        else:
            for err in path_errors:
                print(f"ERROR: {err}", file=sys.stderr)
            return 1

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    output_dir = Path(args.output_dir)

    combos = [(s, m1, m2) for s in samples for m1 in models for m2 in modes]
    total = len(combos)
    all_results: list[SampleResult] = []

    for i, (sample, model, mode) in enumerate(combos, 1):
        print(f"[{i}/{total}] {sample.id}  model={model}  mode={mode} ...", flush=True)
        result = _run_sample(sample, model, mode, args.speed, manifest_base, args.quick)
        if result.error:
            print(f"  ERROR: {result.error}")
        elif result.passed:
            acc = f"{result.accuracy:.3f}" if result.accuracy is not None else "n/a"
            print(f"  PASS  accuracy={acc}")
        else:
            print(f"  FAIL  {'; '.join(result.failure_reasons)}")
        all_results.append(result)

    summary = aggregate_summary(all_results)
    json_path = write_summary_json(summary, output_dir)
    md_path = write_summary_md(summary, output_dir)
    print(
        f"\nSummary: {summary.passed}/{summary.total} passed"
        f"  ({summary.failed} failed, {summary.errors} errors)"
    )
    print(f"Reports: {json_path}  {md_path}")
    return 0 if (summary.failed == 0 and summary.errors == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
