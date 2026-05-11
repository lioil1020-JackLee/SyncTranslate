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
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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
    duration_sec: float = 0.0
    speaker_count: int = 0
    gender: str = ""
    noise_level: str = "unknown"
    speech_rate: str = "unknown"
    has_music: bool = False
    reference_quality: str = "unknown"
    notes: str = ""
    min_accuracy: dict[str, float] = field(default_factory=dict)
    max_dropped_chunks: int = 0
    max_repetition_ratio: float = 0.05
    max_missing_sentence_rate: float = 0.20
    max_duplicate_rate: float = 0.05
    max_hallucination_rate: float = 0.02
    max_average_final_latency_ms: float = 0.0


@dataclass
class SampleResult:
    sample_id: str
    model: str
    mode: str
    accuracy: float | None = None
    accuracy_dedup: float | None = None
    dropped_chunks: int = 0
    repetition_ratio: float = 0.0
    missing_sentence_rate: float = 0.0
    duplicate_rate: float = 0.0
    hallucination_rate: float = 0.0
    average_final_latency_ms: float = 0.0
    transcript: str = ""
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
            duration_sec=float(raw.get("duration_sec", 0.0) or 0.0),
            speaker_count=int(raw.get("speaker_count", 0) or 0),
            gender=str(raw.get("gender", "")),
            noise_level=str(raw.get("noise_level", "unknown")),
            speech_rate=str(raw.get("speech_rate", "unknown")),
            has_music=bool(raw.get("has_music", False)),
            reference_quality=str(raw.get("reference_quality", "unknown")),
            notes=str(raw.get("notes", "")),
            min_accuracy={
                k: float(v)
                for k, v in (raw.get("min_accuracy") or {}).items()
            },
            max_dropped_chunks=int(raw.get("max_dropped_chunks", 0)),
            max_repetition_ratio=float(raw.get("max_repetition_ratio", 0.05)),
            max_missing_sentence_rate=float(raw.get("max_missing_sentence_rate", 0.20)),
            max_duplicate_rate=float(raw.get("max_duplicate_rate", 0.05)),
            max_hallucination_rate=float(raw.get("max_hallucination_rate", 0.02)),
            max_average_final_latency_ms=float(raw.get("max_average_final_latency_ms", 0.0)),
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
    if result.missing_sentence_rate > sample.max_missing_sentence_rate:
        failures.append(
            f"missing_sentence_rate {result.missing_sentence_rate:.4f}"
            f" > max {sample.max_missing_sentence_rate:.4f}"
        )
    if result.duplicate_rate > sample.max_duplicate_rate:
        failures.append(
            f"duplicate_rate {result.duplicate_rate:.4f}"
            f" > max {sample.max_duplicate_rate:.4f}"
        )
    if result.hallucination_rate > sample.max_hallucination_rate:
        failures.append(
            f"hallucination_rate {result.hallucination_rate:.4f}"
            f" > max {sample.max_hallucination_rate:.4f}"
        )
    if sample.max_average_final_latency_ms > 0 and result.average_final_latency_ms > sample.max_average_final_latency_ms:
        failures.append(
            f"average_final_latency_ms {result.average_final_latency_ms:.0f}"
            f" > max {sample.max_average_final_latency_ms:.0f}"
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
                "accuracy_dedup": r.accuracy_dedup,
                "dropped_chunks": r.dropped_chunks,
                "repetition_ratio": r.repetition_ratio,
                "missing_sentence_rate": r.missing_sentence_rate,
                "duplicate_rate": r.duplicate_rate,
                "hallucination_rate": r.hallucination_rate,
                "average_final_latency_ms": r.average_final_latency_ms,
                "transcript": _json_safe_text(r.transcript),
                "passed": r.passed,
                "failure_reasons": r.failure_reasons,
                "error": r.error,
            }
            for r in summary.results
        ],
    }
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def _json_safe_text(value: str) -> str:
    text = str(value or "")
    return "".join(
        ch if ch in "\n\t" or ord(ch) >= 32 else " "
        for ch in text
    )


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
        "| Sample | Model | Mode | Accuracy | Dedup | Drop | RepRatio | Latency | Status |",
        "|--------|-------|------|----------|-------|------|----------|---------|--------|",
    ]
    for r in summary.results:
        acc = f"{r.accuracy:.3f}" if r.accuracy is not None else "n/a"
        acc_dedup = f"{r.accuracy_dedup:.3f}" if r.accuracy_dedup is not None else "n/a"
        if r.error:
            status = "ERR"
        elif r.passed:
            status = "✓"
        else:
            status = "✗"
        lines.append(
            f"| {r.sample_id} | {r.model} | {r.mode} | {acc} | {acc_dedup} | "
            f"{r.dropped_chunks} | {r.repetition_ratio:.4f} | "
            f"{r.average_final_latency_ms:.0f}ms | {status} |"
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
# ASR invocation
# ---------------------------------------------------------------------------

def _run_sample(
    sample: SampleConfig,
    model: str,
    mode: str,
    speed: float,
    manifest_base: Path,
    quick: bool,
    config_path: Path,
) -> SampleResult:
    """Run ASR on one sample and return a SampleResult."""
    result = SampleResult(sample_id=sample.id, model=model, mode=mode)
    audio_path = manifest_base / sample.audio
    ref_path = manifest_base / sample.reference
    model_override = _model_override_for_cli(model)
    try:
        from tools.asr_benchmark.streaming_sim import (
            _cer,
            _dedupe_repetition,
            _duplicate_rate,
            _load_wav,
            _missing_sentence_rate,
            _normalize,
            _repetition_ratio,
            _wer,
            run_streaming_sim,
        )

        audio, sample_rate = _load_wav(audio_path)
        if quick:
            max_samples = int(sample_rate * 45)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
        output = run_streaming_sim(
            audio,
            sample_rate,
            config_path=config_path,
            source="local",
            language=sample.language,
            asr_overrides={"model": model_override} if model_override else None,
            endpoint_profile=_endpoint_profile_for_mode(mode),
            speed_multiplier=speed,
            verbose=False,
        )
        reference_text = ref_path.read_text(encoding="utf-8").strip()
        if reference_text:
            hyp = _normalize(output["transcript"], lang=sample.language, ref=reference_text)
            ref = _normalize(reference_text, lang=sample.language, ref=reference_text)
            hyp_dedup = _dedupe_repetition(hyp, lang=sample.language)
            cer = round(_cer(hyp, ref), 4)
            cer_dedup = round(_cer(hyp_dedup, ref), 4)
            output["accuracy"] = round(1 - cer, 4)
            output["accuracy_dedup"] = round(1 - cer_dedup, 4)
            output["repetition_ratio"] = round(_repetition_ratio(hyp, lang=sample.language), 4)
            output["duplicate_rate"] = round(_duplicate_rate(output["transcript"], lang=sample.language), 4)
            output["missing_sentence_rate"] = round(
                _missing_sentence_rate(output["transcript"], reference_text, lang=sample.language),
                4,
            )
            if not sample.language.lower().startswith(("zh", "cmn", "yue")):
                output["wer_normalized"] = round(_wer(hyp, ref), 4)
        result.accuracy = float(output.get("accuracy", 0.0))
        if output.get("accuracy_dedup") is not None:
            result.accuracy_dedup = float(output.get("accuracy_dedup", 0.0))
        result.dropped_chunks = int(output.get("dropped_chunks", 0))
        result.repetition_ratio = float(output.get("repetition_ratio", 0.0))
        result.missing_sentence_rate = float(output.get("missing_sentence_rate", 0.0))
        result.duplicate_rate = float(output.get("duplicate_rate", result.repetition_ratio))
        result.hallucination_rate = float(output.get("hallucination_rate", 0.0))
        result.average_final_latency_ms = float(output.get("average_final_latency_ms", 0.0))
        result.transcript = str(output.get("transcript", ""))
    except Exception as exc:
        result.error = str(exc)
        return result
    result.failure_reasons = check_thresholds(result, sample)
    result.passed = len(result.failure_reasons) == 0
    return result


def _model_override_for_cli(model: str) -> str:
    """Map stable regression model labels to runtime model identifiers."""
    normalized = str(model or "").strip().lower()
    if normalized in {"default", "config"}:
        return ""
    if normalized in {"turbo", "large-v3-turbo"}:
        return "large-v3-turbo"
    if normalized in {"belle", "belle-zh-ct2"}:
        return r".\runtimes\models\belle-zh-ct2"
    return str(model or "").strip()


def _endpoint_profile_for_mode(mode: str) -> str:
    normalized = str(mode or "").strip().lower().replace("-", "_")
    aliases = {
        "meeting": "meeting_room",
        "meeting_room": "meeting_room",
        "dialogue": "turn_taking",
        "conversation": "turn_taking",
        "turn_taking": "turn_taking",
        "low_latency": "low_latency",
        "max_accuracy": "max_accuracy",
        "default": "default",
    }
    return aliases.get(normalized, str(mode or "").strip())


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
        "--config",
        default="config.yaml",
        help="Application config used by the streaming simulator.",
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
        result = _run_sample(
            sample,
            model,
            mode,
            args.speed,
            manifest_base,
            args.quick,
            Path(args.config),
        )
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
