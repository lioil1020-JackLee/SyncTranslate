"""Run a Chinese ASR preset/model matrix over exact local benchmark samples.

This runner compares the two user-facing experience presets:

- meeting: stable segmentation for longer speech.
- dialogue: lower latency for shorter turns.

Each mode is tested with both Chinese ASR model choices currently exposed in
the settings UI: belle-zh-ct2 and large-v3-turbo.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    from app.bootstrap.external_runtime import configure_external_ai_runtime

    configure_external_ai_runtime()
except Exception as exc:
    print(f"[warn] external runtime setup failed: {exc}", flush=True)

from tools.asr_benchmark.streaming_sim import (  # noqa: E402
    _cer,
    _dedupe_repetition,
    _load_wav,
    _normalize,
    _repetition_ratio,
    run_streaming_sim,
)
from tools.youtube_srt.srt_parser import parse_srt, segments_to_text  # noqa: E402


BELLE_MODEL = r".\runtimes\models\belle-zh-ct2"
TURBO_MODEL = "large-v3-turbo"


@dataclass(frozen=True, slots=True)
class MatrixSample:
    label: str
    audio: Path
    reference: Path


SAMPLES: tuple[MatrixSample, ...] = (
    MatrixSample(
        label="zh_match_girl",
        audio=_ROOT / "downloads/benchmark/zh-TW/【經典故事】安徒生童話 ：賣火柴的小女孩  #兒童故事｜小行星樂樂TV.wav",
        reference=_ROOT / "downloads/benchmark/zh-TW/【經典故事】安徒生童話 ：賣火柴的小女孩  #兒童故事｜小行星樂樂TV.zh-TW.vtt",
    ),
    MatrixSample(
        label="zh_three_pigs",
        audio=_ROOT / "downloads/benchmark/online_zh_three_pigs/zh-TW/【經典故事】世界童話：三隻小豬｜小行星樂樂TV.wav",
        reference=_ROOT / "downloads/benchmark/online_zh_three_pigs/zh-TW/【經典故事】世界童話：三隻小豬｜小行星樂樂TV.zh-TW.vtt",
    ),
    MatrixSample(
        label="zh_fruit_cow",
        audio=_ROOT / "downloads/benchmark/online_zh_fruit_cow/zh-TW/【 佳佳老師說故事 】EP112《 愛吃水果的牛 》｜兒童故事繪本｜幼兒睡前故事.wav",
        reference=_ROOT / "downloads/benchmark/online_zh_fruit_cow/zh-TW/【 佳佳老師說故事 】EP112《 愛吃水果的牛 》｜兒童故事繪本｜幼兒睡前故事.zh-TW.vtt",
    ),
    MatrixSample(
        label="zh_kaidan5",
        audio=_ROOT / "downloads/chinese_srt/kaidan_5min.wav",
        reference=_ROOT / "downloads/benchmark_results/chinese_kaidan_5min_reference.txt",
    ),
    MatrixSample(
        label="zh_kaidan5_denoise",
        audio=_ROOT / "downloads/chinese_srt/kaidan_5min_denoise.wav",
        reference=_ROOT / "downloads/benchmark_results/chinese_kaidan_5min_reference.txt",
    ),
)


MODE_MODEL_PARAMS: dict[tuple[str, str], dict[str, dict[str, Any]]] = {
    ("meeting", "belle"): {
        "asr": {"model": BELLE_MODEL, "beam_size": 2, "final_beam_size": 5, "no_speech_threshold": 0.40},
        "vad": {"min_silence_duration_ms": 600, "speech_pad_ms": 360},
        "streaming": {
            "soft_final_audio_ms": 4000,
            "final_history_seconds": 20,
            "partial_interval_ms": 520,
            "pre_roll_ms": 360,
            "min_partial_audio_ms": 360,
        },
    },
    ("meeting", "turbo"): {
        "asr": {"model": TURBO_MODEL, "beam_size": 1, "final_beam_size": 4, "no_speech_threshold": 0.40},
        "vad": {"min_silence_duration_ms": 600, "speech_pad_ms": 360},
        "streaming": {
            "soft_final_audio_ms": 4000,
            "final_history_seconds": 20,
            "partial_interval_ms": 520,
            "pre_roll_ms": 360,
            "min_partial_audio_ms": 360,
        },
    },
    ("dialogue", "belle"): {
        "asr": {"model": BELLE_MODEL, "beam_size": 2, "final_beam_size": 5, "no_speech_threshold": 0.32},
        "vad": {"min_silence_duration_ms": 320, "speech_pad_ms": 220},
        "streaming": {
            "soft_final_audio_ms": 2200,
            "final_history_seconds": 16,
            "partial_interval_ms": 240,
            "pre_roll_ms": 180,
            "min_partial_audio_ms": 220,
        },
    },
    ("dialogue", "turbo"): {
        "asr": {"model": TURBO_MODEL, "beam_size": 1, "final_beam_size": 4, "no_speech_threshold": 0.32},
        "vad": {"min_silence_duration_ms": 280, "speech_pad_ms": 220},
        "streaming": {
            "soft_final_audio_ms": 1800,
            "final_history_seconds": 10,
            "partial_interval_ms": 240,
            "pre_roll_ms": 180,
            "min_partial_audio_ms": 220,
        },
    },
}


def _read_reference(path: Path) -> str:
    if path.suffix.lower() in {".srt", ".vtt"}:
        return segments_to_text(parse_srt(path))
    return path.read_text(encoding="utf-8-sig", errors="replace").strip()


def _metrics(result: dict[str, Any], reference_text: str) -> dict[str, Any]:
    hyp = _normalize(str(result.get("transcript", "")), lang="zh-TW", ref=reference_text)
    ref = _normalize(reference_text, lang="zh-TW", ref=reference_text)
    hyp_dedup = _dedupe_repetition(hyp, lang="zh-TW")
    cer = round(_cer(hyp, ref), 4)
    cer_dedup = round(_cer(hyp_dedup, ref), 4)
    return {
        "cer_normalized": cer,
        "accuracy": round(1.0 - cer, 4),
        "cer_dedup": cer_dedup,
        "accuracy_dedup": round(1.0 - cer_dedup, 4),
        "repetition_ratio": round(_repetition_ratio(hyp, lang="zh-TW"), 4),
        "reference_chars": len(ref),
        "hypothesis_chars": len(hyp),
    }


def _selected_values(raw: str, *, allowed: set[str], default: tuple[str, ...]) -> list[str]:
    if not raw.strip():
        return list(default)
    values = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [value for value in values if value not in allowed]
    if unknown:
        raise ValueError(f"Unknown value(s): {', '.join(unknown)}")
    return values


def _selected_samples(raw: str) -> list[MatrixSample]:
    if not raw.strip():
        return list(SAMPLES)
    labels = [part.strip() for part in raw.split(",") if part.strip()]
    by_label = {sample.label: sample for sample in SAMPLES}
    missing = [label for label in labels if label not in by_label]
    if missing:
        raise ValueError(f"Unknown sample(s): {', '.join(missing)}")
    return [by_label[label] for label in labels]


def run_matrix(
    *,
    config: Path,
    output_dir: Path,
    modes: list[str],
    models: list[str],
    samples: list[MatrixSample],
    speed: float,
    chunk_ms: int,
    queue_maxsize: int,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for mode in modes:
        for model_key in models:
            params = MODE_MODEL_PARAMS[(mode, model_key)]
            combo_label = f"{mode}_{model_key}"
            print(f"\n[matrix] combo={combo_label}", flush=True)
            combo_dir = output_dir / combo_label
            combo_dir.mkdir(parents=True, exist_ok=True)
            for sample in samples:
                if not sample.audio.exists() or not sample.reference.exists():
                    print(f"[matrix] skip {sample.label}: missing audio/reference", flush=True)
                    continue
                reference_text = _read_reference(sample.reference)
                audio, sample_rate = _load_wav(sample.audio)
                print(
                    f"[matrix] sample={sample.label} duration={len(audio) / max(1, sample_rate):.1f}s",
                    flush=True,
                )
                result = run_streaming_sim(
                    audio,
                    sample_rate,
                    config_path=config,
                    source="local",
                    language="zh-TW",
                    chunk_ms=chunk_ms,
                    speed_multiplier=speed,
                    asr_overrides=params["asr"],
                    vad_overrides=params["vad"],
                    streaming_overrides=params["streaming"],
                    worker_overrides={"queue_maxsize": queue_maxsize},
                    verbose=True,
                )
                result.update(_metrics(result, reference_text))
                result.update(
                    {
                        "label": sample.label,
                        "mode": mode,
                        "model_key": model_key,
                        "model": params["asr"]["model"],
                        "queue_maxsize": queue_maxsize,
                    }
                )
                results.append(result)
                (combo_dir / f"{sample.label}.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(
                    "[matrix] "
                    f"acc={result['accuracy']:.1%} dedup={result['accuracy_dedup']:.1%} "
                    f"cer={result['cer_normalized']:.3f} rep={result['repetition_ratio']:.1%} "
                    f"finals={result.get('final_count', 0)} drops={result.get('dropped_chunks', 0)}",
                    flush=True,
                )
    _write_summary(output_dir, results)
    return results


def _write_summary(output_dir: Path, results: list[dict[str, Any]]) -> None:
    by_combo: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_combo.setdefault(f"{result['mode']}_{result['model_key']}", []).append(result)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "results": results,
        "combo_summary": [],
        "best_by_sample": [],
    }
    for combo, rows in sorted(by_combo.items()):
        if not rows:
            continue
        summary["combo_summary"].append(
            {
                "combo": combo,
                "samples": len(rows),
                "avg_accuracy": round(sum(float(r["accuracy"]) for r in rows) / len(rows), 4),
                "avg_dedup_accuracy": round(sum(float(r["accuracy_dedup"]) for r in rows) / len(rows), 4),
                "avg_repetition_ratio": round(sum(float(r["repetition_ratio"]) for r in rows) / len(rows), 4),
                "total_dropped_chunks": sum(int(r.get("dropped_chunks", 0) or 0) for r in rows),
                "avg_final_count": round(sum(int(r.get("final_count", 0) or 0) for r in rows) / len(rows), 2),
            }
        )

    labels = sorted({str(r["label"]) for r in results})
    for label in labels:
        rows = [r for r in results if r["label"] == label]
        best = max(
            rows,
            key=lambda r: (
                float(r.get("accuracy_dedup", 0.0)),
                float(r.get("accuracy", 0.0)),
                -float(r.get("repetition_ratio", 1.0)),
            ),
        )
        summary["best_by_sample"].append(
            {
                "label": label,
                "best_combo": f"{best['mode']}_{best['model_key']}",
                "accuracy": best["accuracy"],
                "accuracy_dedup": best["accuracy_dedup"],
                "repetition_ratio": best["repetition_ratio"],
            }
        )

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[matrix] summary={output_dir / 'summary.json'}", flush=True)
    print("[matrix] combo summary:", flush=True)
    for row in summary["combo_summary"]:
        print(
            f"  {row['combo']:<16} avg={row['avg_accuracy']:.1%} "
            f"dedup={row['avg_dedup_accuracy']:.1%} rep={row['avg_repetition_ratio']:.1%} "
            f"drops={row['total_dropped_chunks']}",
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Chinese ASR preset/model matrix benchmarks.")
    parser.add_argument("--config", default=str(_ROOT / "config.yaml"))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--modes", default="meeting,dialogue", help="Comma list: meeting,dialogue")
    parser.add_argument("--models", default="belle,turbo", help="Comma list: belle,turbo")
    parser.add_argument("--samples", default="", help="Comma list of sample labels; empty means all.")
    parser.add_argument("--speed", type=float, default=8.0)
    parser.add_argument("--chunk-ms", type=int, default=40)
    parser.add_argument("--queue-maxsize", type=int, default=256)
    args = parser.parse_args(argv)

    modes = _selected_values(args.modes, allowed={"meeting", "dialogue"}, default=("meeting", "dialogue"))
    models = _selected_values(args.models, allowed={"belle", "turbo"}, default=("belle", "turbo"))
    samples = _selected_samples(args.samples)
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else _ROOT / "downloads" / "benchmark_results" / f"zh_preset_matrix_{datetime.now():%Y%m%d_%H%M%S}"
    )
    run_matrix(
        config=Path(args.config),
        output_dir=out_dir,
        modes=modes,
        models=models,
        samples=samples,
        speed=float(args.speed),
        chunk_ms=int(args.chunk_ms),
        queue_maxsize=int(args.queue_maxsize),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
