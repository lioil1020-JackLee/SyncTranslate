from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.infra.config.settings_store import load_config, save_config
from tools.asr_benchmark.run_benchmark import _run_file


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FunASR parameter sweep for Chinese accuracy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument("--audio", required=True, help="Audio file path")
    p.add_argument("--reference", required=True, help="Reference transcript text file")
    p.add_argument("--source", choices=["local", "remote"], default="local")
    p.add_argument("--language", default="zh-TW")
    p.add_argument("--output", default="downloads/benchmark_results/funasr_tuning_loop.json")
    return p


def _candidate_grid() -> list[dict[str, Any]]:
    return [
        {
            "name": "baseline_itn_on",
            "use_itn": True,
            "suppress_low_confidence_short": True,
            "min_speech_ratio_for_short_text": 0.12,
            "short_text_max_chars": 10,
            "batch_size_s_offline": 0.0,
            "window_ms": 15000,
            "overlap_ms": 400,
        },
        {
            "name": "recall_bias_1",
            "use_itn": True,
            "suppress_low_confidence_short": False,
            "min_speech_ratio_for_short_text": 0.08,
            "short_text_max_chars": 14,
            "batch_size_s_offline": 0.0,
            "window_ms": 15000,
            "overlap_ms": 400,
        },
        {
            "name": "recall_bias_2_longer_window",
            "use_itn": True,
            "suppress_low_confidence_short": False,
            "min_speech_ratio_for_short_text": 0.08,
            "short_text_max_chars": 14,
            "batch_size_s_offline": 0.0,
            "window_ms": 22000,
            "overlap_ms": 900,
        },
        {
            "name": "itn_off",
            "use_itn": False,
            "suppress_low_confidence_short": False,
            "min_speech_ratio_for_short_text": 0.08,
            "short_text_max_chars": 14,
            "batch_size_s_offline": 0.0,
            "window_ms": 15000,
            "overlap_ms": 400,
        },
        {
            "name": "small_overlap",
            "use_itn": True,
            "suppress_low_confidence_short": False,
            "min_speech_ratio_for_short_text": 0.08,
            "short_text_max_chars": 14,
            "batch_size_s_offline": 0.0,
            "window_ms": 12000,
            "overlap_ms": 200,
        },
        {
            "name": "large_overlap",
            "use_itn": True,
            "suppress_low_confidence_short": False,
            "min_speech_ratio_for_short_text": 0.08,
            "short_text_max_chars": 14,
            "batch_size_s_offline": 0.0,
            "window_ms": 12000,
            "overlap_ms": 1200,
        },
        {
            "name": "batch30",
            "use_itn": True,
            "suppress_low_confidence_short": False,
            "min_speech_ratio_for_short_text": 0.08,
            "short_text_max_chars": 14,
            "batch_size_s_offline": 30.0,
            "window_ms": 15000,
            "overlap_ms": 400,
        },
        {
            "name": "batch60",
            "use_itn": True,
            "suppress_low_confidence_short": False,
            "min_speech_ratio_for_short_text": 0.08,
            "short_text_max_chars": 14,
            "batch_size_s_offline": 60.0,
            "window_ms": 15000,
            "overlap_ms": 400,
        },
    ]


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")
    reference_path = Path(args.reference)
    if not reference_path.exists():
        raise FileNotFoundError(f"reference not found: {reference_path}")
    reference_text = reference_path.read_text(encoding="utf-8").strip()

    base_cfg = load_config(args.config)
    candidates = _candidate_grid()
    results: list[dict[str, Any]] = []

    for idx, candidate in enumerate(candidates, start=1):
        cfg = load_config(args.config)
        asr_cfg = cfg.asr_channels.remote if args.source == "remote" else cfg.asr_channels.local
        asr_cfg.engine = "funasr"
        asr_cfg.funasr.use_itn = bool(candidate["use_itn"])
        asr_cfg.funasr.suppress_low_confidence_short = bool(candidate["suppress_low_confidence_short"])
        asr_cfg.funasr.min_speech_ratio_for_short_text = float(candidate["min_speech_ratio_for_short_text"])
        asr_cfg.funasr.short_text_max_chars = int(candidate["short_text_max_chars"])
        asr_cfg.funasr.batch_size_s_offline = float(candidate["batch_size_s_offline"])
        asr_cfg.funasr.benchmark_window_ms = int(candidate["window_ms"])
        asr_cfg.funasr.benchmark_overlap_ms = int(candidate["overlap_ms"])

        with tempfile.TemporaryDirectory() as td:
            tmp_config = Path(td) / "config.yaml"
            save_config(cfg, tmp_config)
            record = _run_file(
                audio_path,
                config_path=tmp_config,
                source=args.source,
                profile_name=f"funasr_tune_{idx}",
                reference_text=reference_text,
                language_override=args.language,
                accuracy_mode="quality",
                funasr_window_ms=int(candidate["window_ms"]),
                funasr_overlap_ms=int(candidate["overlap_ms"]),
            )

        record["candidate"] = candidate
        cer_norm = float(record.get("cer_normalized", record.get("cer", 1.0)))
        record["normalized_accuracy"] = round(1.0 - cer_norm, 4)
        print(
            f"[{idx}/{len(candidates)}] {candidate['name']} -> "
            f"acc={record['normalized_accuracy']:.4f}, cer_norm={cer_norm:.4f}"
        )
        results.append(record)

    ranked = sorted(results, key=lambda x: float(x.get("cer_normalized", x.get("cer", 1.0))))
    payload = {
        "audio": str(audio_path),
        "reference": str(reference_path),
        "source": args.source,
        "language": args.language,
        "baseline_config": args.config,
        "candidate_count": len(candidates),
        "best": ranked[0] if ranked else None,
        "results": ranked,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote tuning results: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
