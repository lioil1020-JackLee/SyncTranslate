"""Multi-sample streaming ASR benchmark runner.

Iterates over a list of (wav, subtitle, language, source) samples,
runs streaming_sim for each, then prints a summary table.

Usage
-----
python tools/asr_benchmark/run_multi_benchmark.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

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
except Exception as e:
    print(f"[warn] external runtime setup failed: {e}", flush=True)

from tools.youtube_srt.srt_parser import find_srt, parse_srt, segments_to_text  # noqa: E402
from tools.asr_benchmark.streaming_sim import run_streaming_sim, _load_wav, _normalize, _cer, _wer, _dedupe_repetition, _is_cjk, _repetition_ratio  # noqa: E402

CONFIG_PATH = _ROOT / "config.yaml"
CHUNK_MS = 40
SPEED = 8.0
OUTPUT_DIR = _ROOT / "downloads" / "benchmark_results" / "multi_benchmark"

# ---------------------------------------------------------------------------
# Sample definitions: (label, wav_dir, lang_hint, language, source)
# ---------------------------------------------------------------------------
SAMPLES = [
    # --- Chinese ---
    (
        "zh_match_girl",
        _ROOT / "downloads/benchmark/zh-TW",
        "zh",
        "zh-TW",
        "local",
    ),
    (
        "zh_three_pigs",
        _ROOT / "downloads/benchmark/online_zh_three_pigs/zh-TW",
        "zh",
        "zh-TW",
        "local",
    ),
    (
        "zh_fruit_cow",
        _ROOT / "downloads/benchmark/online_zh_fruit_cow/zh-TW",
        "zh",
        "zh-TW",
        "local",
    ),
    # --- English ---
    (
        "en_reindeer",
        _ROOT / "downloads/benchmark/en",
        "en",
        "en",
        "remote",
    ),
    (
        "en_ai_feelings",
        _ROOT / "downloads/benchmark/online_en_ai_feelings/en",
        "en",
        "en",
        "remote",
    ),
    (
        "en_naps",
        _ROOT / "downloads/benchmark/online_en_naps/en",
        "en",
        "en",
        "remote",
    ),
    (
        "en_energy",
        _ROOT / "downloads/benchmark/online_en_energy/en",
        "en",
        "en",
        "remote",
    ),
]


def _find_wav(directory: Path) -> Path | None:
    wavs = list(directory.glob("*.wav"))
    return wavs[0] if wavs else None


def _subtitle_to_text(directory: Path, lang_hint: str) -> str:
    srt = find_srt(directory, lang_hint)
    if not srt:
        return ""
    segs = parse_srt(srt)
    return segments_to_text(segs)


_PRESET_PARAMS: dict[str, dict] = {
    # 超穩定會議字幕 / meeting_monitor  (非 dialogue 分支)
    "meeting_monitor": {
        "vad": {"min_silence_duration_ms": 640, "speech_pad_ms": 360},
        "streaming": {"soft_final_audio_ms": 4200, "final_history_seconds": 20},
        "asr": {"no_speech_threshold": 0.40},
    },
    # 低延遲雙向對話 / dialogue  (turbo)
    "dialogue": {
        "vad": {"min_silence_duration_ms": 280, "speech_pad_ms": 220},
        "streaming": {"soft_final_audio_ms": 1800, "final_history_seconds": 8},
        "asr": {"no_speech_threshold": 0.32},
    },
    # 低延遲雙向對話 / dialogue  (belle 疊加穩定化)
    "dialogue-belle": {
        "vad": {"min_silence_duration_ms": 320, "speech_pad_ms": 220},
        "streaming": {"soft_final_audio_ms": 2200, "final_history_seconds": 16},
        "asr": {"no_speech_threshold": 0.32, "beam_size": 2, "final_beam_size": 5},
    },
    # ========== 四組微調候選 (20260429) ==========
    # turbo_dialogue_opt1: 改為增加 soft_final (而非減少) + 保守調整
    # 基於第一個樣本結果: soft_final 1600 導致準確度下降，改為激進增加至 2000
    "turbo_dialogue_opt1": {
        "vad": {"min_silence_duration_ms": 280, "speech_pad_ms": 220},
        "streaming": {"soft_final_audio_ms": 2000, "final_history_seconds": 10},
        "asr": {"no_speech_threshold": 0.32},
    },
    "belle_dialogue_opt1": {
        "vad": {"min_silence_duration_ms": 310, "speech_pad_ms": 220},
        "streaming": {"soft_final_audio_ms": 2100, "final_history_seconds": 16},
        "asr": {"no_speech_threshold": 0.32, "beam_size": 2, "final_beam_size": 5},
    },
    "turbo_meeting_opt1": {
        "vad": {"min_silence_duration_ms": 600, "speech_pad_ms": 360},
        "streaming": {"soft_final_audio_ms": 4000, "final_history_seconds": 20},
        "asr": {"no_speech_threshold": 0.40},
    },
    "belle_meeting_opt1": {
        "vad": {"min_silence_duration_ms": 640, "speech_pad_ms": 360},
        "streaming": {"soft_final_audio_ms": 4100, "final_history_seconds": 20},
        "asr": {"no_speech_threshold": 0.40},
    },
}


def run_all(
    model_override: str | None = None,
    output_dir_override: Path | None = None,
    only_lang: str | None = None,
    preset: str | None = None,
) -> None:
    out_dir = output_dir_override or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []

    preset_params = _PRESET_PARAMS.get(preset or "", {})
    preset_vad = preset_params.get("vad")
    preset_streaming = preset_params.get("streaming")
    preset_asr = preset_params.get("asr")
    if preset:
        print(f"[batch] Preset: {preset}  vad={preset_vad}  streaming={preset_streaming}  asr={preset_asr}", flush=True)

    for label, wav_dir, lang_hint, language, source in SAMPLES:
        if only_lang and not str(language).lower().startswith(str(only_lang).lower()):
            continue
        print(f"\n{'='*60}", flush=True)
        print(f"[batch] Sample: {label}  lang={language}  source={source}", flush=True)

        wav = _find_wav(wav_dir)
        if not wav:
            print(f"[batch] SKIP – no wav in {wav_dir}", flush=True)
            continue

        ref_text = _subtitle_to_text(wav_dir, lang_hint)
        if not ref_text:
            print(f"[batch] WARNING – no subtitle found in {wav_dir}, CER will be skipped", flush=True)

        audio, sr = _load_wav(wav)
        print(f"[batch] Audio: {wav.name} ({len(audio)/sr:.1f}s)", flush=True)
        print(f"[batch] Reference: {len(ref_text)} chars", flush=True)

        asr_ov: dict = {}
        if model_override:
            asr_ov["model"] = model_override
        if preset_asr:
            asr_ov.update(preset_asr)

        try:
            result = run_streaming_sim(
                audio, sr,
                config_path=CONFIG_PATH,
                source=source,
                language=language,
                chunk_ms=CHUNK_MS,
                speed_multiplier=SPEED,
                asr_overrides=asr_ov or None,
                vad_overrides=preset_vad or None,
                streaming_overrides=preset_streaming or None,
                verbose=True,
            )
        except Exception as exc:
            print(f"[batch] ERROR: {exc}", flush=True)
            import traceback
            traceback.print_exc()
            continue

        # Compute metrics against reference
        if ref_text:
            hyp = _normalize(result["transcript"], lang=language, ref=ref_text)
            ref = _normalize(ref_text, lang=language, ref=ref_text)
            hyp_dedup = _dedupe_repetition(hyp, lang=language)
            cer = round(_cer(hyp, ref), 4)
            cer_dedup = round(_cer(hyp_dedup, ref), 4)
            rep = round(_repetition_ratio(hyp, lang=language), 4)
            result["cer_normalized"] = cer
            result["accuracy"] = round(1 - cer, 4)
            result["cer_dedup"] = cer_dedup
            result["accuracy_dedup"] = round(1 - cer_dedup, 4)
            result["repetition_ratio"] = rep
            if not _is_cjk(language):
                result["wer_normalized"] = round(_wer(hyp, ref), 4)

        result["label"] = label
        all_results.append(result)

        # Per-sample save
        out_file = out_dir / f"{label}.json"
        out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[batch] Saved {out_file.name}", flush=True)
        if "accuracy" in result:
            print(
                f"[batch] >>> CER={result['cer_normalized']:.3f}  "
                f"accuracy={result['accuracy']:.1%}  "
                f"dedup_acc={result['accuracy_dedup']:.1%}  "
                f"rep={result['repetition_ratio']:.1%}  "
                f"finals={result['final_count']}",
                flush=True,
            )

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("[batch] SUMMARY", flush=True)
    print(f"{'Label':<25} {'Lang':<6} {'Finals':>6} {'Accuracy':>9} {'DedupAcc':>9} {'Rep%':>6} {'WER':>7}", flush=True)
    print("-" * 70, flush=True)
    zh_accs, en_accs = [], []
    for r in all_results:
        acc = r.get("accuracy", float("nan"))
        dedup_acc = r.get("accuracy_dedup", float("nan"))
        rep = r.get("repetition_ratio", float("nan"))
        wer = r.get("wer_normalized", float("nan"))
        lang = r.get("language", "?")
        finals = r.get("final_count", 0)
        print(
            f"{r['label']:<25} {lang:<6} {finals:>6} {acc:>8.1%} {dedup_acc:>9.1%} "
            f"{rep:>5.1%} {('n/a' if wer != wer else f'{wer:.3f}'):>7}",
            flush=True,
        )
        if acc == acc:
            if lang.startswith("zh"):
                zh_accs.append(acc)
            else:
                en_accs.append(acc)

    if zh_accs:
        print(f"\n[batch] Chinese avg accuracy : {sum(zh_accs)/len(zh_accs):.1%}  (n={len(zh_accs)})", flush=True)
    if en_accs:
        print(f"[batch] English avg accuracy : {sum(en_accs)/len(en_accs):.1%}  (n={len(en_accs)})", flush=True)

    summary_file = out_dir / "summary.json"
    summary_file.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[batch] Full summary saved to {summary_file}", flush=True)


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--model", default=None, help="Override ASR model path for all samples")
    _ap.add_argument("--output-dir", default=None, help="Output directory override")
    _ap.add_argument("--only-lang", default=None, help="Run only samples whose language starts with this prefix (e.g. zh, en)")
    _ap.add_argument(
        "--preset", default=None,
        choices=list(_PRESET_PARAMS.keys()),
        help="Simulate experience preset VAD/streaming params: meeting_monitor | dialogue | dialogue-belle",
    )
    _args = _ap.parse_args()
    run_all(
        model_override=_args.model,
        output_dir_override=Path(_args.output_dir) if _args.output_dir else None,
        only_lang=_args.only_lang,
        preset=_args.preset,
    )
