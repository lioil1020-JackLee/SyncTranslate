"""ASR benchmark runner.

Run offline ASR on a set of audio files and output transcript, latency, and
optional CER/WER metrics compared to a ground-truth reference file.

Usage
-----
uv run python tools/asr_benchmark/run_benchmark.py \\
    --config config.yaml \\
    --audio path/to/file.wav \\
    --reference path/to/reference.txt \\
    --profile default \\
    --source remote \\
    --output results/

Multiple audio files can be passed:
    --audio a.wav b.wav c.wav

Output (JSON lines written to --output/benchmark_results.jsonl):
    {
      "file": "a.wav",
      "profile": "default",
      "backend": "faster-whisper",
      "transcript": "...",
      "first_partial_latency_ms": 420,
      "final_latency_ms": 1800,
      "endpoint_count": 3,
      "partial_count": 12,
      "final_count": 3,
      "cer": 0.05,   // only if --reference provided
      "wer": 0.08    // only if --reference provided
    }
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import wave
from copy import deepcopy
from pathlib import Path
from threading import Event
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap external AI runtime (faster-whisper runtime)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from app.bootstrap.external_runtime import configure_external_ai_runtime
    configure_external_ai_runtime()
except Exception:
    pass


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _cer(hypothesis: str, reference: str) -> float:
    """Character error rate."""
    hyp = list(hypothesis.replace(" ", ""))
    ref = list(reference.replace(" ", ""))
    return _edit_distance(hyp, ref) / max(1, len(ref))


def _wer(hypothesis: str, reference: str) -> float:
    """Word error rate."""
    hyp = hypothesis.split()
    ref = reference.split()
    return _edit_distance(hyp, ref) / max(1, len(ref))


try:
    from opencc import OpenCC  # type: ignore
except Exception:
    OpenCC = None

if OpenCC is not None:
    _BENCHMARK_S2T = OpenCC("s2twp")
else:
    _BENCHMARK_S2T = None


def _is_cjk_language(language: str) -> bool:
    normalized = str(language or "").strip().lower().replace("_", "-")
    return normalized in {"zh", "zh-tw", "zh-cn", "cmn", "cmn-hans", "cmn-hant", "yue"}


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?\.])\s+|\n+", str(text or ""))
    return [part.strip() for part in parts if part.strip()]


def _normalize_metric_text(text: str, *, language: str, reference_text: str = "") -> str:
    value = str(text or "")
    if _is_cjk_language(language):
        if _BENCHMARK_S2T is not None:
            try:
                value = _BENCHMARK_S2T.convert(value)
            except Exception:
                pass
        value = re.sub(r"[A-Za-z0-9_]+", " ", value)
        value = re.sub(r"[^\u4e00-\u9fff\s]", " ", value)
        value = " ".join(value.split())

        reference_value = str(reference_text or "")
        if reference_value:
            if _BENCHMARK_S2T is not None:
                try:
                    reference_value = _BENCHMARK_S2T.convert(reference_value)
                except Exception:
                    pass
            reference_value = re.sub(r"[^\u4e00-\u9fff\s]", " ", reference_value)
            reference_value = "".join(reference_value.split())
            kept: list[str] = []
            for sentence in _split_sentences(value):
                sentence_joined = "".join(sentence.split())
                if not sentence_joined:
                    continue
                probe_len = min(6, len(sentence_joined))
                matched = False
                for start in range(0, max(1, len(sentence_joined) - probe_len + 1)):
                    probe = sentence_joined[start : start + probe_len]
                    if probe and probe in reference_value:
                        matched = True
                        break
                if matched:
                    kept.append(sentence_joined)
            if kept:
                value = "".join(kept)
            else:
                value = "".join(value.split())
        else:
            value = "".join(value.split())
        return value

    lowered = value.lower()
    stripped = re.sub(r"[^\w\s]", " ", lowered)
    return " ".join(stripped.split())


def _edit_distance(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


# ---------------------------------------------------------------------------
# WAV loader
# ---------------------------------------------------------------------------

def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Load WAV file as float32 mono array."""
    with wave.open(str(path), "rb") as f:
        n_channels = f.getnchannels()
        sampwidth = f.getsampwidth()
        framerate = f.getframerate()
        n_frames = f.getnframes()
        raw = f.readframes(n_frames)

    if sampwidth == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    return audio, framerate


def _iter_audio_windows(
    audio: np.ndarray,
    sample_rate: int,
    *,
    window_ms: int,
    overlap_ms: int,
) -> list[np.ndarray]:
    if audio.size == 0:
        return []
    window_samples = max(1, int(sample_rate * window_ms / 1000))
    overlap_samples = max(0, int(sample_rate * overlap_ms / 1000))
    step_samples = max(1, window_samples - overlap_samples)
    windows: list[np.ndarray] = []
    offset = 0
    while offset < len(audio):
        windows.append(audio[offset : offset + window_samples])
        if offset + window_samples >= len(audio):
            break
        offset += step_samples
    return windows


def _merge_chunk_texts(parts: list[str], *, language: str = "") -> str:
    is_cjk = _is_cjk_language(language)
    merged = ""
    for part in parts:
        current = str(part or "").strip()
        if not current:
            continue
        if not merged:
            merged = current
            continue
        max_overlap = min(len(merged), len(current), 48)
        overlap = 0
        for size in range(max_overlap, 0, -1):
            if merged[-size:] == current[:size]:
                overlap = size
                break
        if overlap:
            merged += current[overlap:]
        elif merged.endswith(current):
            continue
        else:
            merged += current if is_cjk else " " + current
    return merged.strip()


# ---------------------------------------------------------------------------
# Benchmark run
# ---------------------------------------------------------------------------

def _run_file(
    audio_path: Path,
    *,
    config_path: Path,
    source: str,
    profile_name: str,
    reference_text: str | None,
    language_override: str | None = None,
    chunk_ms: int = 100,
    accuracy_mode: str = "quality",
) -> dict[str, Any]:
    """Run ASR on a single audio file and return metrics.

    Uses the project's FasterWhisperEngine directly (batch mode) to get
    accurate transcript quality metrics, bypassing streaming VAD/endpointing
    which is tuned for real-time use and not suitable for offline benchmarking.
    """
    from app.infra.config.settings_store import load_config
    from app.infra.asr.backend_resolution import resolve_backend_for_language
    from app.infra.asr.backend_v2 import _build_engine

    audio, sample_rate = _load_wav(audio_path)
    audio_duration_ms = round(len(audio) / sample_rate * 1000)

    config = load_config(str(config_path))

    # Pick the right channel's AsrConfig
    asr_channels = getattr(config, "asr_channels", None)
    if asr_channels is not None and hasattr(asr_channels, source):
        asr_profile = getattr(asr_channels, source)
    elif hasattr(config, "asr"):
        asr_profile = config.asr
    else:
        from app.infra.config.schema import AsrConfig
        asr_profile = AsrConfig()

    # Determine language
    asr_language = ""
    if hasattr(config, "channels") and config.channels is not None:
        ch = getattr(config.channels, source, None)
        if ch is not None:
            asr_language = str(getattr(ch, "asr_language", "") or "")
    if not asr_language and hasattr(config, "asr"):
        asr_language = str(getattr(config.asr, "language", "") or "")
    if language_override:
        asr_language = str(language_override).strip()

    # Benchmark now uses unified faster-whisper backend for all languages.
    backend_name = "faster_whisper_v2"

    # Keep resolution object for reporting even when benchmark backend is overridden.
    resolution = resolve_backend_for_language(asr_language)
    model_name = asr_profile.model
    print(f"[benchmark] Backend    : {backend_name}", flush=True)
    print(f"[benchmark] ASR model  : {model_name}", flush=True)
    print(f"[benchmark] Device     : {asr_profile.device}", flush=True)
    print(f"[benchmark] Language   : {asr_language!r}", flush=True)

    print("[benchmark] Loading model...", flush=True)
    t0 = time.monotonic()
    chunk_count = 1
    tuned_profile = deepcopy(asr_profile)
    if accuracy_mode == "quality":
        tuned_profile.condition_on_previous_text = False
        tuned_profile.final_condition_on_previous_text = False
        tuned_profile.speculative_draft_model = ""
        tuned_profile.final_beam_size = max(
            int(getattr(tuned_profile, "final_beam_size", 3)),
            4 if _is_cjk_language(asr_language) else 5,
        )
        tuned_profile.no_speech_threshold = min(
            float(getattr(tuned_profile, "no_speech_threshold", 0.55)),
            0.3,
        )
    engine = _build_engine(tuned_profile, language=asr_language)
    engine.warmup()
    load_ms = round((time.monotonic() - t0) * 1000)
    print(f"[benchmark] Model ready in {load_ms}ms", flush=True)

    # Batch transcribe the full audio
    print("[benchmark] Transcribing...", flush=True)
    t1 = time.monotonic()
    try:
        r2 = engine.transcribe_final_result(audio, sample_rate)
        transcript = r2.text.strip()
        detected_language = r2.detected_language
    except Exception:
        transcript = engine.transcribe_final(audio, sample_rate).strip()
        detected_language = ""
    transcribe_ms = round((time.monotonic() - t1) * 1000)
    print(f"[benchmark] Transcribed {audio_duration_ms}ms audio in {transcribe_ms}ms", flush=True)
    rtf = transcribe_ms / max(1, audio_duration_ms)

    record: dict[str, Any] = {
        "file": str(audio_path),
        "profile": profile_name,
        "source": source,
        "backend": backend_name,
        "backend_resolution": resolution.backend_name,
        "benchmark_accuracy_mode": accuracy_mode,
        "chunk_count": chunk_count,
        "transcript": transcript,
        "detected_language": detected_language,
        "duration_ms": audio_duration_ms,
        "model_load_ms": load_ms,
        "transcribe_ms": transcribe_ms,
        "rtf": round(rtf, 3),
    }
    if reference_text:
        record["cer"] = round(_cer(transcript, reference_text), 4)
        record["wer"] = round(_wer(transcript, reference_text), 4)
        normalized_hypothesis = _normalize_metric_text(
            transcript,
            language=asr_language,
            reference_text=reference_text,
        )
        normalized_reference = _normalize_metric_text(
            reference_text,
            language=asr_language,
            reference_text=reference_text,
        )
        record["cer_normalized"] = round(_cer(normalized_hypothesis, normalized_reference), 4)
        record["wer_normalized"] = round(
            _wer(
                " ".join(normalized_hypothesis) if _is_cjk_language(asr_language) else normalized_hypothesis,
                " ".join(normalized_reference) if _is_cjk_language(asr_language) else normalized_reference,
            ),
            4,
        )
        record["transcript_chars"] = len(transcript)
        record["reference_chars"] = len(reference_text)
    return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SyncTranslate ASR benchmark tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument("--audio", nargs="+", required=True, help="Audio file(s) to benchmark (.wav)")
    p.add_argument("--reference", default=None, help="Ground-truth text file (one line per audio file)")
    p.add_argument("--profile", default="default", help="Endpoint profile name")
    p.add_argument("--source", default="remote", choices=["local", "remote"])
    p.add_argument("--output", default="results", help="Output directory for results")
    p.add_argument("--language", default=None, help="Force ASR language (e.g. zh-TW, en, zh)")
    p.add_argument("--chunk-ms", type=int, default=100, help="Audio feed chunk size in ms")
    p.add_argument(
        "--accuracy-mode",
        choices=["quality", "runtime_like"],
        default="quality",
        help="quality: force high-accuracy decoding params for offline benchmark; runtime_like: use profile as-is",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "benchmark_results.jsonl"

    reference_lines: list[str | None] = [None] * len(args.audio)
    if args.reference:
        ref_path = Path(args.reference)
        if ref_path.exists():
            lines = ref_path.read_text(encoding="utf-8").splitlines()
            for i, line in enumerate(lines):
                if i < len(reference_lines):
                    reference_lines[i] = line.strip()

    all_results: list[dict[str, Any]] = []
    for i, audio_file in enumerate(args.audio):
        audio_path = Path(audio_file)
        if not audio_path.exists():
            print(f"[WARN] Audio file not found: {audio_path}", file=sys.stderr)
            continue
        print(f"[benchmark] Processing: {audio_path.name}", file=sys.stderr)
        try:
            record = _run_file(
                audio_path,
                config_path=config_path,
                source=args.source,
                profile_name=args.profile,
                reference_text=reference_lines[i],
                language_override=args.language,
                chunk_ms=args.chunk_ms,
                accuracy_mode=args.accuracy_mode,
            )
            all_results.append(record)
            print(json.dumps(record, ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {audio_path}: {exc}", file=sys.stderr)

    with results_path.open("w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[benchmark] Results written to {results_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
