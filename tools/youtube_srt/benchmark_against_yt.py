"""End-to-end ASR benchmark against YouTube auto-subtitles.

Downloads audio + SRT from a YouTube video, runs the project's ASR pipeline
on the audio, then computes CER / WER against the YouTube reference subtitles.

Usage
-----
# Chinese video
uv run python tools/youtube_srt/benchmark_against_yt.py \\
    --url https://www.youtube.com/watch?v=4K6tuF-O9t0 \\
    --source local --lang zh-TW --profile default

# English video  
uv run python tools/youtube_srt/benchmark_against_yt.py \\
    --url https://www.youtube.com/watch?v=3bayuw6b-qY \\
    --source remote --lang en --profile meeting_room

Requirements
------------
- yt-dlp      (uv pip install yt-dlp)
- ffmpeg      (in PATH)
- app modules (this project)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

# Ensure workspace root is importable
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import imageio_ffmpeg
from tools.youtube_srt.srt_parser import find_srt, parse_srt, segments_to_text

_FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], cwd: Path | None = None) -> int:
    print("[run]", " ".join(cmd))
    return subprocess.run(cmd, check=False, cwd=cwd).returncode


def _download_audio_and_srt(url: str, out_dir: Path, lang_pattern: str) -> tuple[Path | None, Path | None]:
    """Download audio (wav 16kHz mono) and SRT into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    template = str(out_dir / "%(title)s.%(ext)s")

    # Subtitle file (prefer srt, allow vtt fallback)
    _run([
        sys.executable, "-m", "yt_dlp",
        "--js-runtimes", "node",
        "--remote-components", "ejs:github",
        "--skip-download",
        "--write-auto-sub",
        "--sub-langs", lang_pattern,
        "--sub-format", "srt/vtt/best",
        "-o", template,
        url,
    ])

    # Audio
    rc = _run([
        sys.executable, "-m", "yt_dlp",
        "--js-runtimes", "node",
        "--remote-components", "ejs:github",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--ffmpeg-location", _FFMPEG,
        "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
        "-o", template,
        url,
    ])
    if rc != 0:
        print("[warn] audio download failed, trying best format fallback")

    wav_files = sorted(out_dir.glob("*.wav"))
    srt_file = find_srt(out_dir)
    return (wav_files[0] if wav_files else None), srt_file


def _run_asr_benchmark(
    audio_path: Path,
    reference_text: str,
    *,
    source: str,
    language: str,
    profile: str,
    chunk_ms: int,
    config_path: str,
) -> dict:
    """Run tools/asr_benchmark/run_benchmark.py and return the result dict."""
    from tools.asr_benchmark.run_benchmark import _run_file
    result = _run_file(
        audio_path,
        config_path=config_path,
        source=source,
        profile_name=profile,
        reference_text=reference_text,
        language_override=language,
        chunk_ms=chunk_ms,
    )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ASR benchmark against YouTube auto-subtitles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--url", required=True, help="YouTube URL")
    p.add_argument("--source", choices=["local", "remote"], default="remote")
    p.add_argument("--lang", default="en", help="ASR language (e.g. zh-TW, en)")
    p.add_argument(
        "--lang-pattern", default="",
        help="yt-dlp sub-langs pattern (default: derived from --lang)"
    )
    p.add_argument("--profile", default="default", help="Endpoint profile name")
    p.add_argument("--chunk-ms", type=int, default=40, help="Audio chunk size in ms")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument("--output", default=None, help="Write result JSON to this file")
    p.add_argument(
        "--download-only", action="store_true",
        help="Only download audio + SRT, do not run ASR"
    )
    p.add_argument(
        "--out-dir", default="downloads/benchmark",
        help="Directory to store downloaded files"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Derive sub-langs pattern
    lang_pattern = args.lang_pattern
    if not lang_pattern:
        base = args.lang.split("-")[0].lower()
        if base in ("zh", "cmn", "yue"):
            lang_pattern = args.lang if args.lang else "zh-TW"
        else:
            lang_pattern = f"{base}.*"

    out_dir = Path(args.out_dir) / args.lang
    print(f"\n[benchmark] URL      : {args.url}")
    print(f"[benchmark] Language : {args.lang}")
    print(f"[benchmark] Source   : {args.source}")
    print(f"[benchmark] Profile  : {args.profile}")
    print(f"[benchmark] Out dir  : {out_dir.resolve()}\n")

    # --- Download ----------------------------------------------------------------
    audio_path, srt_path = _download_audio_and_srt(args.url, out_dir, lang_pattern)

    if srt_path:
        print(f"\n[benchmark] SRT      : {srt_path}")
        segments = parse_srt(srt_path)
        reference_text = segments_to_text(segments)
        print(f"[benchmark] Segments : {len(segments)}")
        print(f"[benchmark] Ref len  : {len(reference_text)} chars")
    else:
        print("[warn] No SRT found; WER/CER will be skipped")
        reference_text = ""

    if audio_path:
        print(f"[benchmark] Audio    : {audio_path}")
    else:
        print("[error] No audio downloaded")
        return 1

    if args.download_only:
        print("\n[benchmark] --download-only: skipping ASR")
        return 0

    # --- ASR benchmark -----------------------------------------------------------
    print("\n[benchmark] Running ASR benchmark (this may take a while)...")
    try:
        result = _run_asr_benchmark(
            audio_path,
            reference_text,
            source=args.source,
            language=args.lang,
            profile=args.profile,
            chunk_ms=args.chunk_ms,
            config_path=args.config,
        )
    except Exception as exc:
        print(f"[error] ASR benchmark failed: {exc}")
        import traceback
        traceback.print_exc()
        return 1

    # --- Report ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    for k, v in result.items():
        print(f"  {k:<35}: {v}")
    print("=" * 60)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n[benchmark] Result written to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
