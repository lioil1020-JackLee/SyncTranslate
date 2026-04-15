"""Download Chinese (zh-TW) YouTube auto-subtitles and audio.

Video: https://www.youtube.com/watch?v=4K6tuF-O9t0

Usage
-----
uv run python tools/youtube_srt/download_chinese.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg  # bundled ffmpeg — no system install needed


YOUTUBE_URL = "https://www.youtube.com/watch?v=4K6tuF-O9t0"
OUTPUT_DIR = Path("downloads/chinese_srt")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def run_command(command: list[str], *, fatal: bool = True) -> int:
    print("Running:", " ".join(command))
    result = subprocess.run(command, check=False)
    if result.returncode != 0 and fatal:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")
    return result.returncode


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # List available subtitles (non-fatal)
    run_command([
        sys.executable, "-m", "yt_dlp",
        "--js-runtimes", "node",
        "--remote-components", "ejs:github",
        "--list-subs",
        YOUTUBE_URL,
    ], fatal=False)

    # Download exact zh-TW auto-subtitles. Some videos only expose VTT, so allow fallback.
    run_command([
        sys.executable, "-m", "yt_dlp",
        "--js-runtimes", "node",
        "--remote-components", "ejs:github",
        "--skip-download",
        "--write-auto-sub",
        "--sub-langs", "zh-TW",
        "--sub-format", "srt/vtt/best",
        "-o", str(OUTPUT_DIR / "%(title)s.%(ext)s"),
        YOUTUBE_URL,
    ])

    # Download audio and convert to 16kHz mono WAV via bundled ffmpeg
    run_command([
        sys.executable, "-m", "yt_dlp",
        "--js-runtimes", "node",
        "--remote-components", "ejs:github",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--ffmpeg-location", FFMPEG,
        "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
        "-o", str(OUTPUT_DIR / "%(title)s.%(ext)s"),
        YOUTUBE_URL,
    ])

    print(f"\nDone. Check output folder: {OUTPUT_DIR.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
