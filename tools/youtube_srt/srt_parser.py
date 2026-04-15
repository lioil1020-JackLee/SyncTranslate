"""SRT parser and text extractor for ASR benchmark reference.

Converts an SRT file into:
- plain text (for CER/WER computation)
- list of (start_ms, end_ms, text) segments (for per-segment comparison)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SrtSegment:
    index: int
    start_ms: int
    end_ms: int
    text: str


_TIMESTAMP_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)
# SRT auto-subtitle noise: <tags>, {tags}, speaker labels, formatting marks
_NOISE_RE = re.compile(r"<[^>]+>|\{[^}]+\}|\[.*?\]")


def _ts_to_ms(h: str, m: str, s: str, ms: str) -> int:
    return int(h) * 3_600_000 + int(m) * 60_000 + int(s) * 1_000 + int(ms)


def _clean(text: str) -> str:
    text = _NOISE_RE.sub("", text)
    text = re.sub(r"^WEBVTT\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bNOTE\b.*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_srt(path: str | Path) -> list[SrtSegment]:
    """Parse an SRT or VTT subtitle file and return SrtSegment objects."""
    content = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", content.strip())
    segments: list[SrtSegment] = []
    for block in blocks:
        lines = block.strip().splitlines()
        # First line: index (may be missing in some files)
        ts_line_idx = 0
        for i, line in enumerate(lines):
            if _TIMESTAMP_RE.search(line):
                ts_line_idx = i
                break
        else:
            continue
        m = _TIMESTAMP_RE.search(lines[ts_line_idx])
        if not m:
            continue
        start_ms = _ts_to_ms(*m.group(1, 2, 3, 4))
        end_ms = _ts_to_ms(*m.group(5, 6, 7, 8))
        text_lines = lines[ts_line_idx + 1:]
        if not text_lines:
            continue
        text = _clean(" ".join(text_lines))
        if not text:
            continue
        index = int(lines[0]) if lines[0].strip().isdigit() else len(segments) + 1
        segments.append(SrtSegment(index=index, start_ms=start_ms, end_ms=end_ms, text=text))
    return segments


def segments_to_text(segments: list[SrtSegment]) -> str:
    """Concatenate all segment texts into a single string (for full-file CER/WER)."""
    return " ".join(s.text for s in segments)


def find_srt(directory: str | Path, lang_hint: str = "") -> Path | None:
    """Find the first subtitle file in a directory, optionally filtered by lang_hint."""
    directory = Path(directory)
    subtitle_files = sorted(list(directory.glob("*.srt")) + list(directory.glob("*.vtt")))
    if not subtitle_files:
        return None
    if lang_hint:
        for p in subtitle_files:
            if lang_hint.lower() in p.name.lower():
                return p
    return subtitle_files[0]


__all__ = ["SrtSegment", "parse_srt", "segments_to_text", "find_srt"]
