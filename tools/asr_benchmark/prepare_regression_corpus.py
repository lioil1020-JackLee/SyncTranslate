"""Prepare a small ASR regression corpus from public datasets.

The generated corpus is intentionally written under ``downloads/`` because
audio samples can be large and are not committed to the repository.

Run with:

    uv run --with datasets==2.18.0 --with librosa --with soundfile --with numpy \
        python tools/asr_benchmark/prepare_regression_corpus.py --force
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


@dataclass(frozen=True)
class CorpusItem:
    id: str
    language: str
    category: str
    text: str
    audio: np.ndarray
    sample_rate: int
    source_dataset: str
    source_config: str
    source_split: str
    description: str
    speaker_count: int
    gender: str = "unknown"
    noise_level: str = "low"
    speech_rate: str = "normal"
    has_music: bool = False
    reference_quality: str = "dataset_transcript"
    notes: str = ""


_CJK_LANGS = {"zh", "zh-CN", "zh-TW", "cmn"}


def _safe_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _clean_text(value: Any) -> str:
    text = str(value or "").strip()
    text = " ".join(text.replace("\n", " ").replace("\r", " ").split())
    return text


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _has_latin(text: str) -> bool:
    return any(("a" <= ch.lower() <= "z") for ch in text)


def _to_float32_mono(audio: Any) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    if arr.size == 0:
        return arr
    peak = float(np.max(np.abs(arr)))
    if peak > 1.5:
        arr = arr / max(peak, 1.0)
    return np.clip(arr, -1.0, 1.0)


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16.tobytes())


def _iter_fleurs(config: str, split: str = "test") -> Iterable[dict[str, Any]]:
    load_dataset = _load_dataset_function()
    return load_dataset(
        "google/fleurs",
        config,
        split=split,
        streaming=True,
        trust_remote_code=True,
    )


def _iter_ascend(split: str = "train") -> Iterable[dict[str, Any]]:
    load_dataset = _load_dataset_function()
    return load_dataset("CAiRE/ASCEND", split=split, streaming=True)


def _iter_multidialog(split: str = "train") -> Iterable[dict[str, Any]]:
    load_dataset = _load_dataset_function()
    return load_dataset(
        "IVLLab/MultiDialog",
        "train",
        split=split,
        streaming=True,
        trust_remote_code=True,
    )


def _load_dataset_function():
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - exercised by CLI environment
        raise SystemExit(
            "The 'datasets' package is required. Run with:\n"
            "  uv run --with datasets==2.18.0 --with librosa --with soundfile --with numpy "
            "python tools/asr_benchmark/prepare_regression_corpus.py"
        ) from exc
    return load_dataset


def _take_fleurs(
    *,
    config: str,
    language: str,
    prefix: str,
    categories: list[str],
    count: int,
    min_seconds: float = 2.0,
) -> list[CorpusItem]:
    items: list[CorpusItem] = []
    for row in _iter_fleurs(config):
        text = _clean_text(row.get("transcription"))
        audio_info = row.get("audio") or {}
        audio = _to_float32_mono(audio_info.get("array"))
        sample_rate = int(audio_info.get("sampling_rate") or 16000)
        duration = len(audio) / max(sample_rate, 1)
        if not text or duration < min_seconds:
            continue
        category = categories[len(items) % len(categories)]
        item_id = f"{prefix}_{len(items) + 1:03d}"
        items.append(CorpusItem(
            id=item_id,
            language=language,
            category=category,
            text=text,
            audio=audio,
            sample_rate=sample_rate,
            source_dataset="google/fleurs",
            source_config=config,
            source_split="test",
            description=f"FLEURS {language} read speech sample ({category}).",
            speaker_count=1,
            notes="Public FLEURS dataset sample; useful for clean read-speech ASR regression.",
        ))
        if len(items) >= count:
            return items
    raise RuntimeError(f"Not enough FLEURS samples for {config}: requested {count}, got {len(items)}")


def _take_ascend(
    *,
    language_filter: str,
    language: str,
    prefix: str,
    category: str,
    count: int,
    require_mixed: bool = False,
) -> list[CorpusItem]:
    items: list[CorpusItem] = []
    for row in _iter_ascend():
        row_lang = str(row.get("language") or "").strip().lower()
        text = _clean_text(row.get("transcription"))
        if language_filter and row_lang != language_filter:
            continue
        if require_mixed and not (_has_cjk(text) and _has_latin(text)):
            continue
        audio_info = row.get("audio") or {}
        audio = _to_float32_mono(audio_info.get("array"))
        sample_rate = int(audio_info.get("sampling_rate") or 16000)
        if len(audio) / max(sample_rate, 1) < 1.0 or not text:
            continue
        item_id = f"{prefix}_{len(items) + 1:03d}"
        items.append(CorpusItem(
            id=item_id,
            language=language,
            category=category,
            text=text,
            audio=audio,
            sample_rate=sample_rate,
            source_dataset="CAiRE/ASCEND",
            source_config="default",
            source_split="train",
            description="ASCEND spontaneous Mandarin-English conversation sample.",
            speaker_count=2,
            speech_rate="fast" if require_mixed else "normal",
            reference_quality="dataset_transcript",
            notes="Good for spontaneous dialogue, code-switching, short turns, and filler words.",
        ))
        if len(items) >= count:
            return items
    raise RuntimeError(f"Not enough ASCEND samples for {prefix}: requested {count}, got {len(items)}")


def _take_multidialog(count: int) -> list[CorpusItem]:
    items: list[CorpusItem] = []
    for row in _iter_multidialog():
        text = _clean_text(row.get("value"))
        audio_info = row.get("audio") or {}
        audio = _to_float32_mono(audio_info.get("array"))
        sample_rate = int(audio_info.get("sampling_rate") or 16000)
        if not text or len(audio) / max(sample_rate, 1) < 1.0:
            continue
        item_id = f"en_dialogue_{len(items) + 1:03d}"
        items.append(CorpusItem(
            id=item_id,
            language="en",
            category="dialogue",
            text=text,
            audio=audio,
            sample_rate=sample_rate,
            source_dataset="IVLLab/MultiDialog",
            source_config="train",
            source_split="train",
            description="MultiDialog English spoken dialogue sample.",
            speaker_count=2,
            speech_rate="normal",
            notes="Useful for short conversational turns and dialogue endpointing.",
        ))
        if len(items) >= count:
            return items
    raise RuntimeError(f"Not enough MultiDialog samples: requested {count}, got {len(items)}")


def build_corpus() -> list[CorpusItem]:
    """Return the required 35-item corpus: 20 zh, 10 en, 5 mixed."""
    items: list[CorpusItem] = []
    items.extend(_take_fleurs(
        config="cmn_hans_cn",
        language="zh-CN",
        prefix="zh_read",
        categories=["news", "novel", "short_video", "noise"],
        count=12,
    ))
    items.extend(_take_ascend(
        language_filter="zh",
        language="zh-CN",
        prefix="zh_dialogue",
        category="dialogue",
        count=8,
    ))
    items.extend(_take_fleurs(
        config="en_us",
        language="en",
        prefix="en_read",
        categories=["news", "meeting", "novel"],
        count=6,
    ))
    items.extend(_take_multidialog(count=4))
    items.extend(_take_ascend(
        language_filter="mixed",
        language="zh-CN",
        prefix="mixed_dialogue",
        category="mixed_dialogue",
        count=5,
        require_mixed=True,
    ))
    return items


def _thresholds_for(item: CorpusItem) -> dict[str, Any]:
    is_cjk = item.language in _CJK_LANGS
    min_acc = 0.72 if is_cjk else 0.78
    if item.category in {"dialogue", "mixed_dialogue"}:
        min_acc -= 0.04
    if item.category in {"noise", "short_video"}:
        min_acc -= 0.03
    return {
        "min_accuracy": {
            f"{item.category}_turbo": round(min_acc, 2),
            f"{item.category}_belle": round(max(0.60, min_acc - 0.04), 2),
        },
        "max_dropped_chunks": 0,
        "max_repetition_ratio": 0.05 if item.category in {"dialogue", "mixed_dialogue"} else 0.03,
        "max_missing_sentence_rate": 0.20 if item.category in {"dialogue", "mixed_dialogue"} else 0.14,
        "max_duplicate_rate": 0.05 if item.category in {"dialogue", "mixed_dialogue"} else 0.03,
        "max_hallucination_rate": 0.03,
        "max_average_final_latency_ms": 4000 if is_cjk else 3500,
    }


def write_corpus(items: list[CorpusItem], output_dir: Path, *, force: bool) -> None:
    audio_dir = output_dir / "audio"
    refs_dir = output_dir / "refs"
    metadata_dir = output_dir / "metadata"
    manifest_path = output_dir / "manifest.yaml"
    sources_path = output_dir / "sources.json"
    if manifest_path.exists() and not force:
        raise SystemExit(f"{manifest_path} already exists. Use --force to regenerate.")
    for folder in (audio_dir, refs_dir, metadata_dir):
        folder.mkdir(parents=True, exist_ok=True)

    manifest_samples: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for item in items:
        audio_rel = Path("audio") / f"{item.id}.wav"
        ref_rel = Path("refs") / f"{item.id}.txt"
        metadata_rel = Path("metadata") / f"{item.id}.yaml"
        _write_wav(output_dir / audio_rel, item.audio, item.sample_rate)
        (output_dir / ref_rel).write_text(item.text + "\n", encoding="utf-8")

        duration = len(item.audio) / max(item.sample_rate, 1)
        metadata = {
            "id": item.id,
            "language": item.language,
            "category": item.category,
            "duration_sec": round(duration, 3),
            "speaker_count": item.speaker_count,
            "gender": item.gender,
            "noise_level": item.noise_level,
            "speech_rate": item.speech_rate,
            "has_music": item.has_music,
            "reference_quality": item.reference_quality,
            "source_dataset": item.source_dataset,
            "source_config": item.source_config,
            "source_split": item.source_split,
            "notes": item.notes,
        }
        (output_dir / metadata_rel).write_text(
            yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        sample = {
            "id": item.id,
            "language": item.language,
            "category": item.category,
            "audio": str(audio_rel).replace("\\", "/"),
            "reference": str(ref_rel).replace("\\", "/"),
            "description": item.description,
            "duration_sec": round(duration, 3),
            "speaker_count": item.speaker_count,
            "gender": item.gender,
            "noise_level": item.noise_level,
            "speech_rate": item.speech_rate,
            "has_music": item.has_music,
            "reference_quality": item.reference_quality,
            "notes": item.notes,
        }
        sample.update(_thresholds_for(item))
        manifest_samples.append(sample)
        sources.append({**metadata, "audio": sample["audio"], "reference": sample["reference"]})

    manifest = {
        "schema_version": 1,
        "description": (
            "Generated public ASR regression corpus. Audio/reference files are local "
            "artifacts under downloads/ and are intentionally not committed."
        ),
        "samples": manifest_samples,
    }
    manifest_path.write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False, width=120),
        encoding="utf-8",
    )
    sources_path.write_text(json.dumps(sources, indent=2, ensure_ascii=False), encoding="utf-8")


def _count_by_language(items: list[CorpusItem]) -> dict[str, int]:
    counts = {"zh": 0, "en": 0, "mixed": 0}
    for item in items:
        if item.category == "mixed_dialogue":
            counts["mixed"] += 1
        elif item.language.startswith("zh") or item.language.startswith("cmn"):
            counts["zh"] += 1
        elif item.language.startswith("en"):
            counts["en"] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    _safe_stdout()
    parser = argparse.ArgumentParser(description="Prepare the local ASR regression corpus.")
    parser.add_argument("--output", default="downloads/asr_regression", help="Output corpus directory.")
    parser.add_argument("--force", action="store_true", help="Regenerate manifest/audio/reference files.")
    args = parser.parse_args(argv)

    output_dir = Path(args.output)
    items = build_corpus()
    counts = _count_by_language(items)
    if counts != {"zh": 20, "en": 10, "mixed": 5}:
        raise RuntimeError(f"Unexpected corpus counts: {counts}")
    write_corpus(items, output_dir, force=args.force)
    print(f"Prepared {len(items)} samples in {output_dir}")
    print(f"Counts: {counts}")
    print(f"Manifest: {output_dir / 'manifest.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
