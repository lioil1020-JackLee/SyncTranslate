from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import sys


DEFAULT_ASR_MODEL = "large-v3-turbo"
DEFAULT_ASR_MODEL_REPO = "h2oai/faster-whisper-large-v3-turbo"
DEFAULT_LLM_MODEL = "hy-mt1.5-7b.gguf"

_ASR_MODEL_REPO_ALIASES: dict[str, tuple[str, ...]] = {
    # Public CTranslate2 conversion of openai/whisper-large-v3-turbo.
    # The old Systran large-v3-turbo repo id is not publicly available.
    DEFAULT_ASR_MODEL: (
        DEFAULT_ASR_MODEL_REPO,
        "dropbox-dash/faster-whisper-large-v3-turbo",
        "deepdml/faster-whisper-large-v3-turbo-ct2",
    ),
    "faster-whisper-large-v3-turbo": (
        DEFAULT_ASR_MODEL_REPO,
        "dropbox-dash/faster-whisper-large-v3-turbo",
        "deepdml/faster-whisper-large-v3-turbo-ct2",
    ),
    "Systran/faster-whisper-large-v3-turbo": (
        "Systran/faster-whisper-large-v3-turbo",
        DEFAULT_ASR_MODEL_REPO,
        "dropbox-dash/faster-whisper-large-v3-turbo",
        "deepdml/faster-whisper-large-v3-turbo-ct2",
    ),
}


@dataclass(frozen=True, slots=True)
class ResolvedAsset:
    configured: str
    resolved: Path
    exists: bool
    source: str
    suggested_fix: str
    candidates: tuple[Path, ...]


def runtime_base_candidates(*, repo_root: str | Path | None = None, onedir_root: str | Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    if repo_root:
        candidates.append(Path(repo_root))
    if onedir_root:
        candidates.append(Path(onedir_root))
    explicit_root = bool(repo_root or onedir_root)
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent)
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass))
    if not explicit_root:
        candidates.append(Path.cwd())
        candidates.append(Path(__file__).resolve().parents[2])
    unique: list[Path] = []
    seen: set[str] = set()
    for item in candidates:
        try:
            key = str(item.resolve())
        except Exception:
            key = str(item)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def runtime_root_candidates(*, repo_root: str | Path | None = None, onedir_root: str | Path | None = None) -> list[Path]:
    roots: list[Path] = []
    for base in runtime_base_candidates(repo_root=repo_root, onedir_root=onedir_root):
        roots.append(base / "runtimes")
        roots.append(base / "_internal" / "runtimes")
    return _dedupe_paths(roots)


def resolve_runtime_dir(
    name: str,
    *,
    repo_root: str | Path | None = None,
    onedir_root: str | Path | None = None,
) -> ResolvedAsset:
    candidates = [root / name for root in runtime_root_candidates(repo_root=repo_root, onedir_root=onedir_root)]
    return _first_existing(
        configured=name,
        candidates=candidates,
        fallback=candidates[0] if candidates else Path("runtimes") / name,
        source=f"runtime:{name}",
        suggested_fix="Run: powershell -ExecutionPolicy Bypass -File .\\tools\\runtime_setup\\prepare_external_runtimes.ps1",
    )


def resolve_asr_model_path(
    model: str,
    *,
    repo_root: str | Path | None = None,
    onedir_root: str | Path | None = None,
) -> ResolvedAsset:
    raw = str(model or DEFAULT_ASR_MODEL).strip() or DEFAULT_ASR_MODEL
    configured = Path(raw)
    candidates: list[Path] = []
    if configured.is_absolute():
        candidates.append(configured)
    else:
        is_simple_model_name = configured.parent == Path(".")
        if is_simple_model_name:
            for root in runtime_root_candidates(repo_root=repo_root, onedir_root=onedir_root):
                candidates.append(root / "models" / "asr" / raw)
                candidates.append(root / "models" / raw)
            for base in runtime_base_candidates(repo_root=repo_root, onedir_root=onedir_root):
                candidates.append(base / configured)
        else:
            for base in runtime_base_candidates(repo_root=repo_root, onedir_root=onedir_root):
                candidates.append(base / configured)
            for root in runtime_root_candidates(repo_root=repo_root, onedir_root=onedir_root):
                candidates.append(root / "models" / "asr" / configured.name)
        models_dir = os.environ.get("SYNC_TRANSLATE_MODELS_DIR", "").strip()
        if models_dir:
            candidates.append(Path(models_dir) / "asr" / raw)
            candidates.append(Path(models_dir) / raw)
    fallback = candidates[0] if candidates else configured
    return _first_existing(
        configured=raw,
        candidates=candidates,
        fallback=fallback,
        source="asr_model",
        suggested_fix=(
            "Run: powershell -ExecutionPolicy Bypass -File "
            ".\\tools\\runtime_setup\\prepare_external_runtimes.ps1"
        ),
    )


def asr_model_repo_candidates(repo_id_or_alias: str | None = None) -> tuple[str, ...]:
    raw = str(repo_id_or_alias or os.environ.get("SYNC_TRANSLATE_ASR_MODEL_REPO") or DEFAULT_ASR_MODEL_REPO).strip()
    if not raw:
        raw = DEFAULT_ASR_MODEL_REPO
    candidates = _ASR_MODEL_REPO_ALIASES.get(raw, (raw,))
    return tuple(dict.fromkeys(candidates))


def resolve_llm_model_path(
    model_path: str,
    *,
    repo_root: str | Path | None = None,
    onedir_root: str | Path | None = None,
) -> ResolvedAsset:
    raw = str(model_path or rf".\runtimes\models\llm\{DEFAULT_LLM_MODEL}").strip()
    configured = Path(raw)
    candidates: list[Path] = []
    if configured.is_absolute():
        candidates.append(configured)
    else:
        for base in runtime_base_candidates(repo_root=repo_root, onedir_root=onedir_root):
            candidates.append(base / configured)
        for root in runtime_root_candidates(repo_root=repo_root, onedir_root=onedir_root):
            candidates.append(root / "models" / "llm" / configured.name)
        models_dir = os.environ.get("SYNC_TRANSLATE_MODELS_DIR", "").strip()
        if models_dir:
            candidates.append(Path(models_dir) / "llm" / configured.name)
    fallback = candidates[0] if candidates else configured
    return _first_existing(
        configured=raw,
        candidates=candidates,
        fallback=fallback,
        source="llm_model",
        suggested_fix=(
            "Run: powershell -ExecutionPolicy Bypass -File "
            ".\\tools\\runtime_setup\\prepare_external_runtimes.ps1"
        ),
    )


def _first_existing(
    *,
    configured: str,
    candidates: list[Path],
    fallback: Path,
    source: str,
    suggested_fix: str,
) -> ResolvedAsset:
    deduped = _dedupe_paths(candidates)
    for candidate in deduped:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        if resolved.exists():
            return ResolvedAsset(configured, resolved, True, source, suggested_fix, tuple(deduped))
    return ResolvedAsset(configured, fallback.resolve(), False, source, suggested_fix, tuple(deduped))


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        try:
            key = str(path.resolve())
        except Exception:
            key = str(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


__all__ = [
    "DEFAULT_ASR_MODEL",
    "DEFAULT_ASR_MODEL_REPO",
    "DEFAULT_LLM_MODEL",
    "ResolvedAsset",
    "asr_model_repo_candidates",
    "resolve_asr_model_path",
    "resolve_llm_model_path",
    "resolve_runtime_dir",
    "runtime_base_candidates",
    "runtime_root_candidates",
]
