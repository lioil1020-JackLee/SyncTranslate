from __future__ import annotations

import os
from pathlib import Path
import sys


def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def _site_packages_candidates(base: Path) -> list[Path]:
    # First priority: root runtimes/ (where external runtimes should be)
    # Fallback: _internal/runtimes/ (legacy location for compatibility)
    runtime_roots = [base / "runtimes", base / "_internal" / "runtimes"]
    names = ("shared", "funasr", "faster_whisper")
    candidates: list[Path] = []
    for runtimes_root in runtime_roots:
        for name in names:
            runtime_dir = runtimes_root / name
            candidates.append(runtime_dir / "Lib" / "site-packages")
            candidates.append(runtime_dir / "site-packages")
    return candidates


def _dll_dir_candidates(site_packages: Path) -> list[Path]:
    dirs: list[Path] = []
    dirs.append(site_packages / "torch" / "lib")
    dirs.append(site_packages / "onnxruntime" / "capi")

    nvidia_root = site_packages / "nvidia"
    if nvidia_root.exists():
        for child in nvidia_root.iterdir():
            bin_dir = child / "bin"
            if bin_dir.exists():
                dirs.append(bin_dir)
    return dirs


def _prepend_sys_path(path: Path) -> None:
    text = str(path)
    if text in sys.path:
        return
    sys.path.insert(0, text)


def _add_dll_directory(path: Path) -> None:
    if os.name != "nt":
        return
    if not hasattr(os, "add_dll_directory"):
        return
    try:
        os.add_dll_directory(str(path))
    except Exception:
        pass


def _configure_model_cache_env(base: Path) -> None:
    models_root = base / "models"
    if not models_root.exists():
        return

    modelscope_cache = models_root / "modelscope"
    huggingface_cache = models_root / "huggingface"
    os.environ.setdefault("MODELSCOPE_CACHE", str(modelscope_cache))
    os.environ.setdefault("HF_HOME", str(huggingface_cache))
    os.environ.setdefault("SYNC_TRANSLATE_MODELS_DIR", str(models_root))


def configure_external_ai_runtime() -> dict[str, list[str]]:
    base = _base_dir()
    added_site_packages: list[str] = []
    added_dll_dirs: list[str] = []

    for candidate in _site_packages_candidates(base):
        if not candidate.exists():
            continue
        _prepend_sys_path(candidate)
        added_site_packages.append(str(candidate))
        for dll_dir in _dll_dir_candidates(candidate):
            if not dll_dir.exists():
                continue
            _add_dll_directory(dll_dir)
            added_dll_dirs.append(str(dll_dir))

    _configure_model_cache_env(base)
    return {
        "site_packages": added_site_packages,
        "dll_dirs": added_dll_dirs,
    }


__all__ = ["configure_external_ai_runtime"]
