from __future__ import annotations

import os
from pathlib import Path
import sys


_DLL_DIRECTORY_HANDLES: list[object] = []
_DLL_DIRECTORY_PATHS: set[str] = set()


def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "runtimes").exists():
        return candidate
    # In a git worktree, __file__ resolves into the worktree directory which
    # has no runtimes/.  The worktree's .git file points back to the main
    # repository: "gitdir: <main>/.git/worktrees/<name>".  Resolve that path
    # to find the main repository root where runtimes/ lives.
    git_marker = candidate / ".git"
    if git_marker.is_file():
        try:
            content = git_marker.read_text(encoding="utf-8", errors="replace").strip()
            if content.startswith("gitdir:"):
                git_dir = Path(content[len("gitdir:"):].strip())
                if not git_dir.is_absolute():
                    git_dir = (candidate / git_dir).resolve()
                # git_dir is .../main_repo/.git/worktrees/<name>
                # parents[2] is .../main_repo
                main_repo = git_dir.parents[2]
                if (main_repo / "runtimes").exists():
                    return main_repo
        except Exception:
            pass
    return candidate


def _site_packages_candidates(base: Path) -> list[Path]:
    # First priority: root runtimes/ (where external runtimes should be)
    # Fallback: _internal/runtimes/ (legacy location for compatibility)
    runtime_roots = [base / "runtimes", base / "_internal" / "runtimes"]
    names = ("shared", "faster_whisper")
    candidates: list[Path] = []
    for runtimes_root in runtime_roots:
        for name in names:
            runtime_dir = runtimes_root / name
            candidates.append(runtime_dir / "Lib" / "site-packages")
            candidates.append(runtime_dir / "site-packages")
    candidates.append(base / ".venv" / "Lib" / "site-packages")
    return candidates


def _dll_dir_candidates(site_packages: Path) -> list[Path]:
    dirs: list[Path] = []
    dirs.append(site_packages / "torch" / "lib")
    dirs.append(site_packages / "onnxruntime" / "capi")
    dirs.append(site_packages / "llama_cpp" / "lib")

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
    text = str(path)
    if text in _DLL_DIRECTORY_PATHS:
        return
    current_path = os.environ.get("PATH", "")
    if text not in current_path.split(os.pathsep):
        os.environ["PATH"] = text + os.pathsep + current_path if current_path else text
    if not hasattr(os, "add_dll_directory"):
        return
    try:
        handle = os.add_dll_directory(text)
    except Exception:
        pass
    else:
        # Keep the handle alive for the full process lifetime.  If it is
        # discarded, Windows removes the directory from the DLL search path.
        _DLL_DIRECTORY_HANDLES.append(handle)
        _DLL_DIRECTORY_PATHS.add(text)


def _configure_model_cache_env(base: Path) -> None:
    model_candidates = [
        base / "runtimes" / "models",
        base / "_internal" / "runtimes" / "models",
        base / "models",
    ]
    models_root = next((candidate for candidate in model_candidates if candidate.exists()), None)
    if models_root is None:
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

    existing_site_packages: list[Path] = []
    for candidate in _site_packages_candidates(base):
        if candidate.exists():
            existing_site_packages.append(candidate)

    # _prepend_sys_path inserts at sys.path[0], so iterate in reverse to preserve
    # declared priority order: shared -> faster_whisper.
    for candidate in reversed(existing_site_packages):
        _prepend_sys_path(candidate)

    for candidate in existing_site_packages:
        if not candidate.exists():
            continue
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
