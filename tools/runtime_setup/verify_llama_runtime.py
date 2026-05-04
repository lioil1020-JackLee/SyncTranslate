from __future__ import annotations

import argparse
import os
from pathlib import Path
import site
import sys


_DLL_HANDLES: list[object] = []
_DLL_PATHS: set[str] = set()


def _add_dll_directory(path: Path) -> None:
    if os.name != "nt" or not path.is_dir():
        return
    text = str(path)
    if text in _DLL_PATHS:
        return

    current_path = os.environ.get("PATH", "")
    if text not in current_path.split(os.pathsep):
        os.environ["PATH"] = text + os.pathsep + current_path if current_path else text

    if hasattr(os, "add_dll_directory"):
        handle = os.add_dll_directory(text)
        _DLL_HANDLES.append(handle)
    _DLL_PATHS.add(text)


def _site_packages_candidates() -> list[Path]:
    candidates: list[Path] = []
    prefixes = [Path(sys.prefix), Path(sys.base_prefix)]
    for prefix in prefixes:
        candidates.append(prefix / "Lib" / "site-packages")
        candidates.append(prefix / "site-packages")

    try:
        candidates.extend(Path(path) for path in site.getsitepackages())
    except Exception:
        pass

    try:
        candidates.append(Path(site.getusersitepackages()))
    except Exception:
        pass

    for item in sys.path:
        path = Path(item)
        if path.name == "site-packages":
            candidates.append(path)

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        text = str(resolved)
        if text not in seen:
            unique.append(resolved)
            seen.add(text)
    return unique


def _register_runtime_dll_dirs() -> list[Path]:
    added: list[Path] = []
    for site_packages in _site_packages_candidates():
        if not site_packages.is_dir():
            continue
        for dll_dir in (
            site_packages / "torch" / "lib",
            site_packages / "onnxruntime" / "capi",
            site_packages / "llama_cpp" / "lib",
        ):
            if dll_dir.is_dir():
                _add_dll_directory(dll_dir)
                added.append(dll_dir)

        nvidia_root = site_packages / "nvidia"
        if nvidia_root.is_dir():
            for child in nvidia_root.iterdir():
                bin_dir = child / "bin"
                if bin_dir.is_dir():
                    _add_dll_directory(bin_dir)
                    added.append(bin_dir)
    return added


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify llama-cpp-python runtime loading.")
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail when llama-cpp-python does not report GPU offload support.",
    )
    args = parser.parse_args()

    added = _register_runtime_dll_dirs()
    print("python=" + sys.executable)
    print("dll_dirs=" + ";".join(str(path) for path in added))

    from llama_cpp import Llama  # noqa: PLC0415
    import llama_cpp  # noqa: PLC0415
    from llama_cpp.llama_cpp import llama_supports_gpu_offload  # noqa: PLC0415

    gpu_supported = bool(llama_supports_gpu_offload())
    print("llama_cpp=" + str(getattr(llama_cpp, "__version__", "unknown")))
    print("llama_gpu=" + str(gpu_supported))
    print("Llama=" + str(Llama))

    if args.require_gpu and not gpu_supported:
        raise SystemExit("llama-cpp-python does not support GPU offload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
