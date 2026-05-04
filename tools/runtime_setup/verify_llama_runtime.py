from __future__ import annotations

import argparse
import os
from pathlib import Path
import site
import struct
import sys


_DLL_HANDLES: list[object] = []
_DLL_PATHS: set[str] = set()
_SYSTEM_DLLS = {
    "advapi32.dll",
    "bcrypt.dll",
    "cfgmgr32.dll",
    "comdlg32.dll",
    "crypt32.dll",
    "gdi32.dll",
    "imm32.dll",
    "kernel32.dll",
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_2.dll",
    "ntdll.dll",
    "ole32.dll",
    "oleaut32.dll",
    "rpcrt4.dll",
    "sechost.dll",
    "shell32.dll",
    "ucrtbase.dll",
    "user32.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "winmm.dll",
    "ws2_32.dll",
}
_DEBUG_RUNTIME_DLLS = {
    "msvcp140d.dll",
    "msvcp140_1d.dll",
    "msvcp140_2d.dll",
    "ucrtbased.dll",
    "vcruntime140d.dll",
    "vcruntime140_1d.dll",
}


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


def _find_llama_lib_dir() -> Path:
    for site_packages in _site_packages_candidates():
        lib_dir = site_packages / "llama_cpp" / "lib"
        if (lib_dir / "llama.dll").is_file():
            return lib_dir
    raise RuntimeError("Cannot find llama_cpp/lib containing llama.dll")


def _rva_to_offset(sections: list[tuple[int, int, int, int]], rva: int) -> int | None:
    for virtual_address, raw_offset, raw_size, virtual_size in sections:
        size = max(raw_size, virtual_size)
        if virtual_address <= rva < virtual_address + size:
            return raw_offset + (rva - virtual_address)
    return None


def _read_c_string(data: bytes, offset: int) -> str:
    end = data.find(b"\0", offset)
    if end < 0:
        end = len(data)
    return data[offset:end].decode("ascii", errors="replace")


def _pe_imports(path: Path) -> list[str]:
    data = path.read_bytes()
    pe_offset = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe_offset : pe_offset + 4] != b"PE\0\0":
        raise RuntimeError(f"Not a PE file: {path}")

    coff_offset = pe_offset + 4
    _, section_count, _, _, _, optional_header_size, _ = struct.unpack_from(
        "<HHIIIHH", data, coff_offset
    )
    optional_offset = coff_offset + 20
    magic = struct.unpack_from("<H", data, optional_offset)[0]
    data_directory_offset = optional_offset + (112 if magic == 0x20B else 96)
    import_rva, _ = struct.unpack_from("<II", data, data_directory_offset + 8)
    if import_rva == 0:
        return []

    section_offset = optional_offset + optional_header_size
    sections: list[tuple[int, int, int, int]] = []
    for index in range(section_count):
        offset = section_offset + index * 40
        virtual_size, virtual_address, raw_size, raw_offset = struct.unpack_from(
            "<IIII", data, offset + 8
        )
        sections.append((virtual_address, raw_offset, raw_size, virtual_size))

    import_offset = _rva_to_offset(sections, import_rva)
    if import_offset is None:
        raise RuntimeError(f"Cannot resolve import table for {path}")

    imports: list[str] = []
    for index in range(4096):
        descriptor = struct.unpack_from("<IIIII", data, import_offset + index * 20)
        if descriptor == (0, 0, 0, 0, 0):
            break
        name_offset = _rva_to_offset(sections, descriptor[3])
        if name_offset is not None:
            imports.append(_read_c_string(data, name_offset))
    return imports


def _resolve_dll(name: str, search_dirs: list[Path]) -> Path | None:
    lower = name.lower()
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        direct = directory / name
        if direct.is_file():
            return direct
        try:
            for child in directory.iterdir():
                if child.name.lower() == lower:
                    return child
        except OSError:
            continue
    return None


def _validate_static_cuda(added_dirs: list[Path], allow_missing_nvidia_driver: bool) -> None:
    lib_dir = _find_llama_lib_dir()
    required = ["llama.dll", "ggml.dll", "ggml-base.dll", "ggml-cpu.dll", "ggml-cuda.dll"]
    missing_files = [name for name in required if not (lib_dir / name).is_file()]
    if missing_files:
        raise RuntimeError("Missing llama CUDA runtime files: " + ", ".join(missing_files))

    search_dirs = [lib_dir, *added_dirs]
    search_dirs.extend(Path(item) for item in os.environ.get("PATH", "").split(os.pathsep) if item)

    visited: set[str] = set()
    queue = [lib_dir / "llama.dll"]
    all_imports: dict[str, list[str]] = {}
    missing_deps: set[str] = set()
    debug_deps: set[str] = set()

    while queue:
        dll = queue.pop(0).resolve()
        key = str(dll).lower()
        if key in visited:
            continue
        visited.add(key)
        imports = _pe_imports(dll)
        all_imports[dll.name] = imports
        for dependency in imports:
            dep_lower = dependency.lower()
            if dep_lower in _DEBUG_RUNTIME_DLLS:
                debug_deps.add(dependency)
                continue
            if dep_lower in _SYSTEM_DLLS or dep_lower.startswith("api-ms-win-"):
                continue
            resolved = _resolve_dll(dependency, search_dirs)
            if resolved is None:
                if allow_missing_nvidia_driver and dep_lower == "nvcuda.dll":
                    continue
                missing_deps.add(dependency)
            else:
                queue.append(resolved)

    if "ggml-cuda.dll" not in {name.lower() for name in all_imports.get("ggml.dll", [])}:
        raise RuntimeError("ggml.dll is not linked to ggml-cuda.dll")
    if debug_deps:
        raise RuntimeError("Debug MSVC runtime dependencies found: " + ", ".join(sorted(debug_deps)))
    if missing_deps:
        raise RuntimeError("Missing llama CUDA DLL dependencies: " + ", ".join(sorted(missing_deps)))

    print("llama_static_cuda=True")
    print("llama_lib_dir=" + str(lib_dir))
    for dll_name, imports in sorted(all_imports.items()):
        print("imports." + dll_name + "=" + ",".join(imports))


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify llama-cpp-python runtime loading.")
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail when llama-cpp-python does not report GPU offload support.",
    )
    parser.add_argument(
        "--static-cuda",
        action="store_true",
        help="Verify CUDA llama DLLs statically without loading the native library.",
    )
    parser.add_argument(
        "--skip-import",
        action="store_true",
        help="Skip dynamic llama_cpp import. Intended for CI runners without NVIDIA drivers.",
    )
    parser.add_argument(
        "--allow-missing-nvidia-driver",
        action="store_true",
        help="Allow nvcuda.dll to be absent during static validation.",
    )
    args = parser.parse_args()

    added = _register_runtime_dll_dirs()
    print("python=" + sys.executable)
    print("dll_dirs=" + ";".join(str(path) for path in added))

    if args.static_cuda:
        _validate_static_cuda(added, args.allow_missing_nvidia_driver)
    if args.skip_import:
        return 0

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
