from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from app.bootstrap import external_runtime
from app.infra.translation.inprocess_adapter import _resolve_model_path


class ExternalRuntimeTests(TestCase):
    def test_llama_cpp_lib_is_registered_as_dll_candidate(self) -> None:
        site_packages = Path(r"C:\app\runtimes\shared\Lib\site-packages")

        candidates = external_runtime._dll_dir_candidates(site_packages)

        self.assertIn(site_packages / "llama_cpp" / "lib", candidates)

    def test_add_dll_directory_keeps_handle_alive(self) -> None:
        if not hasattr(external_runtime.os, "add_dll_directory"):
            self.skipTest("os.add_dll_directory is not available")

        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            handle = object()
            before = len(external_runtime._DLL_DIRECTORY_HANDLES)

            with patch.object(external_runtime.os, "add_dll_directory", return_value=handle):
                external_runtime._add_dll_directory(path)

            self.assertIn(handle, external_runtime._DLL_DIRECTORY_HANDLES[before:])

    def test_relative_llm_model_path_resolves_next_to_frozen_exe(self) -> None:
        with TemporaryDirectory() as tmp:
            exe_dir = Path(tmp) / "SyncTranslate-onedir"
            model = exe_dir / "runtimes" / "models" / "llm" / "unit-test-model.gguf"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"gguf")

            with patch.object(sys, "frozen", True, create=True):
                with patch.object(sys, "executable", str(exe_dir / "SyncTranslate.exe")):
                    resolved = _resolve_model_path(r".\runtimes\models\llm\unit-test-model.gguf")

        self.assertEqual(resolved, model.resolve())
