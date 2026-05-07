from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from tools.runtime_smoke.run_runtime_smoke import _check_packaged_onedir


def test_packaged_onedir_structure_reports_missing_runtime_files() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        onedir = root / "SyncTranslate-onedir"
        onedir.mkdir()

        result = _check_packaged_onedir(onedir, root=root)

    assert result.ok is False
    assert result.returncode == 1
    assert "runtimes" in result.stderr_tail
    assert "SyncTranslate.exe" in result.stderr_tail


def test_packaged_onedir_structure_accepts_required_layout() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        onedir = root / "SyncTranslate-onedir"
        (onedir / "runtimes" / "shared" / "Scripts").mkdir(parents=True)
        (onedir / "runtimes" / "faster_whisper").mkdir(parents=True)
        (onedir / "runtimes" / "models").mkdir(parents=True)
        (onedir / "runtimes" / "shared" / "Scripts" / "python.exe").write_text("", encoding="utf-8")
        (onedir / "SyncTranslate.exe").write_text("", encoding="utf-8")

        result = _check_packaged_onedir(onedir, root=root)

    assert result.ok is True
    assert result.returncode == 0
