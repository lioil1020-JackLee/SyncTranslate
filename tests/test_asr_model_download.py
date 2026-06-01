from __future__ import annotations

import types

import pytest

from app.bootstrap.runtime_assets import DEFAULT_ASR_MODEL_REPO, asr_model_repo_candidates
from tools.runtime_setup import download_asr_model


def test_large_v3_turbo_alias_uses_public_ctranslate2_repo() -> None:
    candidates = asr_model_repo_candidates("large-v3-turbo")

    assert candidates[0] == DEFAULT_ASR_MODEL_REPO
    assert "Systran/faster-whisper-large-v3-turbo" not in candidates[:1]


def test_old_systran_large_v3_turbo_repo_falls_back_to_public_repo() -> None:
    candidates = asr_model_repo_candidates("Systran/faster-whisper-large-v3-turbo")

    assert candidates[0] == "Systran/faster-whisper-large-v3-turbo"
    assert DEFAULT_ASR_MODEL_REPO in candidates


def test_download_asr_model_tries_fallback_repo(monkeypatch, tmp_path) -> None:
    calls: list[str] = []

    def fake_snapshot_download(*, repo_id: str, local_dir: str, local_dir_use_symlinks: bool) -> None:
        calls.append(repo_id)
        if repo_id == "Systran/faster-whisper-large-v3-turbo":
            raise RuntimeError("repository not found")
        assert local_dir == str(tmp_path)
        assert local_dir_use_symlinks is False

    fake_module = types.SimpleNamespace(snapshot_download=fake_snapshot_download)
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake_module)

    result = download_asr_model.main([
        "--repo-id",
        "Systran/faster-whisper-large-v3-turbo",
        "--local-dir",
        str(tmp_path),
    ])

    assert result == 0
    assert calls[:2] == ["Systran/faster-whisper-large-v3-turbo", DEFAULT_ASR_MODEL_REPO]


def test_download_asr_model_reports_all_failures(monkeypatch, tmp_path) -> None:
    def fake_snapshot_download(*, repo_id: str, local_dir: str, local_dir_use_symlinks: bool) -> None:
        raise RuntimeError(f"blocked {repo_id}")

    fake_module = types.SimpleNamespace(snapshot_download=fake_snapshot_download)
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake_module)

    with pytest.raises(RuntimeError) as exc_info:
        download_asr_model.main(["--repo-id", "large-v3-turbo", "--local-dir", str(tmp_path)])

    message = str(exc_info.value)
    assert "Unable to download faster-whisper ASR model" in message
    assert DEFAULT_ASR_MODEL_REPO in message
    assert "prepare_external_runtimes.ps1" in message
