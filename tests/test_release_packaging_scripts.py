from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_relocate_runtime_script_copies_and_validates_default_asr_model() -> None:
    script = (ROOT / "tools" / "runtime_setup" / "relocate_ai_runtime_artifacts.ps1").read_text(encoding="utf-8")

    assert '"asr", "belle-zh-ct2", "llm"' in script
    assert "models\\asr\\large-v3-turbo" in script
    assert "Packaged faster-whisper ASR model" in script


def test_github_release_workflow_verifies_packaged_asr_model() -> None:
    workflow = (ROOT / ".github" / "workflows" / "build-release.yml").read_text(encoding="utf-8")

    assert "Prepare external AI runtimes" in workflow
    assert "runtimes\\models\\asr\\large-v3-turbo" in workflow
    assert "Packaged faster-whisper ASR model is missing" in workflow
