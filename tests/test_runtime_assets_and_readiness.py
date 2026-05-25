from __future__ import annotations

import json
import zipfile

from app.application.first_run_readiness import evaluate_first_run_readiness
from app.bootstrap.runtime_assets import resolve_asr_model_path, resolve_llm_model_path
from app.infra.config.schema import AppConfig
from tools.validation import preflight_release_check
from tools.validation.export_diagnostics_bundle import build_bundle


def _make_layout(root):
    (root / "runtimes" / "shared").mkdir(parents=True)
    (root / "runtimes" / "faster_whisper").mkdir(parents=True)
    (root / "runtimes" / "models" / "asr" / "large-v3-turbo").mkdir(parents=True)
    llm = root / "runtimes" / "models" / "llm" / "hy-mt1.5-7b.gguf"
    llm.parent.mkdir(parents=True)
    llm.write_bytes(b"fake")
    audio = root / "runtimes" / "audio" / "sync_audio_bridge.exe"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"fake")


def test_asr_model_resolver_finds_dev_layout(tmp_path) -> None:
    _make_layout(tmp_path)

    resolved = resolve_asr_model_path("large-v3-turbo", repo_root=tmp_path)

    assert resolved.exists
    assert resolved.resolved == (tmp_path / "runtimes" / "models" / "asr" / "large-v3-turbo").resolve()


def test_asr_model_resolver_finds_onedir_layout(tmp_path) -> None:
    onedir = tmp_path / "SyncTranslate-onedir"
    _make_layout(onedir)

    resolved = resolve_asr_model_path("large-v3-turbo", onedir_root=onedir)

    assert resolved.exists
    assert resolved.resolved == (onedir / "runtimes" / "models" / "asr" / "large-v3-turbo").resolve()


def test_asr_model_resolver_prefers_custom_config_path(tmp_path) -> None:
    custom = tmp_path / "custom-asr"
    custom.mkdir()
    _make_layout(tmp_path)

    resolved = resolve_asr_model_path(str(custom), repo_root=tmp_path)

    assert resolved.exists
    assert resolved.resolved == custom.resolve()


def test_llm_model_resolver_finds_release_layout(tmp_path) -> None:
    _make_layout(tmp_path)

    resolved = resolve_llm_model_path(r".\runtimes\models\llm\hy-mt1.5-7b.gguf", repo_root=tmp_path)

    assert resolved.exists
    assert resolved.resolved.name == "hy-mt1.5-7b.gguf"


def test_first_run_readiness_meeting_no_driver_can_be_ready(tmp_path, monkeypatch) -> None:
    _make_layout(tmp_path)
    cfg = AppConfig()
    cfg.runtime.session_mode = "meeting"
    monkeypatch.setattr("app.application.first_run_readiness.DeviceManager", lambda: type("M", (), {
        "list_input_devices": lambda self: [object()],
        "list_output_devices": lambda self: [],
    })())
    monkeypatch.setattr("app.application.first_run_readiness.detect_virtual_audio_install", lambda: type("D", (), {
        "speaker_available": False,
        "microphone_available": False,
        "speaker_name": "",
        "microphone_name": "",
    })())

    report = evaluate_first_run_readiness(cfg, repo_root=tmp_path, probe_bridge=False)

    assert report.summary["meeting_ready"] is True
    assert report.summary["dialogue_ready"] is False
    assert any(item.name == "virtual_driver" and item.status == "WARN" for item in report.items)


def test_preflight_strict_meeting_uses_resolver_and_passes_fake_layout(tmp_path, monkeypatch) -> None:
    _make_layout(tmp_path)
    cfg = AppConfig()
    monkeypatch.setattr(preflight_release_check, "load_config", lambda _path: cfg)
    monkeypatch.setattr(preflight_release_check, "list_audio_devices", lambda: ([{"name": "Mic"}], [], ""))

    report = preflight_release_check.build_report("config.yaml", strict=True, mode="meeting", repo_root=tmp_path)

    assert report.status == "PASS"
    assert report.details["portable_meeting_ready"] is True
    assert report.details["dialogue_ready"] is False
    assert report.details["bridge_probe_performed"] is False
    assert report.details["missing_model_items"] == []


def test_preflight_missing_asr_model_reports_fix(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("SYNC_TRANSLATE_MODELS_DIR", raising=False)
    (tmp_path / "runtimes" / "shared").mkdir(parents=True)
    (tmp_path / "runtimes" / "faster_whisper").mkdir(parents=True)
    llm = tmp_path / "runtimes" / "models" / "llm" / "hy-mt1.5-7b.gguf"
    llm.parent.mkdir(parents=True)
    llm.write_bytes(b"fake")
    cfg = AppConfig()
    monkeypatch.setattr(preflight_release_check, "load_config", lambda _path: cfg)
    monkeypatch.setattr(preflight_release_check, "list_audio_devices", lambda: ([{"name": "Mic"}], [], ""))

    report = preflight_release_check.build_report("config.yaml", strict=True, mode="meeting", repo_root=tmp_path)

    assert report.status == "FAIL"
    assert "asr_model" in report.details["missing_model_items"]
    assert any("prepare_external_runtimes.ps1" in cmd for cmd in report.details["suggested_commands"])


def test_diagnostics_bundle_excludes_models_and_writes_summary(tmp_path, monkeypatch) -> None:
    _make_layout(tmp_path)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("runtime:\n  session_mode: meeting\nsecret_token: abc\n", encoding="utf-8")
    cfg = AppConfig()
    monkeypatch.setattr("tools.validation.export_diagnostics_bundle.load_config", lambda _path: cfg)
    monkeypatch.setattr("tools.validation.export_diagnostics_bundle.build_windows_audio_report", lambda *_a, **_k: type("R", (), {"to_dict": lambda self: {"status": "WARN"}})())
    monkeypatch.setattr("tools.validation.export_diagnostics_bundle.build_preflight_report", lambda *_a, **_k: type("R", (), {"to_dict": lambda self: {"status": "PASS"}})())
    monkeypatch.setattr("tools.validation.export_diagnostics_bundle.evaluate_first_run_readiness", lambda *_a, **_k: type("R", (), {"to_dict": lambda self: {"summary": {"meeting_ready": True}}})())
    monkeypatch.setattr("tools.validation.export_diagnostics_bundle.list_audio_devices", lambda: ([], [], ""))
    monkeypatch.setattr("tools.validation.export_diagnostics_bundle._run_check", lambda _path: "Config OK")

    output = tmp_path / "bundle.zip"
    bundle = build_bundle(config_path=str(cfg_path), output=str(output))

    assert bundle.exists()
    with zipfile.ZipFile(bundle) as zf:
        names = set(zf.namelist())
        assert "config.sanitized.yaml" in names
        assert "main_check.txt" in names
        assert not any(name.startswith("runtimes/models") for name in names)
        sanitized = zf.read("config.sanitized.yaml").decode("utf-8")
        assert "***REDACTED***" in sanitized
