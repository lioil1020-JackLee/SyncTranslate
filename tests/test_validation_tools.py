from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from app.infra.config.schema import AppConfig
from tools.validation import audio_smoke_test, preflight_release_check, validate_windows_audio_runtime
from tools.validation.common import ValidationItem, ValidationReport, aggregate_status, build_check_report


@dataclass(slots=True)
class _Endpoint:
    name: str = "endpoint"


@dataclass(slots=True)
class _DriverStatus:
    installed: bool = False
    speaker_available: bool = False
    microphone_available: bool = False
    speaker_index: int = -1
    microphone_index: int = -1
    speaker_name: str = ""
    microphone_name: str = ""
    render_endpoints: tuple[object, ...] = ()
    capture_endpoints: tuple[object, ...] = ()


def test_aggregate_status_orders_fail_warn_pass() -> None:
    assert aggregate_status([ValidationItem("a", "PASS", ""), ValidationItem("b", "WARN", "")]) == "WARN"
    assert aggregate_status([ValidationItem("a", "WARN", ""), ValidationItem("b", "FAIL", "")]) == "FAIL"
    assert aggregate_status([ValidationItem("a", "PASS", "")]) == "PASS"


def test_windows_runtime_validation_meeting_no_driver_is_warning_not_crash(monkeypatch) -> None:
    cfg = AppConfig()
    cfg.runtime.session_mode = "meeting"
    monkeypatch.setattr(validate_windows_audio_runtime, "load_config", lambda _path: cfg)
    monkeypatch.setattr(
        validate_windows_audio_runtime,
        "list_audio_devices",
        lambda: ([{"name": "Mic"}], [{"name": "Speaker"}], ""),
    )
    monkeypatch.setattr(validate_windows_audio_runtime, "query_default_devices", lambda: {"input": 0, "output": 1})
    monkeypatch.setattr(validate_windows_audio_runtime, "detect_virtual_audio_install", lambda: _DriverStatus())

    report = validate_windows_audio_runtime.build_report(
        "config.yaml",
        probe_bridge=False,
        probe_capture=False,
    )

    assert report.status == "WARN"
    assert any(item.name == "meeting_virtual_audio_dependency" and item.status == "PASS" for item in report.items)
    assert any(item.name == "driver_status" and item.status == "WARN" for item in report.items)


def test_common_check_report_dialogue_missing_driver_fails(monkeypatch) -> None:
    cfg = AppConfig()
    cfg.runtime.session_mode = "dialogue"
    monkeypatch.setattr("tools.validation.common.load_config", lambda _path: cfg)
    monkeypatch.setattr("tools.validation.common.list_audio_devices", lambda: ([{"name": "Mic"}], [{"name": "Speaker"}], ""))
    monkeypatch.setattr("tools.validation.common.detect_virtual_audio_install", lambda: _DriverStatus())
    monkeypatch.setattr("tools.validation.common.safe_probe_bridge", lambda _path: None)
    monkeypatch.setattr("tools.validation.common.probe_virtual_audio_bridge", lambda _path: None)

    report = build_check_report("config.yaml", probe_bridge=False)

    assert report.status == "FAIL"
    assert report.details["virtual_audio_required"] is True
    assert report.details["bridge_required"] is True
    assert any(item.name == "driver_status" and item.status == "FAIL" for item in report.items)


def test_audio_smoke_cli_arguments() -> None:
    args = audio_smoke_test.create_parser().parse_args(
        ["meeting-input", "--duration", "1.5", "--device", "USB Mic", "--wav", "downloads/validation/test.wav"]
    )

    assert args.mode == "meeting-input"
    assert args.duration == 1.5
    assert args.device == "USB Mic"
    assert args.wav.endswith("test.wav")


def test_preflight_release_reports_missing_runtime_assets_as_warn_without_strict(monkeypatch) -> None:
    cfg = AppConfig()
    monkeypatch.setattr(preflight_release_check, "load_config", lambda _path: cfg)
    monkeypatch.setattr("tools.validation.preflight_release_check.resolve_runtime_dir", lambda name, **_kwargs: SimpleNamespace(exists=False, source=name, resolved=name, configured=name, suggested_fix="fix"))
    monkeypatch.setattr("tools.validation.preflight_release_check.resolve_asr_model_path", lambda *_args, **_kwargs: SimpleNamespace(exists=False, source="asr", resolved="asr", configured="asr", suggested_fix="fix"))
    monkeypatch.setattr("tools.validation.preflight_release_check.resolve_llm_model_path", lambda *_args, **_kwargs: SimpleNamespace(exists=False, source="llm", resolved="llm", configured="llm", suggested_fix="fix"))
    monkeypatch.setattr(preflight_release_check, "list_audio_devices", lambda: ([], [], ""))

    report = preflight_release_check.build_report("config.yaml")

    assert report.status == "WARN"
    assert any(item.name == "runtimes_shared" and item.status == "WARN" for item in report.items)


def test_validate_runtime_cli_writes_json(tmp_path, monkeypatch) -> None:
    report = ValidationReport("fake", "WARN", [ValidationItem("fake", "WARN", "ok")])
    monkeypatch.setattr(validate_windows_audio_runtime, "build_report", lambda *_args, **_kwargs: report)
    monkeypatch.setattr(validate_windows_audio_runtime, "print_report", lambda _report: None)

    output = tmp_path / "windows_audio_runtime.json"
    exit_code = validate_windows_audio_runtime.main(["--json", str(output), "--no-bridge-probe", "--no-capture-probe"])

    assert exit_code == 0
    assert output.exists()
