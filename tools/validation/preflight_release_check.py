from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.application.first_run_readiness import evaluate_first_run_readiness
from app.bootstrap.runtime_assets import resolve_asr_model_path, resolve_llm_model_path, resolve_runtime_dir
from app.infra.audio.virtual_devices import detect_virtual_audio_install
from app.infra.config.settings_store import load_config

from tools.validation.common import (
    DRIVER_FORMAT_EXPECTED,
    PCM16_STEREO_BOUNDARY_READY,
    PROTOCOL_V2_READY,
    ValidationItem,
    ValidationReport,
    aggregate_status,
    bridge_status_item,
    driver_format_status_item,
    inspect_wdk_environment,
    list_audio_devices,
    print_report,
    safe_probe_bridge,
    wdk_environment_item,
    write_json_report,
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check portable release readiness for SyncTranslate v2.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--json", default="", help="Write preflight JSON report")
    parser.add_argument("--strict", action="store_true", help="Return failure when selected mode readiness is incomplete")
    parser.add_argument("--mode", choices=["meeting", "dialogue"], default="", help="Validate a specific product mode")
    parser.add_argument("--repo-root", default="", help="Repository root to validate")
    parser.add_argument("--onedir-root", default="", help="PyInstaller onedir root to validate")
    parser.add_argument("--probe-bridge", action="store_true", help="Run bridge heartbeat and PCM loopback probe")
    parser.add_argument("--no-bridge-probe", action="store_true", help="Skip bridge probe even for dialogue mode")
    return parser


def build_report(
    config_path: str = "config.yaml",
    *,
    strict: bool = False,
    mode: str = "",
    repo_root: str | Path | None = None,
    onedir_root: str | Path | None = None,
    probe_bridge: bool = False,
) -> ValidationReport:
    config = load_config(config_path)
    selected_mode = str(mode or getattr(config.runtime, "session_mode", "meeting") or "meeting")
    readiness = evaluate_first_run_readiness(
        config,
        repo_root=repo_root,
        onedir_root=onedir_root,
        probe_bridge=probe_bridge,
    )
    items: list[ValidationItem] = []
    missing_runtime: list[str] = []
    missing_models: list[str] = []
    missing_driver: list[str] = []
    suggested_commands: list[str] = []
    wdk_item = wdk_environment_item()
    if selected_mode == "meeting" and wdk_item.status != "PASS":
        items.append(
            ValidationItem(
                "wdk_environment_status",
                "PASS",
                "WDK build tools are not required for meeting mode.",
                {**wdk_item.details, "actual_status": wdk_item.status},
            )
        )
    else:
        items.append(wdk_item)

    shared = resolve_runtime_dir("shared", repo_root=repo_root, onedir_root=onedir_root)
    fw = resolve_runtime_dir("faster_whisper", repo_root=repo_root, onedir_root=onedir_root)
    asr = resolve_asr_model_path(config.asr.model, repo_root=repo_root, onedir_root=onedir_root)
    llm = resolve_llm_model_path(config.llm.runtime.model_path, repo_root=repo_root, onedir_root=onedir_root)
    for name, asset, required in (
        ("runtimes_shared", shared, True),
        ("runtimes_faster_whisper", fw, True),
        ("asr_model", asr, True),
        ("llm_model", llm, True),
    ):
        status = "PASS" if asset.exists else "FAIL" if (required and strict) else "WARN"
        items.append(
            ValidationItem(
                name,
                status,
                f"{asset.source} found at {asset.resolved}" if asset.exists else f"{asset.source} missing at {asset.resolved}",
                {"configured": asset.configured, "resolved": str(asset.resolved), "suggested_fix": asset.suggested_fix},
            )
        )
        if not asset.exists and name.startswith("runtimes_"):
            missing_runtime.append(name)
        if not asset.exists and name in {"asr_model", "llm_model"}:
            missing_models.append(name)
            if asset.suggested_fix not in suggested_commands:
                suggested_commands.append(asset.suggested_fix)

    inputs, outputs, device_error = list_audio_devices()
    meeting_audio_ok = bool(inputs or outputs)
    if selected_mode == "meeting":
        items.append(
            ValidationItem(
                "meeting_audio_capability",
                "PASS" if meeting_audio_ok else "FAIL" if strict else "WARN",
                "Meeting mode has input/output device candidates."
                if meeting_audio_ok
                else f"Meeting mode needs at least one input or output device. {device_error}",
                {"input_count": len(inputs), "output_count": len(outputs)},
            )
        )

    bridge_path = Path(str(config.audio.virtual_audio.bridge_path or ""))
    bridge_exists = bridge_path.exists()
    if not bridge_exists:
        for root in (repo_root, onedir_root, Path.cwd()):
            if root and (Path(root) / bridge_path).exists():
                bridge_exists = True
                break
    bridge_required = selected_mode == "dialogue"
    items.append(
        ValidationItem(
            "audio_bridge",
            "PASS" if bridge_exists else "FAIL" if bridge_required and strict else "WARN",
            f"Audio bridge found: {config.audio.virtual_audio.bridge_path}"
            if bridge_exists
            else f"Audio bridge missing: {config.audio.virtual_audio.bridge_path}",
            {"path": str(config.audio.virtual_audio.bridge_path), "required": bridge_required},
        )
    )
    if not bridge_exists and bridge_required:
        missing_runtime.append("audio_bridge")

    bridge_probe = None
    if probe_bridge and bridge_exists:
        bridge_probe = safe_probe_bridge(str(config.audio.virtual_audio.bridge_path or ""))
        items.append(
            bridge_status_item(
                bridge_probe,
                required=bridge_required,
                bridge_path=str(config.audio.virtual_audio.bridge_path or ""),
            )
        )
        if bridge_required and not (bridge_probe is not None and bridge_probe.ready):
            missing_runtime.append("audio_bridge_probe")
            suggested_commands.append(
                "Rebuild runtimes/audio/sync_audio_bridge.exe with tools/runtime_setup/package_audio_bridge.ps1, then rerun preflight."
            )
    elif bridge_required:
        items.append(
            ValidationItem(
                "bridge_status",
                "WARN",
                "Bridge probe was not performed; run with --probe-bridge or validate_windows_audio_runtime.py for heartbeat/PCM loopback.",
                {"bridge_path": str(config.audio.virtual_audio.bridge_path or ""), "bridge_probe_performed": False},
            )
        )

    try:
        driver = detect_virtual_audio_install()
        driver_ready = bool(driver.speaker_available and driver.microphone_available)
    except Exception:
        driver_ready = False
        driver = None
    if selected_mode == "dialogue":
        items.append(
            ValidationItem(
                "virtual_driver",
                "PASS" if driver_ready else "FAIL" if strict else "WARN",
                "SyncTranslate virtual speaker/microphone are available."
                if driver_ready
                else "Dialogue mode needs SyncTranslate Virtual Speaker and Virtual Microphone.",
            )
        )
        if not driver_ready:
            missing_driver.append("virtual_driver")
            suggested_commands.append("Install SyncTranslate Virtual Audio Driver, enable Windows Test Mode if required, then reboot.")
    else:
        items.append(ValidationItem("meeting_no_driver_portable", "PASS", "Meeting mode does not require virtual driver or bridge."))
    if selected_mode != "dialogue":
        items.append(
            ValidationItem(
                "driver_format_status",
                "PASS",
                "Driver format is not required for meeting mode.",
                {"expected": DRIVER_FORMAT_EXPECTED, "driver_format_status": "NOT_REQUIRED"},
            )
        )
    elif driver is not None:
        items.append(driver_format_status_item(driver, required=selected_mode == "dialogue"))
        if selected_mode == "dialogue" and items[-1].status == "FAIL":
            missing_driver.append("driver_format")
    else:
        items.append(
            ValidationItem(
                "driver_format_status",
                "FAIL" if selected_mode == "dialogue" else "WARN",
                "Driver format cannot be checked because virtual driver inspection failed.",
                {"expected": DRIVER_FORMAT_EXPECTED, "driver_format_status": "UNKNOWN"},
            )
        )

    portable_meeting_ready = bool(shared.exists and fw.exists and asr.exists and llm.exists and meeting_audio_ok)
    if probe_bridge:
        bridge_ready = bool(bridge_probe is not None and bridge_probe.ready)
        dialogue_ready = bool(
            portable_meeting_ready
            and bridge_exists
            and bridge_ready
            and (driver_ready if selected_mode == "dialogue" else readiness.summary.get("driver_ready", False))
        )
    else:
        # Without a bridge probe, do not claim full dialogue readiness. The
        # package may contain the bridge executable, but heartbeat/PCM loopback
        # still need validate_windows_audio_runtime.py or a probed preflight.
        dialogue_ready = bool(readiness.summary.get("dialogue_ready", False))
    wdk = inspect_wdk_environment()
    details = {
        "config_path": str(config_path),
        "mode": selected_mode,
        "strict": strict,
        "portable_meeting_ready": portable_meeting_ready,
        "dialogue_ready": dialogue_ready,
        "bridge_probe_performed": probe_bridge,
        "bridge_ready": bool(bridge_probe is not None and bridge_probe.ready)
        if probe_bridge
        else bool(readiness.summary.get("bridge_ready", False)),
        "missing_runtime_items": missing_runtime,
        "missing_model_items": missing_models,
        "missing_driver_items": missing_driver,
        "suggested_commands": suggested_commands,
        "readiness": readiness.summary,
        "driver_format_status": next((item.details.get("driver_format_status", item.status) for item in items if item.name == "driver_format_status"), "UNKNOWN"),
        "driver_format_expected": DRIVER_FORMAT_EXPECTED,
        "driver_build_tools_available": wdk["status"] == "PASS",
        "wdk_environment_status": wdk["status"],
        "protocol_v2_ready": PROTOCOL_V2_READY,
        "pcm16_stereo_boundary_ready": PCM16_STEREO_BOUNDARY_READY,
    }
    status = aggregate_status(items)
    if strict:
        if selected_mode == "meeting" and not portable_meeting_ready:
            status = "FAIL"
        if selected_mode == "dialogue" and not dialogue_ready:
            status = "FAIL"
    return ValidationReport("Portable release preflight", status, items, details)


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    try:
        report = build_report(
            args.config,
            strict=args.strict,
            mode=args.mode,
            repo_root=args.repo_root or None,
            onedir_root=args.onedir_root or None,
            probe_bridge=bool(args.probe_bridge or (args.mode == "dialogue" and not args.no_bridge_probe)),
        )
        print_report(report)
        print(
            "summary: "
            f"portable_meeting_ready={report.details.get('portable_meeting_ready')} "
            f"dialogue_ready={report.details.get('dialogue_ready')} "
            f"missing_models={','.join(report.details.get('missing_model_items') or []) or 'none'}"
        )
        if args.json:
            path = write_json_report(report, args.json)
            print(f"JSON written: {path}")
        return 0 if report.status in {"PASS", "WARN"} else 1
    except Exception as exc:
        print(f"FAIL release preflight could not complete: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
