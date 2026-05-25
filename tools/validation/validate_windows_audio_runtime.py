from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.infra.audio.virtual_devices import detect_virtual_audio_install
from app.infra.audio.capture import AudioCapture
from app.infra.audio.frame import ChannelPolicy
from app.infra.audio.loopback_capture import WasapiLoopbackCaptureSource
from app.infra.config.settings_store import load_config

from tools.validation.common import (
    ValidationItem,
    ValidationReport,
    DRIVER_FORMAT_EXPECTED,
    PCM16_STEREO_BOUNDARY_READY,
    PROTOCOL_V2_READY,
    aggregate_status,
    bridge_required,
    bridge_status_item,
    driver_format_status_item,
    driver_status_item,
    inspect_wdk_environment,
    list_audio_devices,
    os_item,
    print_report,
    query_default_devices,
    safe_probe_bridge,
    virtual_audio_required,
    wdk_environment_item,
    write_json_report,
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate SyncTranslate Windows audio runtime readiness.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--json", default="", help="Write validation report JSON to this path")
    parser.add_argument("--no-bridge-probe", action="store_true", help="Skip bridge heartbeat and PCM loopback probe")
    parser.add_argument("--no-capture-probe", action="store_true", help="Only enumerate devices; do not open capture streams")
    return parser


def _probe_input_capture(device: str, sample_rate: int) -> ValidationItem:
    if not device:
        return ValidationItem("meeting_system_input", "WARN", "No input device is configured or available.")
    capture = AudioCapture()
    try:
        capture.start(device, sample_rate=sample_rate, chunk_ms=40, channels_policy=ChannelPolicy.STEREO_OR_MONO.value)
        capture.stop()
        return ValidationItem("meeting_system_input", "PASS", f"System input capture can start: {device}")
    except Exception as exc:
        return ValidationItem("meeting_system_input", "WARN", f"System input capture could not start on '{device}': {exc}")
    finally:
        try:
            capture.stop()
        except Exception:
            pass


def _probe_loopback_capture(device: str, sample_rate: int) -> ValidationItem:
    if not device:
        return ValidationItem("meeting_system_output_loopback", "WARN", "No output loopback device is configured or available.")
    capture = WasapiLoopbackCaptureSource()
    try:
        capture.start(device, sample_rate=sample_rate, chunk_ms=40, channels_policy=ChannelPolicy.STEREO_OR_MONO.value)
        capture.stop()
        return ValidationItem("meeting_system_output_loopback", "PASS", f"WASAPI output loopback capture can start: {device}")
    except Exception as exc:
        return ValidationItem(
            "meeting_system_output_loopback",
            "WARN",
            f"WASAPI output loopback capture could not start on '{device}': {exc}",
        )
    finally:
        try:
            capture.stop()
        except Exception:
            pass


def build_report(config_path: str, *, probe_bridge: bool = True, probe_capture: bool = True) -> ValidationReport:
    items: list[ValidationItem] = [os_item()]
    config = load_config(config_path)
    inputs, outputs, device_error = list_audio_devices()
    if device_error:
        items.append(ValidationItem("devices", "FAIL", f"Cannot list audio devices: {device_error}"))
    else:
        items.append(
            ValidationItem(
                "devices",
                "PASS" if inputs and outputs else "WARN",
                f"Found {len(inputs)} input device(s), {len(outputs)} output device(s).",
                {
                    "inputs": inputs,
                    "outputs": outputs,
                    "default": query_default_devices(),
                },
            )
        )

    input_device = config.meeting.input_device or config.audio.microphone_in or config.audio.meeting_in
    if not input_device and inputs:
        input_device = str(inputs[0].get("name", "") or "")
    loopback_device = config.meeting.output_loopback_device or config.audio.speaker_out
    if not loopback_device and outputs:
        loopback_device = str(outputs[0].get("name", "") or "")
    if probe_capture:
        items.append(_probe_input_capture(input_device, config.runtime.sample_rate))
        items.append(_probe_loopback_capture(loopback_device, config.runtime.sample_rate))
    else:
        items.append(
            ValidationItem(
                "meeting_system_input",
                "PASS" if inputs else "WARN",
                "System input capture has candidate devices." if inputs else "No input device was found for meeting capture.",
            )
        )
        items.append(
            ValidationItem(
                "meeting_system_output_loopback",
                "PASS" if outputs else "WARN",
                "System output loopback has candidate render devices."
                if outputs
                else "No output device was found; WASAPI loopback cannot start until an output endpoint exists.",
            )
        )
    items.append(
        ValidationItem(
            "meeting_virtual_audio_dependency",
            "PASS",
            "Meeting mode does not require SyncTranslate virtual speaker, virtual microphone, or bridge.",
        )
    )

    driver = detect_virtual_audio_install()
    items.append(driver_status_item(driver, required=virtual_audio_required(config)))
    items.append(driver_format_status_item(driver, required=virtual_audio_required(config)))
    items.append(wdk_environment_item())
    bridge_probe = safe_probe_bridge(config.audio.virtual_audio.bridge_path) if probe_bridge else None
    items.append(
        bridge_status_item(
            bridge_probe,
            required=bridge_required(config),
            bridge_path=str(config.audio.virtual_audio.bridge_path or ""),
        )
    )
    dialogue_ok = bool(driver.speaker_available and driver.microphone_available and (bridge_probe.ready if bridge_probe else False))
    items.append(
        ValidationItem(
            "dialogue_mode",
            "PASS" if dialogue_ok else "WARN",
            "Dialogue mode is ready." if dialogue_ok else "Dialogue mode unavailable until virtual driver and bridge checks pass.",
        )
    )
    return ValidationReport(
        "Windows audio runtime validation",
        aggregate_status(items),
        items,
        {
            "config_path": str(Path(config_path)),
            "session_mode": str(getattr(config.runtime, "session_mode", "meeting") or "meeting"),
            "meeting_audio_source": str(config.meeting.audio_source or "system_input"),
            "driver_format_status": next((item.details.get("driver_format_status", item.status) for item in items if item.name == "driver_format_status"), "UNKNOWN"),
            "driver_format_expected": DRIVER_FORMAT_EXPECTED,
            "driver_build_tools_available": inspect_wdk_environment()["status"] == "PASS",
            "wdk_environment_status": inspect_wdk_environment()["status"],
            "protocol_v2_ready": PROTOCOL_V2_READY,
            "pcm16_stereo_boundary_ready": PCM16_STEREO_BOUNDARY_READY,
        },
    )


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    try:
        report = build_report(args.config, probe_bridge=not args.no_bridge_probe, probe_capture=not args.no_capture_probe)
        print_report(report)
        if args.json:
            path = write_json_report(report, args.json)
            print(f"JSON written: {path}")
        return 0 if report.status in {"PASS", "WARN"} else 1
    except Exception as exc:
        print(f"FAIL validation could not complete: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
