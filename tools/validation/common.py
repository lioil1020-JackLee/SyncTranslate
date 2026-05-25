from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import platform
from pathlib import Path
import shutil
from typing import Any

import sounddevice as sd

from app.infra.audio.device_registry import DeviceManager
from app.infra.audio.virtual_bridge_probe import VirtualBridgeProbeResult, probe_virtual_audio_bridge
from app.infra.audio.virtual_devices import VirtualAudioInstallStatus, detect_virtual_audio_install
from app.bootstrap.runtime_assets import resolve_asr_model_path, resolve_llm_model_path
from app.infra.config.schema import AppConfig
from app.infra.config.settings_store import load_config
from tools.validation.wasapi_endpoint_format import query_synctranslate_endpoint_formats


VALIDATION_OUTPUT_DIR = Path("downloads") / "validation"
DRIVER_FORMAT_EXPECTED = "48000Hz PCM16 2ch"
PROTOCOL_V2_READY = True
PCM16_STEREO_BOUNDARY_READY = True


@dataclass(slots=True)
class ValidationItem:
    name: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "details": dict(self.details),
        }


@dataclass(slots=True)
class ValidationReport:
    name: str
    status: str
    items: list[ValidationItem]
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "details": dict(self.details),
            "items": [item.to_dict() for item in self.items],
        }


def aggregate_status(items: list[ValidationItem]) -> str:
    if any(item.status == "FAIL" for item in items):
        return "FAIL"
    if any(item.status == "WARN" for item in items):
        return "WARN"
    return "PASS"


def write_json_report(report: ValidationReport, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def print_report(report: ValidationReport) -> None:
    print(f"{report.status} {report.name}")
    for item in report.items:
        print(f"[{item.status}] {item.name}: {item.message}")


def query_default_devices() -> dict[str, Any]:
    try:
        default_input, default_output = sd.default.device
    except Exception as exc:
        return {"input": None, "output": None, "error": str(exc)}
    return {"input": default_input, "output": default_output, "error": ""}


def list_audio_devices() -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    try:
        manager = DeviceManager()
        inputs = [asdict(item) for item in manager.list_input_devices()]
        outputs = [asdict(item) for item in manager.list_output_devices()]
        return inputs, outputs, ""
    except Exception as exc:
        return [], [], str(exc)


def load_validation_config(config_path: str | Path) -> tuple[AppConfig | None, str]:
    try:
        return load_config(config_path), ""
    except Exception as exc:
        return None, str(exc)


def virtual_audio_required(config: AppConfig) -> bool:
    return str(getattr(config.runtime, "session_mode", "meeting") or "meeting").strip().lower() == "dialogue"


def bridge_required(config: AppConfig) -> bool:
    return virtual_audio_required(config) and bool(config.audio.virtual_audio.bridge_enabled)


def driver_status_item(status: VirtualAudioInstallStatus, *, required: bool) -> ValidationItem:
    ok = bool(status.speaker_available and status.microphone_available)
    speaker_interface_available = bool(getattr(status, "speaker_interface_available", False))
    microphone_interface_available = bool(getattr(status, "microphone_interface_available", False))
    speaker_interface_count = int(getattr(status, "speaker_interface_count", 0) or 0)
    microphone_interface_count = int(getattr(status, "microphone_interface_count", 0) or 0)
    if ok:
        return ValidationItem(
            "driver_status",
            "PASS",
            "SyncTranslate Virtual Speaker and Virtual Microphone are available.",
            {
                "speaker_name": status.speaker_name,
                "microphone_name": status.microphone_name,
                "speaker_index": status.speaker_index,
                "microphone_index": status.microphone_index,
            },
        )
    severity = "FAIL" if required else "WARN"
    return ValidationItem(
        "driver_status",
        severity,
        (
            "SyncTranslate virtual audio KS interfaces are present, but user-mode audio endpoints are not visible in this session."
            if speaker_interface_available and microphone_interface_available
            else "SyncTranslate virtual audio driver is not available. Meeting mode can still run; dialogue mode needs the driver."
        ),
        {
            "speaker_available": status.speaker_available,
            "microphone_available": status.microphone_available,
            "render_endpoint_count": len(status.render_endpoints),
            "capture_endpoint_count": len(status.capture_endpoints),
            "speaker_interface_available": speaker_interface_available,
            "microphone_interface_available": microphone_interface_available,
            "speaker_interface_count": speaker_interface_count,
            "microphone_interface_count": microphone_interface_count,
        },
    )


def bridge_status_item(probe: VirtualBridgeProbeResult | None, *, required: bool, bridge_path: str) -> ValidationItem:
    path_exists = bool(bridge_path) and Path(bridge_path).exists()
    if probe is not None and probe.ready:
        return ValidationItem(
            "bridge_status",
            "PASS",
            "Bridge heartbeat and PCM loopback passed.",
            {
                "bridge_path": bridge_path,
                "bridge_path_exists": path_exists,
                "heartbeat_ok": probe.heartbeat_ok,
                "loopback_ok": probe.loopback_ok,
                "heartbeat_roundtrip_ms": probe.heartbeat_roundtrip_ms,
                "loopback_latency_ms": probe.loopback_latency_ms,
            },
        )
    severity = "FAIL" if required else "WARN"
    error = str(getattr(probe, "error", "") or "bridge not required or not connected")
    return ValidationItem(
        "bridge_status",
        severity,
        "Bridge is not ready. Meeting mode does not require it; dialogue mode requires a working bridge/driver path.",
        {
            "bridge_path": bridge_path,
            "bridge_path_exists": path_exists,
            "error": error,
            "heartbeat_ok": bool(getattr(probe, "heartbeat_ok", False)),
            "loopback_ok": bool(getattr(probe, "loopback_ok", False)),
        },
    )


def _find_tool(name: str, candidates: list[Path] | None = None) -> str:
    found = shutil.which(name)
    if found:
        return found
    for candidate in candidates or []:
        if candidate.exists():
            return str(candidate)
    return ""


def inspect_wdk_environment() -> dict[str, Any]:
    kits_root = Path("C:/Program Files (x86)/Windows Kits/10")
    vs_root = Path("C:/Program Files/Microsoft Visual Studio/2022")
    msbuild_candidates = list(Path("C:/Program Files").glob("Microsoft Visual Studio/2022/*/MSBuild/Current/Bin/MSBuild.exe"))
    msbuild_candidates += list(Path("C:/Program Files (x86)").glob("Microsoft Visual Studio/2022/*/MSBuild/Current/Bin/MSBuild.exe"))
    tool_candidates = list(kits_root.glob("bin/*/x64")) + list(kits_root.glob("Tools/*/x64"))
    tools = {
        "msbuild": _find_tool("msbuild.exe", msbuild_candidates),
        "infverif": _find_tool("infverif.exe", [path / "infverif.exe" for path in tool_candidates]),
        "stampinf": _find_tool("stampinf.exe", [path / "stampinf.exe" for path in tool_candidates]),
        "signtool": _find_tool("signtool.exe", [path / "signtool.exe" for path in tool_candidates]),
    }
    missing = [name for name, path in tools.items() if not path]
    return {
        "status": "PASS" if not missing else "WARN",
        "tools": tools,
        "missing_tools": missing,
        "windows_kits_root_exists": kits_root.exists(),
        "visual_studio_root_exists": vs_root.exists(),
    }


def wdk_environment_item() -> ValidationItem:
    status = inspect_wdk_environment()
    missing = status["missing_tools"]
    return ValidationItem(
        "wdk_environment_status",
        "PASS" if not missing else "WARN",
        "WDK build tools are available." if not missing else f"WDK build tools missing or not on PATH: {', '.join(missing)}.",
        status,
    )


def driver_format_status_item(status: VirtualAudioInstallStatus, *, required: bool) -> ValidationItem:
    speaker_interface_available = bool(getattr(status, "speaker_interface_available", False))
    microphone_interface_available = bool(getattr(status, "microphone_interface_available", False))
    speaker_interface_count = int(getattr(status, "speaker_interface_count", 0) or 0)
    microphone_interface_count = int(getattr(status, "microphone_interface_count", 0) or 0)
    expected = {
        "sample_rate": 48000,
        "bit_depth": 16,
        "channels": 2,
        "dtype": "PCM16",
        "summary": DRIVER_FORMAT_EXPECTED,
    }
    if not status.speaker_available or not status.microphone_available:
        interface_reason = (
            "ks_interfaces_present_but_user_endpoints_hidden"
            if speaker_interface_available and microphone_interface_available
            else "virtual_endpoints_missing"
        )
        severity = "FAIL" if required else "WARN"
        return ValidationItem(
            "driver_format_status",
            severity,
            (
                "Driver KS interfaces exist, but user-mode endpoint format cannot be checked from this session."
                if interface_reason == "ks_interfaces_present_but_user_endpoints_hidden"
                else "Driver format cannot be checked because SyncTranslate virtual endpoints are missing."
            ),
            {
                "expected": expected,
                "driver_format_status": "UNKNOWN",
                "reason": interface_reason,
                "speaker_interface_count": speaker_interface_count,
                "microphone_interface_count": microphone_interface_count,
            },
        )

    speaker = next((item for item in status.render_endpoints if item.index == status.speaker_index), None)
    microphone = next((item for item in status.capture_endpoints if item.index == status.microphone_index), None)
    mismatches: list[str] = []
    if speaker and int(speaker.default_samplerate) != 48000:
        mismatches.append(f"speaker_default_samplerate={speaker.default_samplerate}")
    if microphone and int(microphone.default_samplerate) != 48000:
        mismatches.append(f"microphone_default_samplerate={microphone.default_samplerate}")
    if speaker and int(speaker.max_output_channels) < 2:
        mismatches.append(f"speaker_channels={speaker.max_output_channels}")
    if microphone and int(microphone.max_input_channels) < 2:
        mismatches.append(f"microphone_channels={microphone.max_input_channels}")
    if mismatches:
        return ValidationItem(
            "driver_format_status",
            "FAIL" if required else "WARN",
            f"Driver endpoint format mismatch; expected {DRIVER_FORMAT_EXPECTED}.",
            {"expected": expected, "driver_format_status": "FAIL" if required else "WARN", "mismatches": mismatches},
        )
    wasapi_report = query_synctranslate_endpoint_formats()
    if wasapi_report.endpoints:
        if wasapi_report.status == "PASS":
            return ValidationItem(
                "driver_format_status",
                "PASS",
                wasapi_report.message,
                {
                    "expected": expected,
                    "driver_format_status": "PASS",
                    "endpoint_device_formats": wasapi_report.to_dict()["endpoints"],
                    "shared_mix_formats": wasapi_report.to_dict().get("shared_mix_formats", []),
                },
            )
        if wasapi_report.status == "FAIL":
            severity = "FAIL" if required else "WARN"
            return ValidationItem(
                "driver_format_status",
                severity,
                wasapi_report.message,
                {
                    "expected": expected,
                    "driver_format_status": severity,
                    "endpoint_device_formats": wasapi_report.to_dict()["endpoints"],
                    "shared_mix_formats": wasapi_report.to_dict().get("shared_mix_formats", []),
                },
            )
    return ValidationItem(
        "driver_format_status",
        "WARN",
        "Endpoint sample rate/channel checks match v2 expectations, but PCM bit depth cannot be reliably read from this Python path.",
        {
            "expected": expected,
            "driver_format_status": "UNKNOWN",
            "speaker": asdict(speaker) if speaker else {},
            "microphone": asdict(microphone) if microphone else {},
            "suggested_fix": "Run drivers/synctranslate_virtual_audio/scripts/verify_driver_format.ps1 on the target Windows machine.",
        },
    )


def build_check_report(config_path: str | Path = "config.yaml", *, probe_bridge: bool = True) -> ValidationReport:
    items: list[ValidationItem] = []
    config, config_error = load_validation_config(config_path)
    if config is None:
        items.append(ValidationItem("config", "FAIL", f"Unable to load config: {config_error}"))
        return ValidationReport("SyncTranslate health check", "FAIL", items)

    schema_version = int(getattr(config.runtime, "config_schema_version", 0) or 0)
    session_mode = str(getattr(config.runtime, "session_mode", "meeting") or "meeting")
    migration_note = str(getattr(config.runtime, "last_migration_note", "") or "")
    items.append(
        ValidationItem(
            "config",
            "PASS",
            "Config loaded and migrated to runtime schema.",
            {"config_schema_version": schema_version, "migration_note": migration_note},
        )
    )

    inputs, outputs, device_error = list_audio_devices()
    if device_error:
        items.append(ValidationItem("audio_devices", "WARN", f"Unable to enumerate audio devices: {device_error}"))
    else:
        status = "PASS" if inputs and outputs else "WARN"
        items.append(
            ValidationItem(
                "audio_devices",
                status,
                f"Found {len(inputs)} input device(s) and {len(outputs)} output device(s).",
                {"input_count": len(inputs), "output_count": len(outputs), "default": query_default_devices()},
            )
        )

    required = virtual_audio_required(config)
    try:
        driver_status = detect_virtual_audio_install()
        items.append(driver_status_item(driver_status, required=required))
        items.append(driver_format_status_item(driver_status, required=required))
    except Exception as exc:
        items.append(
            ValidationItem(
                "driver_status",
                "FAIL" if required else "WARN",
                f"Unable to inspect SyncTranslate virtual driver: {exc}",
            )
        )
        items.append(
            ValidationItem(
                "driver_format_status",
                "FAIL" if required else "WARN",
                f"Unable to inspect SyncTranslate driver format: {exc}",
                {"expected": DRIVER_FORMAT_EXPECTED, "driver_format_status": "UNKNOWN"},
            )
        )
    items.append(wdk_environment_item())

    bridge_path = str(config.audio.virtual_audio.bridge_path or "")
    probe = None
    if probe_bridge:
        try:
            probe = probe_virtual_audio_bridge(bridge_path)
        except Exception:
            probe = None
    items.append(bridge_status_item(probe, required=bridge_required(config), bridge_path=bridge_path))

    warnings: list[str] = []
    if config.asr.engine != "faster_whisper":
        warnings.append(f"selected_asr_backend_is_{config.asr.engine}")
        items.append(
            ValidationItem(
                "asr_backend",
                "WARN",
                f"Selected ASR backend is {config.asr.engine}; v2 product path is faster-whisper/CTranslate2.",
            )
        )
    else:
        items.append(ValidationItem("asr_backend", "PASS", "Selected ASR backend is faster-whisper/CTranslate2."))
    asr_asset = resolve_asr_model_path(config.asr.model)
    llm_asset = resolve_llm_model_path(config.llm.runtime.model_path)
    items.append(
        ValidationItem(
            "asr_model",
            "PASS" if asr_asset.exists else "WARN",
            f"ASR model ready: {asr_asset.resolved}" if asr_asset.exists else f"ASR model missing: {asr_asset.resolved}",
            {"suggested_fix": asr_asset.suggested_fix},
        )
    )
    items.append(
        ValidationItem(
            "llm_model",
            "PASS" if llm_asset.exists else "WARN",
            f"LLM model ready: {llm_asset.resolved}" if llm_asset.exists else f"LLM model missing: {llm_asset.resolved}",
            {"suggested_fix": llm_asset.suggested_fix},
        )
    )

    return ValidationReport(
        "SyncTranslate health check",
        aggregate_status(items),
        items,
        {
            "config_path": str(config_path),
            "config_schema_version": schema_version,
            "session_mode": session_mode,
            "meeting_audio_source": str(config.meeting.audio_source or "system_input"),
            "asr_language_mode": "fixed",
            "selected_asr_backend": "faster-whisper" if config.asr.engine == "faster_whisper" else config.asr.engine,
            "virtual_audio_required": required,
            "bridge_required": bridge_required(config),
            "driver_format_expected": DRIVER_FORMAT_EXPECTED,
            "driver_format_status": next((item.status for item in items if item.name == "driver_format_status"), "UNKNOWN"),
            "driver_build_tools_available": inspect_wdk_environment()["status"] == "PASS",
            "wdk_environment_status": inspect_wdk_environment()["status"],
            "protocol_v2_ready": PROTOCOL_V2_READY,
            "pcm16_stereo_boundary_ready": PCM16_STEREO_BOUNDARY_READY,
            "config_migration_status": migration_note or "current",
            "validation_warnings": warnings,
            "asr_model_ready": asr_asset.exists,
            "llm_model_ready": llm_asset.exists,
        },
    )


def os_item() -> ValidationItem:
    system = platform.system()
    if system == "Windows":
        return ValidationItem("os", "PASS", "Windows detected.", {"system": system, "release": platform.release()})
    return ValidationItem(
        "os",
        "WARN",
        "This validation tool is intended for Windows. Non-Windows systems can only run limited checks.",
        {"system": system, "release": platform.release()},
    )


def safe_probe_bridge(bridge_path: str) -> VirtualBridgeProbeResult | None:
    try:
        return probe_virtual_audio_bridge(bridge_path)
    except Exception:
        return None
