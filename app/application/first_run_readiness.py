from __future__ import annotations

from dataclasses import dataclass, field
import platform
from pathlib import Path
from typing import Any

from app.bootstrap.runtime_assets import resolve_asr_model_path, resolve_llm_model_path, resolve_runtime_dir
from app.domain.version import build_metadata
from app.infra.audio.device_registry import DeviceManager
from app.infra.audio.virtual_bridge_probe import probe_virtual_audio_bridge
from app.infra.audio.virtual_devices import detect_virtual_audio_install
from app.infra.config.schema import AppConfig


@dataclass(slots=True)
class ReadinessItem:
    name: str
    status: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)
    suggested_fix: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "detail": dict(self.detail),
            "suggested_fix": self.suggested_fix,
        }


@dataclass(slots=True)
class ReadinessReport:
    status: str
    items: list[ReadinessItem]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "summary": dict(self.summary),
            "items": [item.to_dict() for item in self.items],
        }


def evaluate_first_run_readiness(
    config: AppConfig,
    *,
    repo_root: str | Path | None = None,
    onedir_root: str | Path | None = None,
    probe_bridge: bool = True,
) -> ReadinessReport:
    items: list[ReadinessItem] = []
    system = platform.system()
    items.append(
        ReadinessItem(
            "os",
            "PASS" if system == "Windows" else "WARN",
            "Windows detected." if system == "Windows" else "This app is productized for Windows.",
            {"system": system, "release": platform.release()},
        )
    )
    shared = resolve_runtime_dir("shared", repo_root=repo_root, onedir_root=onedir_root)
    fw = resolve_runtime_dir("faster_whisper", repo_root=repo_root, onedir_root=onedir_root)
    items.append(_asset_item("python_runtime", shared, required=True, label="Shared Python runtime"))
    items.append(_asset_item("faster_whisper_runtime", fw, required=True, label="faster-whisper runtime"))
    asr = resolve_asr_model_path(config.asr.model, repo_root=repo_root, onedir_root=onedir_root)
    llm = resolve_llm_model_path(config.llm.runtime.model_path, repo_root=repo_root, onedir_root=onedir_root)
    items.append(_asset_item("asr_model", asr, required=True, label="faster-whisper ASR model"))
    items.append(_asset_item("llm_model", llm, required=True, label="LLM GGUF model"))

    try:
        manager = DeviceManager()
        inputs = manager.list_input_devices()
        outputs = manager.list_output_devices()
        device_error = ""
    except Exception as exc:
        inputs = []
        outputs = []
        device_error = str(exc)
    meeting_ready = bool(inputs or outputs) and asr.exists and llm.exists
    items.append(
        ReadinessItem(
            "meeting_mode",
            "PASS" if meeting_ready else "FAIL",
            "Meeting mode available; virtual audio driver is not required."
            if meeting_ready
            else "Meeting mode needs at least one audio input/output device plus ASR and LLM models.",
            {"input_count": len(inputs), "output_count": len(outputs), "device_error": device_error},
            "Connect a microphone or output device, then run .\\tools\\runtime_setup\\prepare_external_runtimes.ps1",
        )
    )

    driver_ready = False
    try:
        driver = detect_virtual_audio_install()
        driver_ready = bool(driver.speaker_available and driver.microphone_available)
        items.append(
            ReadinessItem(
                "virtual_driver",
                "PASS" if driver_ready else "WARN",
                "SyncTranslate virtual speaker and microphone are available."
                if driver_ready
                else "Dialogue mode needs SyncTranslate Virtual Speaker and Virtual Microphone.",
                {
                    "speaker_available": driver.speaker_available,
                    "microphone_available": driver.microphone_available,
                    "speaker_name": driver.speaker_name,
                    "microphone_name": driver.microphone_name,
                },
                "Install the SyncTranslate Virtual Audio Driver and enable Windows Test Mode if required.",
            )
        )
    except Exception as exc:
        items.append(ReadinessItem("virtual_driver", "WARN", f"Driver status unavailable: {exc}"))

    bridge_path = Path(str(config.audio.virtual_audio.bridge_path or ""))
    bridge_exe_ready = bridge_path.exists()
    if not bridge_exe_ready:
        # The configured relative path may be relative to repo/onedir root.
        for root in (repo_root, onedir_root, Path.cwd()):
            if root and (Path(root) / bridge_path).exists():
                bridge_exe_ready = True
                break
    bridge_ready = False
    bridge_detail: dict[str, Any] = {"bridge_path": str(config.audio.virtual_audio.bridge_path), "bridge_exe_exists": bridge_exe_ready}
    if probe_bridge and bridge_exe_ready:
        try:
            probe = probe_virtual_audio_bridge(str(config.audio.virtual_audio.bridge_path))
            bridge_ready = bool(probe.ready)
            bridge_detail.update(
                {
                    "heartbeat_ok": probe.heartbeat_ok,
                    "loopback_ok": probe.loopback_ok,
                    "error": probe.error,
                }
            )
        except Exception as exc:
            bridge_detail.update({"heartbeat_ok": False, "loopback_ok": False, "error": str(exc)})
    items.append(
        ReadinessItem(
            "bridge",
            "PASS" if bridge_ready else "WARN",
            "Bridge heartbeat and PCM loopback are ready."
            if bridge_ready
            else "Dialogue mode needs bridge heartbeat and PCM loopback; meeting mode does not.",
            bridge_detail,
            "Run .\\tools\\validation\\validate_windows_audio_runtime.py after installing the driver.",
        )
    )

    dialogue_ready = bool(driver_ready and bridge_ready and asr.exists and llm.exists)
    items.append(
        ReadinessItem(
            "dialogue_mode",
            "PASS" if dialogue_ready else "WARN",
            "Dialogue mode available." if dialogue_ready else "Dialogue mode unavailable until driver, bridge, ASR, and LLM checks pass.",
            {},
            "Install the virtual driver, verify bridge readiness, and run runtime setup.",
        )
    )
    schema_ok = int(getattr(config.runtime, "config_schema_version", 0) or 0) == 7
    migration_note = str(getattr(config.runtime, "last_migration_note", "") or "")
    items.append(
        ReadinessItem(
            "config_schema",
            "PASS" if schema_ok else "WARN",
            "Config schema is v7." if schema_ok else "Config should be migrated to schema v7.",
            {"config_schema_version": int(getattr(config.runtime, "config_schema_version", 0) or 0), "migration_note": migration_note},
        )
    )
    session_mode = str(getattr(config.runtime, "session_mode", "meeting") or "meeting")
    current_ready = meeting_ready if session_mode == "meeting" else dialogue_ready
    items.append(
        ReadinessItem(
            "current_session_mode",
            "PASS" if current_ready else "FAIL",
            f"Current session mode '{session_mode}' is ready." if current_ready else f"Current session mode '{session_mode}' is missing required runtime items.",
        )
    )
    status = "FAIL" if any(item.status == "FAIL" for item in items) else "WARN" if any(item.status == "WARN" for item in items) else "PASS"
    missing = [item.name for item in items if item.status in {"WARN", "FAIL"}]
    next_action = _suggest_next_action(items, session_mode=session_mode)
    meta = build_metadata(
        config_schema_version=int(getattr(config.runtime, "config_schema_version", 0) or 0),
        runtime_mode=session_mode,
    )
    return ReadinessReport(
        status,
        items,
        {
            "meeting_ready": meeting_ready,
            "dialogue_ready": dialogue_ready,
            "asr_model_ready": asr.exists,
            "llm_model_ready": llm.exists,
            "bridge_ready": bridge_ready,
            "driver_ready": driver_ready,
            "suggested_next_action": next_action,
            "missing_items": missing,
            "build": meta.to_dict(),
        },
    )


def _asset_item(name: str, asset, *, required: bool, label: str) -> ReadinessItem:
    return ReadinessItem(
        name,
        "PASS" if asset.exists else "FAIL" if required else "WARN",
        f"{label} found." if asset.exists else f"{label} missing: {asset.resolved}",
        {"configured": asset.configured, "resolved": str(asset.resolved), "exists": asset.exists},
        asset.suggested_fix,
    )


def _suggest_next_action(items: list[ReadinessItem], *, session_mode: str) -> str:
    for name in ("asr_model", "llm_model", "python_runtime", "faster_whisper_runtime"):
        item = next((entry for entry in items if entry.name == name and entry.status == "FAIL"), None)
        if item:
            return item.suggested_fix
    if session_mode == "dialogue":
        for name in ("virtual_driver", "bridge"):
            item = next((entry for entry in items if entry.name == name and entry.status != "PASS"), None)
            if item:
                return item.suggested_fix
    return "Meeting mode is available. Start with meeting captions, or run validation tools for dialogue mode."


__all__ = ["ReadinessItem", "ReadinessReport", "evaluate_first_run_readiness"]
