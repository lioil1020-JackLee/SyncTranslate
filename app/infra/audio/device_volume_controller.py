from __future__ import annotations

import logging
from pathlib import Path
import re
import subprocess
import sys

from app.infra.audio.device_registry import canonical_device_name
from app.infra.config.schema import AudioRouteConfig


# P0-4: 白名單 flow 值；device_name 允許常見音訊裝置字符（不允許引號/換行等注入字符）
_SAFE_FLOW_VALUES: frozenset[str] = frozenset({"capture", "render"})
_SAFE_DEVICE_NAME_RE = re.compile(r"^[^\x00-\x1f\x7f\"\'\`\|\;\&\$\<\>]+$")
_log = logging.getLogger(__name__)


class SystemDeviceVolumeController:
    def __init__(self, script_path: Path | None = None) -> None:
        self._script_path = script_path or self._default_script_path()

    def apply_audio_route_config(self, audio: AudioRouteConfig) -> None:
        del audio
        # Route-level gain/volume control was removed; keep this method as a no-op
        # for backward compatibility with older call sites.
        return

    def set_input_volume(self, selector: str, scalar: float) -> None:
        self._set_endpoint_volume(flow="capture", selector=selector, scalar=scalar)

    def set_output_volume(self, selector: str, scalar: float) -> None:
        self._set_endpoint_volume(flow="render", selector=selector, scalar=scalar)

    def _set_endpoint_volume(self, *, flow: str, selector: str, scalar: float) -> None:
        if sys.platform != "win32":
            raise RuntimeError("Real device volume control is only supported on Windows.")
        # P0-4: 驗證 flow 防止注入
        if flow not in _SAFE_FLOW_VALUES:
            raise ValueError(f"Invalid flow value: {flow!r}")
        device_name = canonical_device_name(selector).strip()
        if not device_name:
            return
        # P0-4: 驗證 device_name 防止 PowerShell 參數注入
        if not _SAFE_DEVICE_NAME_RE.match(device_name):
            _log.warning("Device name contains unsafe characters, skip volume apply: %r", device_name)
            return
        clamped = max(0.0, min(1.0, float(scalar)))
        command = [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(self._script_path),
            "-Flow",
            flow,
            "-DeviceName",
            device_name,
            "-VolumePercent",
            str(int(round(clamped * 100.0))),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            **self._hidden_subprocess_kwargs(),
        )
        if completed.returncode == 0:
            return
        detail = (completed.stderr or completed.stdout or "").strip()
        if not detail:
            detail = f"Unable to set {flow} device volume for '{device_name}'."
        raise RuntimeError(detail)

    @staticmethod
    def _default_script_path() -> Path:
        if getattr(sys, "frozen", False):
            base = Path(getattr(sys, "_MEIPASS", Path.cwd()))
            return base / "app" / "infra" / "audio" / "windows_endpoint_volume.ps1"
        return Path(__file__).with_name("windows_endpoint_volume.ps1")

    @staticmethod
    def _hidden_subprocess_kwargs() -> dict[str, object]:
        if sys.platform != "win32":
            return {}
        kwargs: dict[str, object] = {
            "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),
        }
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0
        kwargs["startupinfo"] = startupinfo
        return kwargs


__all__ = ["SystemDeviceVolumeController"]
