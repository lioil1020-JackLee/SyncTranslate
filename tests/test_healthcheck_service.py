from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from app.application.healthcheck_service import HealthCheckService
from app.application.settings_service import SettingsService


def test_healthcheck_service_uses_main_entrypoint_for_dev_subprocess() -> None:
    service = HealthCheckService(settings_service=SettingsService("config.yaml"))

    with (
        patch("app.application.healthcheck_service.subprocess.run") as mock_run,
        patch.object(HealthCheckService, "_hidden_subprocess_kwargs", return_value={}),
    ):
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="{}", stderr="")
        service._run_subprocess(snapshot_path="snapshot.yaml")

    args = mock_run.call_args.args[0]
    kwargs = mock_run.call_args.kwargs

    assert args[0] == sys.executable
    assert Path(args[1]).name == "main.py"
    assert args[2:] == ["--healthcheck-worker", "--healthcheck-config", "snapshot.yaml"]
    assert kwargs["cwd"] == str(service._project_root())
    assert kwargs["encoding"] == "utf-8"
    assert kwargs["errors"] == "replace"
    assert kwargs["env"]["PYTHONUTF8"] == "1"
    assert kwargs["env"]["PYTHONIOENCODING"] == "utf-8"


def test_healthcheck_service_summarizes_traceback_for_ui() -> None:
    text = "\n".join(
        [
            "Traceback (most recent call last):",
            '  File "<frozen runpy>", line 189, in _run_module_as_main',
            "ModuleNotFoundError: No module named app.local_ai.healthcheck_worker",
        ]
    )

    assert HealthCheckService._summarize_subprocess_error(text) == (
        "健康檢查子程序啟動失敗：ModuleNotFoundError: No module named app.local_ai.healthcheck_worker"
    )
