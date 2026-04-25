from __future__ import annotations

import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from threading import Lock, Thread
import time
from dataclasses import dataclass

from app.application.settings_service import SettingsService
from app.local_ai.healthcheck import LocalHealthReport
from app.infra.config.schema import AppConfig


@dataclass(slots=True)
class HealthCheckUpdate:
    kind: str
    message: str
    report: LocalHealthReport | None = None


class HealthCheckService:
    def __init__(self, *, settings_service: SettingsService, timeout_sec: float = 45.0) -> None:
        self._settings_service = settings_service
        self._timeout_sec = max(1.0, float(timeout_sec))
        self._queue: Queue[tuple[bool, object]] = Queue()
        self._lock = Lock()
        self._running = False
        self._started_at = 0.0

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def timeout_sec(self) -> float:
        return self._timeout_sec

    def start(self, *, config: AppConfig) -> bool:
        with self._lock:
            if self._running:
                return False
            self._running = True
            self._started_at = time.monotonic()
            self._clear_queue_locked()

        snapshot_path = self._settings_service.save_snapshot(config, prefix="healthcheck_")
        Thread(target=self._worker, args=(snapshot_path,), daemon=True).start()
        return True

    def poll(self) -> HealthCheckUpdate | None:
        with self._lock:
            if not self._running:
                return None
            elapsed = time.monotonic() - self._started_at if self._started_at else 0.0
            if elapsed >= self._timeout_sec:
                self._running = False
                self._started_at = 0.0
                return HealthCheckUpdate(
                    kind="timeout",
                    message=f"健康檢查逾時，超過 {int(self._timeout_sec)} 秒",
                )

        try:
            ok, payload = self._queue.get_nowait()
        except Empty:
            return None

        with self._lock:
            self._running = False
            self._started_at = 0.0

        if not ok:
            return HealthCheckUpdate(kind="error", message=str(payload))

        if isinstance(payload, LocalHealthReport):
            summary = "健康檢查：正常" if payload.ok else "健康檢查：有異常"
            return HealthCheckUpdate(kind="success", message=summary, report=payload)
        return HealthCheckUpdate(kind="error", message="健康檢查回傳格式錯誤")

    def _worker(self, snapshot_path: str) -> None:
        try:
            completed = self._run_subprocess(snapshot_path=snapshot_path)
            if completed.returncode != 0:
                stderr = (completed.stderr or "").strip()
                stdout = (completed.stdout or "").strip()
                detail = self._summarize_subprocess_error(
                    stderr or stdout or f"健康檢查子程序結束，代碼 {completed.returncode}"
                )
                raise RuntimeError(detail)
            payload = (completed.stdout or "").strip()
            if not payload:
                raise RuntimeError("健康檢查子程序未回傳結果")
            report = LocalHealthReport(**json.loads(payload))
            self._queue.put((True, report))
        except Exception as exc:
            self._queue.put((False, exc))
        finally:
            try:
                os.unlink(snapshot_path)
            except Exception:
                pass

    @staticmethod
    def _project_root() -> Path:
        # app/application/healthcheck_service.py -> repo root
        return Path(__file__).resolve().parent.parent.parent

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

    @staticmethod
    def _summarize_subprocess_error(text: str) -> str:
        lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
        if not lines:
            return "健康檢查子程序失敗"
        if lines[0].startswith("Traceback"):
            return f"健康檢查子程序啟動失敗：{lines[-1]}"
        return lines[0] if len(lines) == 1 else f"{lines[0]} | {lines[-1]}"

    def _run_subprocess(self, *, snapshot_path: str) -> subprocess.CompletedProcess[str]:
        hidden_kwargs = self._hidden_subprocess_kwargs()
        env = os.environ.copy()
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        if getattr(sys, "frozen", False):
            return subprocess.run(
                [
                    sys.executable,
                    "--healthcheck-worker",
                    "--healthcheck-config",
                    snapshot_path,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                **hidden_kwargs,
            )
        entrypoint = self._project_root() / "main.py"
        return subprocess.run(
            [
                sys.executable,
                str(entrypoint),
                "--healthcheck-worker",
                "--healthcheck-config",
                snapshot_path,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            cwd=str(self._project_root()),
            **hidden_kwargs,
        )

    def _clear_queue_locked(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                return


__all__ = ["HealthCheckUpdate", "HealthCheckService"]
