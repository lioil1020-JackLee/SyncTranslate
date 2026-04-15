"""RuntimeLogger — 結構化執行期日誌。

支援 JSON Lines 或 key-value text 兩種格式，
可輸出到滾動檔案，不阻塞主流程（非同步 queue-based）。
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RuntimeLogRecord:
    """一筆結構化 runtime log 記錄。"""
    timestamp: float = field(default_factory=time.time)
    module: str = ""
    source: str = ""
    event_type: str = ""
    utterance_id: str = ""
    revision: int = 0
    pipeline_revision: int = 0
    backend_name: str = ""
    queue_size: int = 0
    latency_ms: int = 0
    degradation_level: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class RuntimeLogger:
    """結構化 runtime logger。

    Parameters
    ----------
    log_path:
        輸出檔案路徑；None 表示只用 Python logging。
    format:
        "jsonl" 或 "kv"（key-value text）。
    enabled:
        總開關；False 時所有 log 操作都是 no-op。
    max_queue_size:
        非同步佇列容量；滿時丟棄（不阻塞主執行緒）。
    """

    def __init__(
        self,
        *,
        log_path: Path | str | None = None,
        format: str = "jsonl",
        enabled: bool = True,
        max_queue_size: int = 2048,
    ) -> None:
        self._enabled = enabled
        self._format = format if format in ("jsonl", "kv") else "jsonl"
        self._log_path = Path(log_path) if log_path else None
        self._queue: queue.Queue[RuntimeLogRecord | None] = queue.Queue(maxsize=max_queue_size)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._py_logger = logging.getLogger("synctranslate.runtime")

        if self._enabled and self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._start_worker()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(
        self,
        event_type: str,
        *,
        module: str = "",
        source: str = "",
        utterance_id: str = "",
        revision: int = 0,
        pipeline_revision: int = 0,
        backend_name: str = "",
        queue_size: int = 0,
        latency_ms: int = 0,
        degradation_level: str = "",
        **extra: Any,
    ) -> None:
        """記錄一筆 runtime event。"""
        if not self._enabled:
            return
        record = RuntimeLogRecord(
            module=module,
            source=source,
            event_type=event_type,
            utterance_id=utterance_id,
            revision=revision,
            pipeline_revision=pipeline_revision,
            backend_name=backend_name,
            queue_size=queue_size,
            latency_ms=latency_ms,
            degradation_level=degradation_level,
            extra=dict(extra),
        )
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            pass  # 丟棄，不阻塞主流程

    def stop(self) -> None:
        """停止後台 worker，flush 剩餘記錄。"""
        if self._thread and self._thread.is_alive():
            self._queue.put_nowait(None)  # sentinel
            self._thread.join(timeout=3.0)
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _start_worker(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="RuntimeLogger")
        self._thread.start()

    def _worker(self) -> None:
        assert self._log_path is not None
        try:
            with self._log_path.open("a", encoding="utf-8") as fp:
                while not self._stop_event.is_set():
                    try:
                        record = self._queue.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    if record is None:
                        break
                    fp.write(self._format_record(record) + "\n")
                    fp.flush()
                # flush remaining
                while True:
                    try:
                        record = self._queue.get_nowait()
                        if record is None:
                            break
                        fp.write(self._format_record(record) + "\n")
                    except queue.Empty:
                        break
        except Exception as exc:
            self._py_logger.warning("RuntimeLogger worker error: %s", exc)

    def _format_record(self, record: RuntimeLogRecord) -> str:
        if self._format == "jsonl":
            d = asdict(record)
            d["timestamp"] = record.timestamp
            if not record.extra:
                d.pop("extra", None)
            return json.dumps(d, ensure_ascii=False)
        # kv format
        parts = [
            f"ts={record.timestamp:.3f}",
            f"module={record.module}",
            f"source={record.source}",
            f"event={record.event_type}",
        ]
        if record.utterance_id:
            parts.append(f"utterance={record.utterance_id[:8]}")
        if record.revision:
            parts.append(f"rev={record.revision}")
        if record.latency_ms:
            parts.append(f"latency={record.latency_ms}ms")
        if record.degradation_level:
            parts.append(f"degradation={record.degradation_level}")
        if record.extra:
            for k, v in record.extra.items():
                parts.append(f"{k}={v}")
        return " ".join(parts)


# 模組層級的單例，可由 bootstrap 初始化後替換
_instance: RuntimeLogger | None = None


def get_runtime_logger() -> RuntimeLogger:
    """取得模組層級 RuntimeLogger 單例（若未初始化則回傳 disabled 實例）。"""
    global _instance
    if _instance is None:
        _instance = RuntimeLogger(enabled=False)
    return _instance


def init_runtime_logger(
    *,
    log_path: Path | str | None = None,
    format: str = "jsonl",
    enabled: bool = True,
) -> RuntimeLogger:
    """初始化模組層級 RuntimeLogger。應在 bootstrap 時呼叫一次。"""
    global _instance
    if _instance is not None:
        _instance.stop()
    _instance = RuntimeLogger(log_path=log_path, format=format, enabled=enabled)
    return _instance


__all__ = [
    "RuntimeLogRecord",
    "RuntimeLogger",
    "get_runtime_logger",
    "init_runtime_logger",
]
