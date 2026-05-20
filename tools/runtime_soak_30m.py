from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path


def _now_iso() -> str:
    return datetime.now().isoformat()


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _memory_snapshot() -> dict[str, object]:
    rss_mb = None
    try:
        import psutil  # type: ignore

        rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        rss_mb = None
    return {
        "rss_mb": float(rss_mb) if rss_mb is not None else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 30-minute runtime soak test and export JSON report.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--duration-sec", type=int, default=1800, help="Soak duration in seconds (default: 1800)")
    parser.add_argument("--sample-interval-sec", type=float, default=5.0, help="Sampling interval in seconds")
    parser.add_argument("--local-asr-language", default="", help="Override local ASR language (e.g. none)")
    parser.add_argument("--remote-asr-language", default="", help="Override remote ASR language")
    parser.add_argument("--tts-output-mode", default="", help="Override TTS output mode (e.g. subtitle_only)")
    args = parser.parse_args()

    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QTimer
    from app.bootstrap.external_runtime import configure_external_ai_runtime
    from app.ui.main_window import MainWindow

    configure_external_ai_runtime()
    app = QApplication(sys.argv)
    window = MainWindow(str(args.config))

    cfg = getattr(window, "config", None)
    if cfg is not None:
        if str(args.local_asr_language or "").strip():
            cfg.runtime.local_asr_language = str(args.local_asr_language).strip()
        if str(args.remote_asr_language or "").strip():
            cfg.runtime.remote_asr_language = str(args.remote_asr_language).strip()
        if str(args.tts_output_mode or "").strip():
            cfg.runtime.tts_output_mode = str(args.tts_output_mode).strip()
        # Rebuild runtime so override values are used by newly created workers.
        try:
            window._runtime_facade.mark_dirty()
            window._ensure_pipelines_ready()
        except Exception as exc:
            print(f"runtime_rebuild_failed: {exc}", file=sys.stderr)

    window.show()

    started_at = time.time()
    deadline = started_at + max(5, int(args.duration_sec))
    samples: list[dict[str, object]] = []
    events: list[str] = []
    failures: list[str] = []
    session_stop_payload: dict[str, object] = {}
    stop_requested = False
    stop_deadline = 0.0
    finished = False

    try:
        window.start_session()
        events.append(f"{_now_iso()} session_start_called")
    except Exception as exc:
        failures.append(f"start_session_failed: {exc}")

    sample_timer = QTimer()
    sample_timer.setInterval(max(200, int(float(args.sample_interval_sec) * 1000.0)))

    def _collect_sample() -> None:
        nonlocal stop_requested, stop_deadline, finished
        if finished:
            return
        now = time.time()
        sample: dict[str, object] = {
            "ts": _now_iso(),
            "elapsed_sec": round(now - started_at, 3),
            "memory": _memory_snapshot(),
        }
        try:
            if window.audio_router is not None:
                router_stats = window.audio_router.stats()
                sample["router_running"] = bool(router_stats.running)
                sample["active_sources"] = list(router_stats.active_sources)
                sample["asr"] = {
                    "remote_queue": _safe_int((router_stats.asr.get("remote") or {}).get("queue_size", 0)),
                    "local_queue": _safe_int((router_stats.asr.get("local") or {}).get("queue_size", 0)),
                    "remote_drop": _safe_int((router_stats.asr.get("remote") or {}).get("dropped_chunks", 0)),
                    "local_drop": _safe_int((router_stats.asr.get("local") or {}).get("dropped_chunks", 0)),
                }
                sample["tts"] = {
                    "queue_depth": _safe_int((router_stats.tts or {}).get("queue_depth", 0)),
                    "drop_local": _safe_int((router_stats.tts or {}).get("drop_count_local", 0)),
                    "drop_remote": _safe_int((router_stats.tts or {}).get("drop_count_remote", 0)),
                }
                bridge = getattr(router_stats, "bridge", {}) or {}
                sample["bridge"] = {
                    "connected": bool(bridge.get("connected", False)),
                    "remote_input_running": bool(bridge.get("remote_input_running", False)),
                    "remote_input_frames": _safe_int(bridge.get("remote_input_frames", 0)),
                    "remote_input_buffered_frames": _safe_int(bridge.get("remote_input_buffered_frames", 0)),
                    "remote_input_capacity_frames": _safe_int(bridge.get("remote_input_buffer_capacity_frames", 0)),
                    "virtual_mic_frames": _safe_int(bridge.get("virtual_microphone_frames", 0)),
                    "virtual_mic_buffered_frames": _safe_int(bridge.get("virtual_microphone_buffered_frames", 0)),
                    "virtual_mic_capacity_frames": _safe_int(bridge.get("virtual_microphone_buffer_capacity_frames", 0)),
                    "virtual_mic_dropped_frames": _safe_int(bridge.get("virtual_microphone_dropped_frames", 0)),
                    "sink_dropped_frames": _safe_int(bridge.get("sink_dropped_frames", 0)),
                    "sink_write_failures": _safe_int(bridge.get("sink_write_failures", 0)),
                    "sink_backpressure_flushes": _safe_int(bridge.get("sink_backpressure_flushes", 0)),
                    "last_error": str(bridge.get("last_error", "") or ""),
                    "sink_last_error": str(bridge.get("sink_last_error", "") or ""),
                }
                latest_latency = (router_stats.latency or [{}])[0] if router_stats.latency else {}
                sample["latency"] = {
                    "asr_final_ms": _safe_float(latest_latency.get("speech_end_to_asr_final_ms", 0.0)),
                    "llm_final_ms": _safe_float(latest_latency.get("asr_final_to_llm_final_ms", 0.0)),
                    "tts_start_ms": _safe_float(latest_latency.get("tts_enqueue_to_playback_start_ms", 0.0)),
                }
        except Exception as exc:
            failures.append(f"sample_failed: {exc}")
        samples.append(sample)

        is_running = bool(window.session_controller and window.session_controller.is_running())
        if now >= deadline and not stop_requested:
            stop_requested = True
            stop_deadline = now + 30.0
            try:
                if is_running:
                    window.start_session()  # toggle stop asynchronously
                    events.append(f"{_now_iso()} session_stop_called")
                else:
                    events.append(f"{_now_iso()} session_already_stopped")
            except Exception as exc:
                failures.append(f"stop_session_failed: {exc}")

        if stop_requested:
            if (not bool(getattr(window, "_session_action_running", False))) and (not is_running):
                finished = True
                sample_timer.stop()
                try:
                    if window.session_controller is not None:
                        session_stop_payload.update(
                            {
                                "ok": True,
                                "message": "session stopped",
                                "payload": {},
                            }
                        )
                except Exception:
                    pass
                app.quit()
                return
            if now >= stop_deadline:
                failures.append("stop_session_timeout")
                finished = True
                sample_timer.stop()
                app.quit()

    sample_timer.timeout.connect(_collect_sample)
    sample_timer.start()
    app.exec()

    window.close()

    report_dir = Path("logs") / "session_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = report_dir / f"runtime_soak_report_{stamp}.json"

    report = {
        "kind": "runtime_soak",
        "version": 1,
        "started_at": datetime.fromtimestamp(started_at).isoformat(),
        "ended_at": _now_iso(),
        "duration_sec_requested": int(args.duration_sec),
        "duration_sec_actual": round(time.time() - started_at, 3),
        "sample_interval_sec": float(args.sample_interval_sec),
        "config_path": str(args.config),
        "events": events,
        "failures": failures,
        "session_stop": session_stop_payload,
        "samples": samples,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
