from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from app.application.transcript_service import TranscriptBuffer
from app.bootstrap.dependency_container import build_pipeline_bundle
from app.bootstrap.external_runtime import configure_external_ai_runtime
from app.infra.audio.capture import AudioCapture
from app.infra.audio.playback import AudioPlayback
from app.infra.config.schema import AudioRouteConfig
from app.infra.config.settings_store import load_config


def _inject_tone(client, sample_rate: int, duration_sec: float = 2.0, tone_hz: float = 440.0) -> int:
    frames = max(1, int(sample_rate * duration_sec))
    t = np.arange(frames, dtype=np.float32) / float(sample_rate)
    tone = (np.sin(2.0 * np.pi * float(tone_hz) * t) * 0.2).reshape((-1, 1))
    chunk = max(1, int(sample_rate * 0.02))
    sent = 0
    for i in range(0, tone.shape[0], chunk):
        seg = tone[i : i + chunk]
        client.debug_inject_remote_input(seg, sample_rate=int(sample_rate))
        sent += int(seg.shape[0])
        time.sleep(seg.shape[0] / max(1, sample_rate))
    return sent


def run_d5_resampling_validation(config_path: str, out_json: str | None = None) -> dict[str, object]:
    configure_external_ai_runtime()
    cfg = load_config(config_path)
    cfg.runtime.local_asr_language = "none"
    cfg.runtime.tts_output_mode = "subtitle_only"
    cfg.audio.virtual_audio.require_driver = False

    bundle = build_pipeline_bundle(
        config=cfg,
        pipeline_revision=1405,
        transcript_buffer=TranscriptBuffer(max_items=None),
        local_capture=AudioCapture(),
        speaker_playback=AudioPlayback(),
        get_local_output_device=lambda: "",
        on_error=lambda e: None,
        on_diagnostic_event=lambda m: None,
        on_asr_event=lambda e: None,
    )
    session = bundle.session_controller
    router = bundle.audio_router

    routes = AudioRouteConfig(
        meeting_in="dummy-remote",
        microphone_in="dummy-local",
        speaker_out="dummy-spk",
        meeting_out="dummy-meet",
    )
    start = session.start(routes, sample_rate=cfg.runtime.sample_rate, chunk_ms=cfg.runtime.chunk_ms, mode="meeting_to_local")

    remote_source = router._input_manager._captures.get("remote")
    client = getattr(remote_source, "_bridge", None)
    if client is None:
        result = {
            "kind": "d5_resampling_validation",
            "pass_d5": False,
            "start_ok": bool(start.ok),
            "error": "remote_bridge_client_unavailable",
        }
        if out_json:
            Path(out_json).parent.mkdir(parents=True, exist_ok=True)
            Path(out_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    before = router.stats()
    before_asr = (before.asr or {}).get("remote") or {}
    before_capture = (before.capture or {}).get("remote") or {}

    sr_results: list[dict[str, object]] = []
    for sr in (48000, 16000):
        sent = _inject_tone(client, sample_rate=sr, duration_sec=2.0)
        time.sleep(2.0)
        stats = router.stats()
        remote_asr = (stats.asr or {}).get("remote") or {}
        remote_capture = (stats.capture or {}).get("remote") or {}
        sr_results.append(
            {
                "sample_rate": sr,
                "frames_sent": sent,
                "capture_frames": int(remote_capture.get("frame_count", 0) or 0),
                "asr_partial": int(remote_asr.get("partial_count", 0) or 0),
                "asr_final": int(remote_asr.get("final_count", 0) or 0),
                "init_failure": str(remote_asr.get("init_failure", "") or ""),
            }
        )

    time.sleep(8.0)
    final_stats = router.stats()
    remote_final = (final_stats.asr or {}).get("remote") or {}
    remote_capture_final = (final_stats.capture or {}).get("remote") or {}

    stop = session.stop()

    total_capture_delta = int(remote_capture_final.get("frame_count", 0) or 0) - int(before_capture.get("frame_count", 0) or 0)
    total_final_delta = int(remote_final.get("final_count", 0) or 0) - int(before_asr.get("final_count", 0) or 0)

    pass_d5 = bool(
        start.ok
        and stop.ok
        and total_capture_delta > 0
        and total_final_delta >= 1
        and all(not str(item.get("init_failure", "")) for item in sr_results)
    )

    result = {
        "kind": "d5_resampling_validation",
        "pass_d5": pass_d5,
        "start_ok": bool(start.ok),
        "stop_ok": bool(stop.ok),
        "capture_frames_delta": int(total_capture_delta),
        "final_count_delta": int(total_final_delta),
        "per_sample_rate": sr_results,
    }
    if out_json:
        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate D5 48k/16k resampling path.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="logs/session_reports/d5_resampling_validation.json")
    args = parser.parse_args()

    result = run_d5_resampling_validation(args.config, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if bool(result.get("pass_d5", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
