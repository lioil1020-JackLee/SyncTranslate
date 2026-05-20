"""Multi-route runtime verification for Phase 4 integration."""

from __future__ import annotations

import argparse
import json
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

from app.application.transcript_service import TranscriptBuffer
from app.bootstrap.dependency_container import build_pipeline_bundle
from app.bootstrap.external_runtime import configure_external_ai_runtime
from app.infra.audio.capture import AudioCapture
from app.infra.audio.playback import AudioPlayback
from app.infra.config.schema import AudioRouteConfig
from app.infra.config.settings_store import load_config


def _to_float32_pcm(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = int(wf.getframerate())
        sample_width = wf.getsampwidth()
        frame_count = wf.getnframes()
        raw = wf.readframes(frame_count)
    if sample_width == 2:
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    audio = audio.reshape(-1, channels) if channels > 1 else audio.reshape(-1, 1)
    return audio, sample_rate


def run_multi_route_test(
    config_path: str,
    wav_path: str,
    loops: int = 2,
    observe_sec: float = 12.0,
    out_json: str | None = None,
    profile_name: str = "default",
) -> dict[str, Any]:
    configure_external_ai_runtime()
    cfg = load_config(config_path)
    cfg.runtime.local_asr_language = "zh"
    cfg.runtime.tts_output_mode = "subtitle_only"
    cfg.audio.virtual_audio.require_driver = False

    routes = AudioRouteConfig(
        meeting_in="dummy-remote",
        microphone_in="dummy-local",
        speaker_out="dummy-spk",
        meeting_out="dummy-meet",
    )
    bundle = build_pipeline_bundle(
        config=cfg,
        pipeline_revision=1200,
        transcript_buffer=TranscriptBuffer(max_items=None),
        local_capture=AudioCapture(),
        speaker_playback=AudioPlayback(),
        get_local_output_device=lambda: "",
        on_error=lambda e: None,
        on_diagnostic_event=lambda m: None,
        on_asr_event=lambda e: None,
    )
    session_controller = bundle.session_controller
    router = bundle.audio_router

    start_result = session_controller.start(
        routes,
        sample_rate=cfg.runtime.sample_rate,
        chunk_ms=cfg.runtime.chunk_ms,
        mode="meeting_to_local",
    )

    stream = {
        "frames_sent": 0,
        "loops": int(loops),
        "wav_exists": Path(wav_path).exists(),
        "observe_sec": float(observe_sec),
    }

    remote_source = router._input_manager._captures.get("remote")
    client = getattr(remote_source, "_bridge", None)

    if stream["wav_exists"] and client is not None:
        audio, sample_rate = _to_float32_pcm(Path(wav_path))
        chunk_frames = 960
        for _ in range(int(loops)):
            for i in range(0, int(audio.shape[0]), chunk_frames):
                segment = audio[i : i + chunk_frames]
                client.debug_inject_remote_input(segment, sample_rate=int(sample_rate))
                stream["frames_sent"] += int(segment.shape[0])
                time.sleep(segment.shape[0] / max(1, sample_rate))
            time.sleep(0.05)

    deadline = time.time() + max(1.0, float(observe_sec))
    while time.time() < deadline:
        time.sleep(0.2)

    stats = router.stats()
    remote = stats.asr.get("remote", {})
    local = stats.asr.get("local", {})
    tts = stats.tts or {}

    stop_t = time.time()
    stop_result = session_controller.stop()
    stop_seconds = time.time() - stop_t

    result: dict[str, Any] = {
        "profile": profile_name,
        "start_ok": bool(start_result.ok),
        "stop_ok": bool(stop_result.ok),
        "stop_seconds": round(stop_seconds, 3),
        "remote_partial": int(remote.get("partial_count", 0) or 0),
        "remote_final": int(remote.get("final_count", 0) or 0),
        "local_partial": int(local.get("partial_count", 0) or 0),
        "local_final": int(local.get("final_count", 0) or 0),
        "remote_init_failure": str(remote.get("init_failure", "") or ""),
        "local_init_failure": str(local.get("init_failure", "") or ""),
        "remote_model_init_mode": str(remote.get("model_init_mode", "") or ""),
        "local_model_init_mode": str(local.get("model_init_mode", "") or ""),
        "bridge_remote_input_frames": int((stats.bridge or {}).get("remote_input_frames", 0) or 0),
        "remote_capture_frames": int((stats.capture.get("remote") or {}).get("frame_count", 0) or 0),
        "local_capture_frames": int((stats.capture.get("local") or {}).get("frame_count", 0) or 0),
        "tts_queue_depth": int(tts.get("queue_depth", 0) or 0),
        "tts_drop_count_local": int(tts.get("drop_count_local", 0) or 0),
        "tts_drop_count_remote": int(tts.get("drop_count_remote", 0) or 0),
        "stream": stream,
    }
    result["pass_d1"] = bool(result["start_ok"] and result["stop_ok"] and result["remote_partial"] >= 1 and result["remote_final"] >= 1)
    result["pass_remote_tts"] = bool(result["tts_drop_count_remote"] >= 0)
    result["pass_local_tts"] = bool(result["tts_drop_count_local"] >= 0)

    if out_json:
        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 4 multi-route verification.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--wav", default="artifacts/driver/synctranslate_virtual_audio/virtual_mic_recording.wav")
    parser.add_argument("--loops", type=int, default=2)
    parser.add_argument("--observe-sec", type=float, default=12.0)
    parser.add_argument("--profile", default="default")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    result = run_multi_route_test(
        config_path=args.config,
        wav_path=args.wav,
        loops=args.loops,
        observe_sec=args.observe_sec,
        out_json=args.out,
        profile_name=args.profile,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if bool(result.get("pass_d1", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
