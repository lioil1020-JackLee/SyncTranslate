from __future__ import annotations

import argparse
import json
import time
import wave
from pathlib import Path

import numpy as np

from app.application.transcript_service import TranscriptBuffer
from app.bootstrap.dependency_container import build_pipeline_bundle
from app.bootstrap.external_runtime import configure_external_ai_runtime
from app.infra.audio.capture import AudioCapture
from app.infra.audio.playback import AudioPlayback
from app.infra.config.schema import AudioRouteConfig
from app.infra.config.settings_store import load_config


ROOT = Path(__file__).resolve().parents[2]


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


def _run_case(*, label: str, wav_path: Path, remote_asr_language: str, meeting_source: str, meeting_target: str, pause_before: float) -> dict[str, object]:
    print(f"\n=== {label} ===", flush=True)
    print(f"準備播放到本地輸出，{pause_before:.1f} 秒後開始。", flush=True)
    time.sleep(max(0.0, float(pause_before)))

    cfg = load_config(str(ROOT / "config.yaml"))
    cfg.runtime.local_asr_language = "none"
    cfg.runtime.remote_asr_language = remote_asr_language
    cfg.runtime.remote_translation_enabled = True
    cfg.runtime.local_translation_enabled = False
    cfg.runtime.translation_enabled = True
    cfg.runtime.remote_translation_target = meeting_target
    cfg.runtime.local_translation_target = "none"
    cfg.runtime.tts_output_mode = "passthrough"
    cfg.runtime.remote_tts_enabled = False
    cfg.runtime.local_tts_enabled = False
    cfg.runtime.remote_tts_voice = "none"
    cfg.runtime.local_tts_voice = "none"
    cfg.language.meeting_source = meeting_source
    cfg.language.meeting_target = meeting_target
    cfg.audio.virtual_audio.require_driver = False

    asr_finals: list[str] = []
    translated_finals: list[str] = []
    tts_requests: list[str] = []
    diagnostics: list[str] = []
    errors: list[str] = []

    def on_error(event) -> None:
        errors.append(str(event))

    def on_diag(message: str) -> None:
        diagnostics.append(str(message))

    def on_asr(event) -> None:
        if getattr(event, "source", "") == "remote" and bool(getattr(event, "is_final", False)):
            asr_finals.append(str(getattr(event, "text", "") or ""))

    def on_translation(event) -> None:
        if getattr(event, "source", "") == "remote" and bool(getattr(event, "is_final", False)):
            translated_finals.append(str(getattr(event, "text", "") or ""))

    configure_external_ai_runtime()
    routes = AudioRouteConfig(
        meeting_in="dummy-remote",
        microphone_in="dummy-local",
        speaker_out=str(cfg.audio.speaker_out or ""),
        meeting_out="dummy-meet",
    )
    bundle = build_pipeline_bundle(
        config=cfg,
        pipeline_revision=2300,
        transcript_buffer=TranscriptBuffer(max_items=None),
        local_capture=AudioCapture(),
        speaker_playback=AudioPlayback(),
        get_local_output_device=lambda: str(cfg.audio.speaker_out or ""),
        on_error=on_error,
        on_diagnostic_event=on_diag,
        on_asr_event=on_asr,
        on_translation_event=on_translation,
    )
    router = bundle.audio_router
    session_controller = bundle.session_controller
    router._on_tts_request = lambda channel, text: tts_requests.append(f"{channel}:{text}")

    start_result = session_controller.start(
        routes,
        sample_rate=cfg.runtime.sample_rate,
        chunk_ms=cfg.runtime.chunk_ms,
        mode="meeting_to_local",
    )
    router.set_output_mode("local", "passthrough")
    router.set_output_mode("remote", "subtitle_only")

    audio, sample_rate = _to_float32_pcm(wav_path)
    audio_duration = float(audio.shape[0]) / max(1, sample_rate)
    client = getattr(router._input_manager._captures.get("remote"), "_bridge", None)
    frames_sent = 0

    try:
        print(f"正在播放：{wav_path.name}", flush=True)
        if client is not None:
            chunk_frames = 960
            for i in range(0, int(audio.shape[0]), chunk_frames):
                segment = audio[i : i + chunk_frames]
                client.debug_inject_remote_input(segment, sample_rate=int(sample_rate))
                frames_sent += int(segment.shape[0])
                time.sleep(segment.shape[0] / max(1, sample_rate))
        time.sleep(max(1.2, audio_duration + 0.8))
        stats = router.stats()
        remote_stats = stats.asr.get("remote", {})
        result = {
            "label": label,
            "start_ok": bool(start_result.ok),
            "stop_ok": False,
            "speaker_out": str(cfg.audio.speaker_out or ""),
            "local_output_mode": router._tts_manager.output_mode("local"),
            "frames_sent": int(frames_sent),
            "remote_partial": int(remote_stats.get("partial_count", 0) or 0),
            "remote_final": int(remote_stats.get("final_count", 0) or 0),
            "asr_finals": asr_finals,
            "translated_finals": translated_finals,
            "tts_request_count": len(tts_requests),
            "diagnostics": diagnostics[-6:],
            "errors": errors[-6:],
        }
    finally:
        stop_result = session_controller.stop()
        result["stop_ok"] = bool(stop_result.ok)

    print("案例播放完成。", flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Play two remote passthrough cases to the configured local output device.")
    parser.add_argument("--report", default=str(ROOT / "logs" / "session_reports" / "remote_passthrough_demo.json"))
    parser.add_argument("--pause-before", type=float, default=2.0)
    args = parser.parse_args()

    cases = [
        {
            "label": "case_1_en_to_zh_tts_off_passthrough",
            "wav_path": ROOT / "downloads" / "asr_regression_en_pool" / "audio" / "en_read_001.wav",
            "remote_asr_language": "en",
            "meeting_source": "en",
            "meeting_target": "zh-TW",
            "pause_before": args.pause_before,
        },
        {
            "label": "case_2_zh_to_en_tts_off_passthrough",
            "wav_path": ROOT / "downloads" / "asr_regression_zh_pool" / "audio" / "zh_read_001.wav",
            "remote_asr_language": "zh-TW",
            "meeting_source": "zh-TW",
            "meeting_target": "en",
            "pause_before": args.pause_before,
        },
    ]
    results = [_run_case(**case) for case in cases]
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())