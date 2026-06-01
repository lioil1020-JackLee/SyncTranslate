from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.bootstrap.external_runtime import configure_external_ai_runtime

configure_external_ai_runtime()

from app.infra.asr.backend_v2 import _build_engine
from app.infra.asr.frontend_v2 import AsrAudioFrontendV2
from app.infra.asr.manager_v2 import ASRManagerV2
from app.infra.asr.profile_selection import asr_profile_for_language
from app.infra.audio.capture import AudioCapture
from app.infra.audio.frame import ChannelPolicy
from app.infra.audio.loopback_capture import WasapiLoopbackCaptureSource
from app.infra.audio.playback import AudioPlayback
from app.infra.config.settings_store import load_config
from app.infra.tts.edge_tts_adapter import EdgeTtsProvider

from tools.validation.common import VALIDATION_OUTPUT_DIR


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare meeting-mode streaming sources. "
            "system-input records a Windows input device; output-loopback records a Windows output device via WASAPI loopback."
        )
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--sources", choices=["input", "loopback", "both"], default="both")
    parser.add_argument("--duration", type=float, default=6.0, help="Seconds to record per source")
    parser.add_argument("--input-device", default="", help="Input device override")
    parser.add_argument("--output-device", default="", help="Output loopback device override")
    parser.add_argument("--language", default="", help="ASR language override, e.g. zh-TW/en/ja/ko/th")
    parser.add_argument("--asr", action="store_true", help="Run faster-whisper final ASR on captured WAVs")
    parser.add_argument(
        "--raw-asr",
        action="store_true",
        help="Skip the app streaming ASR frontend before final ASR. By default --asr mirrors app frontend gain/filtering.",
    )
    parser.add_argument("--play-test-speech", action="store_true", help="Play a short Edge TTS phrase to the output device during capture")
    parser.add_argument(
        "--test-text",
        default="這是一段 SyncTranslate 會議字幕測試語音，用來比較系統輸入與系統輸出 loopback 的辨識品質。",
        help="Text used with --play-test-speech",
    )
    parser.add_argument("--test-voice", default="zh-TW-HsiaoChenNeural", help="Edge TTS voice for --play-test-speech")
    parser.add_argument("--json", dest="json_path", default="", help="Optional JSON output path")
    parser.add_argument("--output-dir", default=str(VALIDATION_OUTPUT_DIR), help="Directory for WAV/JSON artifacts")
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Record sources one after another. By default --sources both records input and loopback simultaneously.",
    )
    return parser


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = np.asarray(audio, dtype=np.float32)
    if payload.ndim == 1:
        payload = payload.reshape((-1, 1))
    pcm16 = (np.clip(payload, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as fp:
        fp.setnchannels(int(pcm16.shape[1]))
        fp.setsampwidth(2)
        fp.setframerate(int(sample_rate))
        fp.writeframes(pcm16.tobytes())


def _metrics(audio: np.ndarray, sample_rate: float, callback_deltas: list[float]) -> dict[str, object]:
    payload = np.asarray(audio, dtype=np.float32)
    if payload.ndim == 1:
        payload = payload.reshape((-1, 1))
    abs_audio = np.abs(payload)
    rms = float(np.sqrt(np.mean(np.square(payload)))) if payload.size else 0.0
    peak = float(np.max(abs_audio)) if payload.size else 0.0
    clipped_ratio = float(np.mean(abs_audio >= 0.999)) if payload.size else 0.0
    near_silence_ratio = float(np.mean(abs_audio < 1e-4)) if payload.size else 1.0
    recommended_gain = min(8.0, 0.05 / max(rms, 1e-6)) if payload.size else 1.0
    if rms <= 0.001:
        level_status = "FAIL"
        level_message = "captured audio is effectively silent"
    elif rms < 0.02:
        level_status = "WARN"
        level_message = "captured audio level is low; ASR may be less accurate"
    else:
        level_status = "PASS"
        level_message = "captured audio level is suitable for ASR"
    duration_sec = float(payload.shape[0]) / float(sample_rate or 1.0) if payload.ndim == 2 else 0.0
    deltas_ms = [value * 1000.0 for value in callback_deltas if value > 0]
    return {
        "sample_rate": int(round(float(sample_rate or 0.0))),
        "channels": int(payload.shape[1]) if payload.ndim == 2 and payload.size else 0,
        "frames": int(payload.shape[0]) if payload.ndim == 2 else int(payload.size),
        "duration_sec": round(duration_sec, 3),
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "clipped_ratio": round(clipped_ratio, 6),
        "near_silence_ratio": round(near_silence_ratio, 6),
        "level_status": level_status,
        "level_message": level_message,
        "recommended_frontend_gain_to_0_05_rms": round(float(recommended_gain), 3),
        "callback_count": len(callback_deltas) + (1 if payload.size else 0),
        "callback_delta_avg_ms": round(float(np.mean(deltas_ms)), 2) if deltas_ms else 0.0,
        "callback_delta_max_ms": round(float(np.max(deltas_ms)), 2) if deltas_ms else 0.0,
    }


def _capture(
    source,
    *,
    device: str,
    sample_rate: int,
    duration: float,
    channels_policy: str,
    playback_audio: np.ndarray | None = None,
    playback_device: str = "",
) -> tuple[np.ndarray, float, list[float], str]:
    chunks: list[np.ndarray] = []
    callback_deltas: list[float] = []
    latest_rate = float(sample_rate)
    last_callback = 0.0

    def on_audio(audio: np.ndarray, rate: float) -> None:
        nonlocal latest_rate, last_callback
        now = time.perf_counter()
        if last_callback:
            callback_deltas.append(now - last_callback)
        last_callback = now
        latest_rate = float(rate)
        chunks.append(np.asarray(audio, dtype=np.float32).copy())

    source.add_consumer(on_audio)
    source.start(device, sample_rate=sample_rate, chunk_ms=40, channels_policy=channels_policy)
    try:
        if playback_audio is not None and playback_audio.size and playback_device:
            import threading

            def _play() -> None:
                time.sleep(0.35)
                AudioPlayback().play(playback_audio, sample_rate=24000, output_device_name=playback_device, blocking=True)

            threading.Thread(target=_play, daemon=True).start()
        time.sleep(max(0.1, float(duration)))
        stats = source.stats()
        error = str(getattr(stats, "last_error", "") or "")
    finally:
        source.stop()
    if chunks:
        audio = np.concatenate(chunks, axis=0)
    else:
        audio = np.zeros((0, 1), dtype=np.float32)
    return audio, latest_rate, callback_deltas, error


def _capture_simultaneous(
    plans: list[tuple[str, object, str, str]],
    *,
    sample_rate: int,
    duration: float,
    playback_audio: np.ndarray | None = None,
    playback_device: str = "",
) -> dict[str, tuple[np.ndarray, float, list[float], str]]:
    buffers: dict[str, list[np.ndarray]] = {name: [] for name, *_ in plans}
    callback_deltas: dict[str, list[float]] = {name: [] for name, *_ in plans}
    latest_rates: dict[str, float] = {name: float(sample_rate) for name, *_ in plans}
    last_callbacks: dict[str, float] = {name: 0.0 for name, *_ in plans}
    errors: dict[str, str] = {name: "" for name, *_ in plans}
    started: list[tuple[str, object]] = []

    def make_consumer(source_name: str):
        def on_audio(audio: np.ndarray, rate: float) -> None:
            now = time.perf_counter()
            if last_callbacks[source_name]:
                callback_deltas[source_name].append(now - last_callbacks[source_name])
            last_callbacks[source_name] = now
            latest_rates[source_name] = float(rate)
            buffers[source_name].append(np.asarray(audio, dtype=np.float32).copy())

        return on_audio

    try:
        for source_name, source, device, policy in plans:
            source.add_consumer(make_consumer(source_name))
            source.start(device, sample_rate=sample_rate, chunk_ms=40, channels_policy=policy)
            started.append((source_name, source))
        if playback_audio is not None and playback_audio.size and playback_device:
            import threading

            def _play() -> None:
                time.sleep(0.35)
                AudioPlayback().play(playback_audio, sample_rate=24000, output_device_name=playback_device, blocking=True)

            threading.Thread(target=_play, daemon=True).start()
        time.sleep(max(0.1, float(duration)))
        for source_name, source in started:
            stats = source.stats()
            errors[source_name] = str(getattr(stats, "last_error", "") or "")
    finally:
        for _source_name, source in reversed(started):
            source.stop()

    result: dict[str, tuple[np.ndarray, float, list[float], str]] = {}
    for source_name, *_ in plans:
        chunks = buffers[source_name]
        audio = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 1), dtype=np.float32)
        result[source_name] = (audio, latest_rates[source_name], callback_deltas[source_name], errors[source_name])
    return result


def _apply_streaming_frontend(config, audio: np.ndarray, sample_rate: int, language: str) -> tuple[np.ndarray, dict[str, object]]:
    manager = ASRManagerV2(config)
    enhancement_enabled, noise_reduce_strength, music_suppress_strength = manager._frontend_enhancement_settings(
        language,
        source="remote",
    )
    frontend = AsrAudioFrontendV2(
        enabled=bool(getattr(config.runtime, "asr_frontend_enabled", True)),
        target_rms=float(getattr(config.runtime, "asr_frontend_target_rms", 0.05)),
        max_gain=manager._frontend_max_gain(),
        highpass_alpha=float(getattr(config.runtime, "asr_frontend_highpass_alpha", 0.96)),
        enhancement_enabled=enhancement_enabled,
        enhancement_noise_reduce_strength=noise_reduce_strength,
        enhancement_noise_adapt_rate=float(getattr(config.runtime, "asr_enhancement_noise_adapt_rate", 0.18)),
        enhancement_music_suppress_strength=music_suppress_strength,
    )
    payload = np.asarray(audio, dtype=np.float32)
    if payload.ndim == 1:
        payload = payload.reshape((-1, 1))
    chunk_frames = max(1, int(float(sample_rate) * int(getattr(config.runtime, "chunk_ms", 40)) / 1000.0))
    chunks: list[np.ndarray] = []
    for start in range(0, payload.shape[0], chunk_frames):
        processed = frontend.process(payload[start : start + chunk_frames], sample_rate)
        chunks.append(processed.audio.astype(np.float32, copy=False))
    processed_audio = np.concatenate(chunks, axis=0) if chunks else np.zeros((0,), dtype=np.float32)
    return processed_audio.astype(np.float32, copy=False), frontend.stats()


def _run_asr(config, audio: np.ndarray, sample_rate: int, language: str, *, raw_asr: bool) -> dict[str, object]:
    frontend_stats: dict[str, object] = {}
    asr_audio = np.asarray(audio, dtype=np.float32)
    if not raw_asr:
        asr_audio, frontend_stats = _apply_streaming_frontend(config, asr_audio, sample_rate, language)
    profile = asr_profile_for_language(config, language)
    engine = _build_engine(profile, language=language)
    start = time.perf_counter()
    result = engine.transcribe_final_result(asr_audio, sample_rate)
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return {
        "language": language,
        "frontend_applied": not raw_asr,
        "frontend": frontend_stats,
        "detected_language": result.detected_language,
        "text": result.text,
        "transcribe_ms": elapsed_ms,
        "avg_logprob": result.avg_logprob,
        "max_no_speech_prob": result.max_no_speech_prob,
        "max_compression_ratio": result.max_compression_ratio,
    }


def _source_plan(config, args) -> list[tuple[str, object, str, str]]:
    plan: list[tuple[str, object, str, str]] = []
    if args.sources in {"input", "both"}:
        device = args.input_device or config.meeting.input_device or config.audio.microphone_in or config.audio.meeting_in
        plan.append(("system_input", AudioCapture(), device, ChannelPolicy.MONO.value))
    if args.sources in {"loopback", "both"}:
        device = args.output_device or config.meeting.output_loopback_device or config.audio.speaker_out
        plan.append(("output_loopback", WasapiLoopbackCaptureSource(), device, ChannelPolicy.STEREO_OR_MONO.value))
    return plan


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    try:
        config = load_config(args.config)
    except Exception as exc:
        print(f"FAIL cannot load config: {exc}")
        return 1

    language = args.language or config.meeting.asr_language or config.runtime.remote_asr_language or "zh-TW"
    sample_rate = int(config.runtime.sample_rate or 48000)
    playback_audio = None
    playback_device = args.output_device or config.meeting.output_loopback_device or config.audio.speaker_out
    if args.play_test_speech:
        print(f"[stream] synthesizing test speech voice={args.test_voice!r}", flush=True)
        playback_audio = EdgeTtsProvider(voice=args.test_voice).synthesize(args.test_text, sample_rate=24000)
    records: list[dict[str, object]] = []
    exit_code = 0
    plan = _source_plan(config, args)
    capture_results: dict[str, tuple[np.ndarray, float, list[float], str]] = {}
    simultaneous = args.sources == "both" and not args.sequential
    if simultaneous:
        valid_plan = [(name, source, device, policy) for name, source, device, policy in plan if device]
        if valid_plan:
            print(f"[stream] recording {len(valid_plan)} sources simultaneously for {args.duration}s")
            try:
                capture_results = _capture_simultaneous(
                    valid_plan,
                    sample_rate=sample_rate,
                    duration=args.duration,
                    playback_audio=playback_audio,
                    playback_device=playback_device,
                )
            except Exception as exc:
                exit_code = 1
                message = str(exc)
                for source_name, _source, device, _policy in valid_plan:
                    records.append({"source": source_name, "status": "FAIL", "device": device, "message": message})
                print(f"FAIL simultaneous capture: {message}")

    for source_name, source, device, policy in plan:
        if not device:
            print(f"WARN {source_name}: no device configured")
            records.append({"source": source_name, "status": "WARN", "message": "no device configured"})
            continue
        print(f"[stream] source={source_name} device={device!r} duration={args.duration}s")
        try:
            if source_name in capture_results:
                audio, rate, callback_deltas, error = capture_results[source_name]
            else:
                audio, rate, callback_deltas, error = _capture(
                    source,
                    device=device,
                    sample_rate=sample_rate,
                    duration=args.duration,
                    channels_policy=policy,
                    playback_audio=playback_audio,
                    playback_device=playback_device,
                )
            wav_path = output_dir / f"meeting_{source_name}.wav"
            _write_wav(wav_path, audio, int(round(rate or sample_rate)))
            record = {
                "source": source_name,
                "status": "PASS",
                "device": device,
                "wav": str(wav_path),
                "capture_error": error,
                **_metrics(audio, rate, callback_deltas),
            }
            if args.asr:
                record["asr"] = _run_asr(
                    config,
                    audio,
                    int(round(rate or sample_rate)),
                    language,
                    raw_asr=args.raw_asr,
                )
            records.append(record)
            print(
                "PASS "
                f"{source_name} rate={record['sample_rate']} ch={record['channels']} "
                f"rms={record['rms']} peak={record['peak']} silence={record['near_silence_ratio']} "
                f"level={record['level_status']} gain_hint={record['recommended_frontend_gain_to_0_05_rms']} "
                f"cb_max_ms={record['callback_delta_max_ms']} wav={wav_path}"
            )
            if args.asr:
                asr = record["asr"]
                assert isinstance(asr, dict)
                print(f"ASR {source_name}: {asr.get('text', '')}")
        except Exception as exc:
            exit_code = 1 if source_name == "system_input" else exit_code
            message = str(exc)
            records.append({"source": source_name, "status": "FAIL", "device": device, "message": message})
            print(f"FAIL {source_name}: {message}")

    summary = {"status": "FAIL" if any(r.get("status") == "FAIL" for r in records) else "PASS", "records": records}
    json_path = Path(args.json_path) if args.json_path else output_dir / "meeting_source_stream_test.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"JSON written: {json_path}")
    return exit_code


if __name__ == "__main__":
    code = main()
    if "--asr" in sys.argv:
        sys.stdout.flush()
        sys.stderr.flush()
        import os

        os._exit(code)
    raise SystemExit(code)
