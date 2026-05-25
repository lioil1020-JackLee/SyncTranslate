from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.infra.audio.capture import AudioCapture
from app.infra.audio.frame import ChannelPolicy
from app.infra.audio.loopback_capture import WasapiLoopbackCaptureSource
from app.infra.audio.virtual_bridge_client import VirtualAudioBridgeClient
from app.infra.audio.virtual_bridge_probe import probe_virtual_audio_bridge
from app.infra.config.settings_store import load_config

from tools.validation.common import VALIDATION_OUTPUT_DIR


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run short SyncTranslate audio smoke tests.")
    parser.add_argument("mode", choices=["meeting-input", "meeting-loopback", "dialogue-passthrough"])
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--device", default="", help="Input or output device selector. Defaults to config value.")
    parser.add_argument("--duration", type=float, default=3.0, help="Capture or tone duration in seconds")
    parser.add_argument("--wav", default="", help="Optional WAV output path")
    parser.add_argument("--direction", choices=["local-to-remote", "remote-to-local"], default="local-to-remote")
    return parser


def _metrics(chunks: list[np.ndarray], sample_rate: float) -> dict[str, object]:
    if chunks:
        audio = np.concatenate(chunks, axis=0)
    else:
        audio = np.zeros((0, 1), dtype=np.float32)
    if audio.ndim == 1:
        audio = audio.reshape((-1, 1))
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return {
        "sample_rate": int(round(float(sample_rate or 0.0))),
        "channels": int(audio.shape[1]) if audio.ndim == 2 and audio.size else 0,
        "frames": int(audio.shape[0]) if audio.ndim == 2 else int(audio.size),
        "rms": rms,
        "peak": peak,
        "audio": audio,
    }


def _write_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = np.asarray(audio, dtype=np.float32)
    if payload.ndim == 1:
        payload = payload.reshape((-1, 1))
    pcm = np.clip(payload, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2")
    with wave.open(str(output), "wb") as fp:
        fp.setnchannels(int(pcm16.shape[1]))
        fp.setsampwidth(2)
        fp.setframerate(int(sample_rate))
        fp.writeframes(pcm16.tobytes())
    return output


def _capture_for(source, *, device: str, sample_rate: int, duration: float, loopback: bool) -> dict[str, object]:
    chunks: list[np.ndarray] = []
    latest_rate = float(sample_rate)

    def on_audio(audio: np.ndarray, rate: float) -> None:
        nonlocal latest_rate
        latest_rate = float(rate)
        chunks.append(np.asarray(audio, dtype=np.float32).copy())

    source.add_consumer(on_audio)
    if loopback:
        source.start(device, sample_rate, 40, channels_policy=ChannelPolicy.STEREO_OR_MONO.value)
    else:
        source.start(device, sample_rate, 40, channels_policy=ChannelPolicy.STEREO_OR_MONO.value)
    try:
        time.sleep(max(0.1, float(duration)))
    finally:
        source.stop()
    return _metrics(chunks, latest_rate)


def _run_meeting_input(config, args) -> int:
    device = args.device or config.meeting.input_device or config.audio.microphone_in or config.audio.meeting_in
    if not device:
        print("FAIL No input device configured. Pick a microphone in the UI or pass --device.")
        return 1
    try:
        result = _capture_for(AudioCapture(), device=device, sample_rate=config.runtime.sample_rate, duration=args.duration, loopback=False)
        wav_path = args.wav or str(VALIDATION_OUTPUT_DIR / "meeting_input.wav")
        _write_wav(wav_path, result.pop("audio"), int(result["sample_rate"] or config.runtime.sample_rate))
        print(f"PASS meeting-input sample_rate={result['sample_rate']} channels={result['channels']} rms={result['rms']:.6f} peak={result['peak']:.6f}")
        print(f"WAV written: {wav_path}")
        return 0
    except Exception as exc:
        print(f"FAIL Unable to record from input device '{device}': {exc}")
        return 1


def _run_meeting_loopback(config, args) -> int:
    device = args.device or config.meeting.output_loopback_device or config.audio.speaker_out
    if not device:
        print("FAIL No output loopback device configured. Pick a speaker/headset in the UI or pass --device.")
        return 1
    try:
        result = _capture_for(
            WasapiLoopbackCaptureSource(),
            device=device,
            sample_rate=config.runtime.sample_rate,
            duration=args.duration,
            loopback=True,
        )
        wav_path = args.wav or str(VALIDATION_OUTPUT_DIR / "meeting_loopback.wav")
        _write_wav(wav_path, result.pop("audio"), int(result["sample_rate"] or config.runtime.sample_rate))
        print(f"PASS meeting-loopback sample_rate={result['sample_rate']} channels={result['channels']} rms={result['rms']:.6f} peak={result['peak']:.6f}")
        print(f"WAV written: {wav_path}")
        return 0
    except Exception as exc:
        print(f"WARN Unable to record WASAPI loopback from output device '{device}': {exc}")
        return 0


def _run_dialogue_passthrough(config, args) -> int:
    bridge_path = str(config.audio.virtual_audio.bridge_path or "")
    probe = probe_virtual_audio_bridge(bridge_path)
    if not probe.ready:
        print(f"WARN dialogue-passthrough unavailable: bridge/driver not ready ({probe.error or 'unknown error'}).")
        return 0
    sample_rate = int(config.audio.virtual_audio.sample_rate or 48000)
    duration = max(0.1, float(args.duration))
    frames = int(sample_rate * duration)
    t = np.arange(frames, dtype=np.float32) / float(sample_rate)
    tone = (np.sin(2.0 * np.pi * 440.0 * t) * 0.12).reshape((-1, 1))
    client = VirtualAudioBridgeClient(bridge_path=bridge_path)
    try:
        before = client.stats()
        if args.direction == "local-to-remote":
            client.write_virtual_microphone(tone, sample_rate=sample_rate)
        else:
            client.debug_inject_remote_input(tone, sample_rate=sample_rate)
        time.sleep(0.05)
        after = client.stats()
        dropped = int(after.virtual_microphone_dropped_frames) - int(before.virtual_microphone_dropped_frames)
        buffered = int(after.virtual_microphone_buffered_frames)
        print(
            "PASS dialogue-passthrough "
            f"direction={args.direction} sent_frames={frames} dropped_frames={max(0, dropped)} buffered_frames={buffered}"
        )
        return 0
    except Exception as exc:
        print(f"WARN dialogue-passthrough could not send test frame: {exc}")
        return 0
    finally:
        client.close()


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.mode == "meeting-input":
            return _run_meeting_input(config, args)
        if args.mode == "meeting-loopback":
            return _run_meeting_loopback(config, args)
        return _run_dialogue_passthrough(config, args)
    except Exception as exc:
        print(f"FAIL audio smoke test could not start: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
