from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

from app.infra.audio.virtual_devices import detect_virtual_audio_install


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    payload = np.asarray(audio, dtype=np.float32)
    if payload.ndim == 1:
        payload = payload.reshape((-1, 1))
    pcm16 = (np.clip(payload, -1.0, 1.0) * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as fp:
        fp.setnchannels(int(payload.shape[1]))
        fp.setsampwidth(2)
        fp.setframerate(int(sample_rate))
        fp.writeframes(pcm16.tobytes())


def _sine(sample_rate: int, duration_sec: float, frequency: float, channels: int) -> np.ndarray:
    frames = max(1, int(sample_rate * duration_sec))
    t = np.arange(frames, dtype=np.float32) / float(sample_rate)
    mono = np.sin(2.0 * np.pi * float(frequency) * t) * 0.15
    if channels <= 1:
        return mono.reshape((-1, 1))
    return np.repeat(mono.reshape((-1, 1)), int(channels), axis=1)


def list_status() -> int:
    status = detect_virtual_audio_install()
    print(f"installed={status.installed}")
    print(f"speaker_available={status.speaker_available}")
    print(f"microphone_available={status.microphone_available}")
    print(f"speaker_name={status.speaker_name}")
    print(f"microphone_name={status.microphone_name}")
    print(f"speaker_index={status.speaker_index}")
    print(f"microphone_index={status.microphone_index}")
    print("render_endpoints:")
    for endpoint in status.render_endpoints:
        print(
            f"  [{endpoint.index}] {endpoint.name} "
            f"hostapi={endpoint.hostapi_name} out={endpoint.max_output_channels} sr={endpoint.default_samplerate:g}"
        )
    print("capture_endpoints:")
    for endpoint in status.capture_endpoints:
        print(
            f"  [{endpoint.index}] {endpoint.name} "
            f"hostapi={endpoint.hostapi_name} in={endpoint.max_input_channels} sr={endpoint.default_samplerate:g}"
        )
    return 0 if status.installed else 1


def play_sine(duration_sec: float, sample_rate: int, frequency: float) -> int:
    status = detect_virtual_audio_install()
    if not status.speaker_available:
        raise RuntimeError("SyncTranslate render endpoint was not found")
    device_index = status.speaker_index
    channels = max(1, int(sd.query_devices(device_index).get("max_output_channels", 1) or 1))
    audio = _sine(sample_rate, duration_sec, frequency, min(channels, 2))
    print(f"playing {duration_sec:.2f}s sine to [{device_index}] {status.speaker_name}")
    sd.play(audio, samplerate=sample_rate, device=device_index, blocking=True)
    return 0


def record_microphone(duration_sec: float, sample_rate: int, output_path: Path) -> int:
    status = detect_virtual_audio_install()
    if not status.microphone_available:
        raise RuntimeError("SyncTranslate capture endpoint was not found")
    device_index = status.microphone_index
    channels = max(1, int(sd.query_devices(device_index).get("max_input_channels", 1) or 1))
    print(f"recording {duration_sec:.2f}s from [{device_index}] {status.microphone_name}")
    audio = sd.rec(
        max(1, int(sample_rate * duration_sec)),
        samplerate=sample_rate,
        channels=min(channels, 2),
        dtype="float32",
        device=device_index,
        blocking=True,
    )
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    _write_wav(output_path, audio, sample_rate)
    print(f"wrote={output_path}")
    print(f"rms={rms:.6f}")
    print(f"peak={peak:.6f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="SyncTranslate virtual audio endpoint smoke tool")
    parser.add_argument("--list", action="store_true", help="List detected SyncTranslate endpoints")
    parser.add_argument("--play-sine", action="store_true", help="Play a sine wave to the detected render endpoint")
    parser.add_argument("--record", action="store_true", help="Record from the detected capture endpoint")
    parser.add_argument("--duration-sec", type=float, default=1.0)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--frequency", type=float, default=440.0)
    parser.add_argument("--output", default="artifacts/driver/synctranslate_virtual_audio/virtual_mic_recording.wav")
    args = parser.parse_args()

    if args.play_sine:
        return play_sine(args.duration_sec, args.sample_rate, args.frequency)
    if args.record:
        return record_microphone(args.duration_sec, args.sample_rate, Path(args.output))
    return list_status()


if __name__ == "__main__":
    raise SystemExit(main())
