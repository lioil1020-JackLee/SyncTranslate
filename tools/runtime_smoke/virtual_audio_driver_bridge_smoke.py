from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.infra.audio.virtual_bridge_client import VirtualAudioBridgeClient
from app.infra.audio.virtual_devices import detect_virtual_audio_install


def _sine(sample_rate: int, duration_sec: float, frequency: float, channels: int) -> np.ndarray:
    frames = max(1, int(sample_rate * duration_sec))
    t = np.arange(frames, dtype=np.float32) / float(sample_rate)
    mono = np.sin(2.0 * np.pi * float(frequency) * t) * 0.15
    if channels <= 1:
        return mono.reshape((-1, 1))
    return np.repeat(mono.reshape((-1, 1)), int(channels), axis=1)


def run_smoke(*, bridge_path: str, sample_rate: int, duration_sec: float, frequency: float) -> dict[str, object]:
    status = detect_virtual_audio_install()
    if not status.speaker_available:
        raise RuntimeError("SyncTranslate render endpoint was not found")

    device_info = sd.query_devices(status.speaker_index)
    channels = min(2, max(1, int(device_info.get("max_output_channels", 1) or 1)))
    audio = _sine(sample_rate, duration_sec, frequency, channels)

    client = VirtualAudioBridgeClient(bridge_path=bridge_path)
    try:
        client.start_remote_input(sample_rate=sample_rate, device_name=status.speaker_name)
        before = client.stats()
        sd.play(audio, samplerate=sample_rate, device=status.speaker_index, blocking=True)
        time.sleep(0.2)
        after = client.stats()
    finally:
        client.close()

    delta_frames = int(after.remote_input_frames) - int(before.remote_input_frames)
    return {
        "ok": delta_frames > 0,
        "speaker_index": status.speaker_index,
        "speaker_name": status.speaker_name,
        "sample_rate": sample_rate,
        "duration_sec": duration_sec,
        "remote_input_frames_before": int(before.remote_input_frames),
        "remote_input_frames_after": int(after.remote_input_frames),
        "remote_input_delta_frames": delta_frames,
        "remote_input_buffer_capacity_frames": int(after.remote_input_buffer_capacity_frames),
        "virtual_microphone_shared_memory_name": after.virtual_microphone_shared_memory_name,
        "virtual_microphone_event_name": after.virtual_microphone_event_name,
        "last_error": after.last_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify driver render PCM reaches the SyncTranslate bridge")
    parser.add_argument("--bridge-path", default=r"runtimes\audio\sync_audio_bridge.exe")
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--duration-sec", type=float, default=1.0)
    parser.add_argument("--frequency", type=float, default=440.0)
    parser.add_argument("--json-output", default="")
    args = parser.parse_args()

    result = run_smoke(
        bridge_path=args.bridge_path,
        sample_rate=int(args.sample_rate),
        duration_sec=float(args.duration_sec),
        frequency=float(args.frequency),
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    return 0 if bool(result.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
