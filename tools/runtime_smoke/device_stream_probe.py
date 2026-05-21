from __future__ import annotations

import argparse
import json
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

from app.infra.audio.device_registry import normalize_device_text
from app.infra.audio.virtual_bridge_client import VirtualAudioBridgeClient
from app.infra.config.settings_store import load_config
from app.infra.audio.capture import AudioCapture


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        channels = int(wf.getnchannels())
        sample_rate = int(wf.getframerate())
        sample_width = int(wf.getsampwidth())
        frame_count = int(wf.getnframes())
        raw = wf.readframes(frame_count)
    if sample_width == 2:
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")
    if channels > 1:
        audio = audio.reshape(-1, channels)
    else:
        audio = audio.reshape(-1, 1)
    return audio, sample_rate


def _resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate <= 0 or dst_rate <= 0 or src_rate == dst_rate:
        return audio.astype(np.float32, copy=False)
    src = np.asarray(audio, dtype=np.float32)
    if src.ndim == 1:
        src = src.reshape(-1, 1)
    src_len = int(src.shape[0])
    if src_len <= 1:
        return src
    dst_len = max(1, int(round(src_len * float(dst_rate) / float(src_rate))))
    src_x = np.linspace(0.0, 1.0, src_len, endpoint=False)
    dst_x = np.linspace(0.0, 1.0, dst_len, endpoint=False)
    channels = int(src.shape[1])
    out = np.empty((dst_len, channels), dtype=np.float32)
    for ch in range(channels):
        out[:, ch] = np.interp(dst_x, src_x, src[:, ch]).astype(np.float32, copy=False)
    return out


def _find_output_index(name: str) -> int:
    target = normalize_device_text(name)
    devices = sd.query_devices()
    candidates: list[tuple[int, int]] = []
    for idx, dev in enumerate(devices):
        if int(dev.get("max_output_channels", 0) or 0) <= 0:
            continue
        dev_name = str(dev.get("name") or "").strip()
        if not dev_name:
            continue
        norm = normalize_device_text(dev_name)
        if norm == target:
            return idx
        if target and target in norm:
            candidates.append((0, idx))
        elif "synctranslate" in norm and "synctranslate" in target:
            candidates.append((1, idx))
    if candidates:
        candidates.sort()
        return candidates[0][1]
    raise ValueError(f"No output device matched: {name}")


def run_probe(config_path: str, wav_path: str, *, local_observe_sec: float = 3.0) -> dict[str, object]:
    cfg = load_config(config_path)
    wav_file = Path(wav_path)
    audio, wav_sr = _load_wav(wav_file)
    target_sr = int(cfg.runtime.sample_rate)
    if wav_sr != target_sr:
        audio = _resample_linear(audio, wav_sr, target_sr)
        wav_sr = target_sr

    result: dict[str, object] = {
        "config": config_path,
        "wav": str(wav_file),
        "meeting_in": str(cfg.audio.meeting_in),
        "microphone_in": str(cfg.audio.microphone_in),
        "runtime_sample_rate": int(cfg.runtime.sample_rate),
        "runtime_chunk_ms": int(cfg.runtime.chunk_ms),
    }

    # Remote path: actual playback to configured meeting_in output device,
    # then verify bridge remote_input frame growth.
    remote: dict[str, object] = {}
    bridge = VirtualAudioBridgeClient(
        bridge_path=str(getattr(cfg.audio.virtual_audio, "bridge_path", "") or "")
    )
    try:
        out_idx = _find_output_index(str(cfg.audio.meeting_in or ""))
        remote["playback_output_index"] = int(out_idx)
        remote["playback_output_name"] = str(sd.query_devices(out_idx).get("name") or "")
        bridge.start_remote_input(
            sample_rate=int(cfg.runtime.sample_rate),
            device_name=str(cfg.audio.meeting_in or ""),
            chunk_ms=max(10, int(cfg.runtime.chunk_ms)),
        )
        before = bridge.stats()
        sd.play(audio, samplerate=wav_sr, device=out_idx, blocking=True)
        time.sleep(0.25)
        after = bridge.stats()
        remote["ok"] = bool(int(after.remote_input_frames) > int(before.remote_input_frames))
        remote["frames_before"] = int(before.remote_input_frames)
        remote["frames_after"] = int(after.remote_input_frames)
        remote["delta_frames"] = int(after.remote_input_frames) - int(before.remote_input_frames)
        remote["bridge_last_error"] = str(after.last_error or "")
    except Exception as exc:
        remote["ok"] = False
        remote["error"] = str(exc)
    finally:
        try:
            bridge.close()
        except Exception:
            pass
    result["remote_path"] = remote

    # Local path: start actual local capture device and verify frame count moves.
    local: dict[str, object] = {}
    cap = AudioCapture()
    try:
        cap.start(
            str(cfg.audio.microphone_in or ""),
            sample_rate=int(cfg.runtime.sample_rate),
            chunk_ms=int(cfg.runtime.chunk_ms),
        )
        s0 = cap.stats()
        time.sleep(max(0.5, float(local_observe_sec)))
        s1 = cap.stats()
        local["ok"] = bool(int(s1.frame_count) > int(s0.frame_count))
        local["frames_before"] = int(s0.frame_count)
        local["frames_after"] = int(s1.frame_count)
        local["delta_frames"] = int(s1.frame_count) - int(s0.frame_count)
        local["level"] = float(s1.level)
        local["last_error"] = str(s1.last_error or "")
    except Exception as exc:
        local["ok"] = False
        local["error"] = str(exc)
    finally:
        try:
            cap.stop()
        except Exception:
            pass
    result["local_path"] = local

    result["ok"] = bool(remote.get("ok")) and bool(local.get("ok"))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe local/remote streaming paths using configured real devices")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--wav", default="downloads/asr_regression_en_pool/audio/en_read_001.wav")
    parser.add_argument("--local-observe-sec", type=float, default=3.0)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    result = run_probe(
        config_path=str(args.config),
        wav_path=str(args.wav),
        local_observe_sec=float(args.local_observe_sec),
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    return 0 if bool(result.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
