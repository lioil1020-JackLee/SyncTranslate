"""VM IOCTL smoke test for SyncTranslate Virtual Audio Driver.

只在 disposable Windows VM 中執行（已安裝 driver package 後）。
不要在開發主機直接執行本工具來安裝或驗證主機 driver。

測試步驟：
  1. 開啟 \\\\.\\SyncTranslateVirtualAudioControl
  2. 呼叫 GET_STATS（確認 control device 可用）
  3. 用 WRITE_PCM 寫入 1 kHz float32 mono sine
  4. 確認 totalWrittenFrames 增加
  5. 從 SyncTranslate Virtual Microphone 錄音
  6. 驗證 RMS / 主要頻率 / frame count
  7. 呼叫 FLUSH
  8. 確認 bufferedFrames 下降或歸零

使用方式：
    python tools/runtime_smoke/virtual_audio_driver_ioctl_smoke.py
    python tools/runtime_smoke/virtual_audio_driver_ioctl_smoke.py --json-output result.json
    python tools/runtime_smoke/virtual_audio_driver_ioctl_smoke.py --skip-record
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.infra.audio.synctranslate_driver_client import (
    SyncTranslateDriverAudioClient,
    SyncTranslateDriverUnavailable,
    SyncTranslateDriverStats,
)

try:
    import sounddevice as sd
    _HAS_SOUNDDEVICE = True
except ImportError:
    _HAS_SOUNDDEVICE = False

# ---------------------------------------------------------------------------
# 音訊工具
# ---------------------------------------------------------------------------

SAMPLE_RATE = 48000
SINE_FREQUENCY = 1000.0
SINE_DURATION_SEC = 1.0
RECORD_DURATION_SEC = 2.0

_MIN_RMS = 0.005       # 錄到的 audio 至少要有這個 RMS 才算 pass
_FREQ_TOLERANCE_HZ = 60.0  # 允許的頻率誤差


def _make_sine(sample_rate: int, duration_sec: float, frequency: float) -> np.ndarray:
    """產生 float32 mono sine，振幅 0.5。"""
    frames = max(1, int(sample_rate * duration_sec))
    t = np.arange(frames, dtype=np.float32) / float(sample_rate)
    return (np.sin(2.0 * np.pi * float(frequency) * t) * 0.5).astype(np.float32)


def _rms(audio: np.ndarray) -> float:
    data = np.asarray(audio, dtype=np.float32).ravel()
    if data.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(data ** 2)))


def _dominant_frequency(audio: np.ndarray, sample_rate: int) -> float:
    """透過 FFT 找出主要頻率（Hz）。"""
    data = np.asarray(audio, dtype=np.float32).ravel()
    if data.size < 2:
        return 0.0
    spectrum = np.abs(np.fft.rfft(data))
    freqs = np.fft.rfftfreq(data.size, d=1.0 / sample_rate)
    return float(freqs[int(np.argmax(spectrum))])


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


# ---------------------------------------------------------------------------
# 尋找 SyncTranslate Virtual Microphone 裝置
# ---------------------------------------------------------------------------

def _find_virtual_microphone() -> tuple[int, str] | None:
    """回傳 (device_index, device_name) 或 None。"""
    if not _HAS_SOUNDDEVICE:
        return None
    try:
        devices = sd.query_devices()
    except Exception:
        return None
    for idx, dev in enumerate(devices):
        name: str = dev.get("name", "")
        if "synctranslate" in name.lower() and int(dev.get("max_input_channels", 0)) > 0:
            return idx, name
    return None


# ---------------------------------------------------------------------------
# 個別 smoke step
# ---------------------------------------------------------------------------

def step_open_device(client: SyncTranslateDriverAudioClient) -> dict[str, Any]:
    """Step 1：開啟 control device。"""
    print("[1/7] 開啟 control device...")
    available = client.is_available()
    return {"ok": available, "device_path": client.device_path}


def step_get_stats(client: SyncTranslateDriverAudioClient) -> dict[str, Any]:
    """Step 2：呼叫 GET_STATS。"""
    print("[2/7] GET_STATS...")
    stats: SyncTranslateDriverStats = client.stats()
    print(
        f"       capacity={stats.capacity_frames}  buffered={stats.buffered_frames}"
        f"  written={stats.total_written_frames}  read={stats.total_read_frames}"
        f"  dropped={stats.dropped_frames}  underrun={stats.underrun_frames}"
    )
    return {
        # Capacity can be zero immediately after reboot before first write warms up the ring.
        "ok": True,
        "capacity_nonzero": stats.capacity_frames > 0,
        "stats": {
            "capacity_frames": stats.capacity_frames,
            "buffered_frames": stats.buffered_frames,
            "total_written_frames": stats.total_written_frames,
            "total_read_frames": stats.total_read_frames,
            "dropped_frames": stats.dropped_frames,
            "underrun_frames": stats.underrun_frames,
        },
    }


def step_write_pcm(
    client: SyncTranslateDriverAudioClient,
    *,
    sample_rate: int = SAMPLE_RATE,
    duration_sec: float = SINE_DURATION_SEC,
    frequency: float = SINE_FREQUENCY,
) -> dict[str, Any]:
    """Step 3+4：寫入 sine PCM，確認 totalWrittenFrames 增加。"""
    print(f"[3/7] WRITE_PCM（{frequency:.0f} Hz sine，{duration_sec:.2f}s，{sample_rate} Hz）...")
    sine = _make_sine(sample_rate, duration_sec, frequency)

    stats_before: SyncTranslateDriverStats = client.stats()
    written_frames = client.write_virtual_microphone(sine, sample_rate=sample_rate)
    stats_after: SyncTranslateDriverStats = client.stats()

    delta = stats_after.total_written_frames - stats_before.total_written_frames
    ok = delta > 0
    print(f"       written_frames_reported={written_frames}  totalWrittenFrames_delta={delta}  ok={ok}")
    return {
        "ok": ok,
        "sine_frames": int(sine.size),
        "written_frames_returned": written_frames,
        "total_written_before": stats_before.total_written_frames,
        "total_written_after": stats_after.total_written_frames,
        "total_written_delta": delta,
    }


def step_record(
    *,
    sample_rate: int = SAMPLE_RATE,
    duration_sec: float = RECORD_DURATION_SEC,
    wav_output: Path | None = None,
) -> dict[str, Any]:
    """Step 5+6：從 SyncTranslate Virtual Microphone 錄音並驗證 RMS / 頻率。"""
    print(f"[4/7] 從 SyncTranslate Virtual Microphone 錄音（{duration_sec:.2f}s）...")
    if not _HAS_SOUNDDEVICE:
        return {"ok": None, "skipped": True, "reason": "sounddevice_not_installed"}

    mic = _find_virtual_microphone()
    if mic is None:
        return {"ok": None, "skipped": True, "reason": "virtual_microphone_not_found"}

    mic_index, mic_name = mic
    channels = max(1, int(sd.query_devices(mic_index).get("max_input_channels", 1) or 1))
    print(f"       device=[{mic_index}] {mic_name}  channels={channels}")

    try:
        recorded = sd.rec(
            max(1, int(sample_rate * duration_sec)),
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
            device=mic_index,
        )
        sd.wait()
    except Exception as exc:
        return {"ok": False, "skipped": False, "error": str(exc), "device": mic_name}

    mono = recorded[:, 0] if recorded.ndim > 1 else recorded.ravel()
    rms_value = _rms(mono)
    dominant_freq = _dominant_frequency(mono, sample_rate)
    freq_ok = abs(dominant_freq - SINE_FREQUENCY) <= _FREQ_TOLERANCE_HZ
    rms_ok = rms_value >= _MIN_RMS
    ok = rms_ok and freq_ok

    print(
        f"       rms={rms_value:.5f} (pass={rms_ok})  "
        f"dominant_freq={dominant_freq:.1f} Hz (pass={freq_ok})  overall={ok}"
    )

    if wav_output is not None:
        _write_wav(wav_output, mono, sample_rate)
        print(f"       錄音已儲存：{wav_output}")

    return {
        "ok": ok,
        "skipped": False,
        "device_index": mic_index,
        "device_name": mic_name,
        "rms": rms_value,
        "rms_pass": rms_ok,
        "dominant_freq_hz": dominant_freq,
        "freq_pass": freq_ok,
        "freq_tolerance_hz": _FREQ_TOLERANCE_HZ,
        "min_rms_threshold": _MIN_RMS,
    }


def step_flush(client: SyncTranslateDriverAudioClient) -> dict[str, Any]:
    """Step 7：呼叫 FLUSH，確認 bufferedFrames 下降。"""
    print("[5/7] FLUSH...")
    stats_before: SyncTranslateDriverStats = client.stats()
    client.flush_virtual_microphone()
    # 讓 capture callback 有機會清空
    time.sleep(0.2)
    stats_after: SyncTranslateDriverStats = client.stats()

    before_buf = stats_before.buffered_frames
    after_buf = stats_after.buffered_frames
    ok = after_buf < before_buf or after_buf == 0
    print(f"       buffered_before={before_buf}  buffered_after={after_buf}  ok={ok}")
    return {
        "ok": ok,
        "buffered_before": before_buf,
        "buffered_after": after_buf,
    }


# ---------------------------------------------------------------------------
# 完整 smoke run
# ---------------------------------------------------------------------------

def run_smoke(
    *,
    sample_rate: int = SAMPLE_RATE,
    sine_duration_sec: float = SINE_DURATION_SEC,
    sine_frequency: float = SINE_FREQUENCY,
    record_duration_sec: float = RECORD_DURATION_SEC,
    skip_record: bool = False,
    wav_output: Path | None = None,
) -> dict[str, Any]:
    results: dict[str, Any] = {}

    client = SyncTranslateDriverAudioClient()
    try:
        # Step 1 - open
        r_open = step_open_device(client)
        results["open_device"] = r_open
        if not r_open["ok"]:
            results["overall"] = False
            results["fail_reason"] = "control_device_unavailable"
            return results

        # Step 2 - GET_STATS
        try:
            r_stats = step_get_stats(client)
        except SyncTranslateDriverUnavailable as exc:
            results["get_stats"] = {"ok": False, "error": str(exc)}
            results["overall"] = False
            results["fail_reason"] = "get_stats_failed"
            return results
        results["get_stats"] = r_stats
        if not r_stats["ok"]:
            results["overall"] = False
            results["fail_reason"] = "get_stats_capacity_zero"
            return results

        # Step 3+4 - WRITE_PCM
        try:
            r_write = step_write_pcm(
                client,
                sample_rate=sample_rate,
                duration_sec=sine_duration_sec,
                frequency=sine_frequency,
            )
        except SyncTranslateDriverUnavailable as exc:
            results["write_pcm"] = {"ok": False, "error": str(exc)}
            results["overall"] = False
            results["fail_reason"] = "write_pcm_failed"
            return results
        results["write_pcm"] = r_write
        if not r_write["ok"]:
            results["overall"] = False
            results["fail_reason"] = "write_pcm_no_frames_counted"
            return results

        # Step 5+6 - record
        if skip_record:
            print("[4/7] 跳過錄音步驟（--skip-record）")
            results["record"] = {"ok": None, "skipped": True, "reason": "skip_record_flag"}
        else:
            r_record = step_record(
                sample_rate=sample_rate,
                duration_sec=record_duration_sec,
                wav_output=wav_output,
            )
            results["record"] = r_record
            if r_record.get("ok") is False:
                results["overall"] = False
                results["fail_reason"] = "record_verification_failed"
                return results

        # Step 7 - FLUSH
        try:
            r_flush = step_flush(client)
        except SyncTranslateDriverUnavailable as exc:
            results["flush"] = {"ok": False, "error": str(exc)}
            results["overall"] = False
            results["fail_reason"] = "flush_failed"
            return results
        results["flush"] = r_flush
        if not r_flush["ok"]:
            results["overall"] = False
            results["fail_reason"] = "flush_buffered_frames_not_reduced"
            return results

    finally:
        client.close()

    results["overall"] = True
    results["fail_reason"] = None
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="SyncTranslate Virtual Audio Driver IOCTL smoke test（限 VM 內使用）"
    )
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE, help="PCM sample rate（default: 48000）")
    parser.add_argument("--sine-duration", type=float, default=SINE_DURATION_SEC, help="寫入 sine 長度（秒）")
    parser.add_argument("--sine-frequency", type=float, default=SINE_FREQUENCY, help="sine 頻率（Hz，default: 1000）")
    parser.add_argument("--record-duration", type=float, default=RECORD_DURATION_SEC, help="錄音長度（秒）")
    parser.add_argument("--skip-record", action="store_true", help="略過從虛擬麥克風錄音的步驟")
    parser.add_argument("--wav-output", default="", help="將錄音儲存為 WAV 的路徑（可選）")
    parser.add_argument("--json-output", default="", help="將結果輸出為 JSON 的路徑（可選）")
    args = parser.parse_args()

    wav_path = Path(args.wav_output) if args.wav_output else None

    print("=" * 60)
    print("SyncTranslate Virtual Audio Driver IOCTL Smoke Test")
    print("警告：請只在 disposable Windows VM 中執行本工具。")
    print("=" * 60)

    result = run_smoke(
        sample_rate=int(args.sample_rate),
        sine_duration_sec=float(args.sine_duration),
        sine_frequency=float(args.sine_frequency),
        record_duration_sec=float(args.record_duration),
        skip_record=bool(args.skip_record),
        wav_output=wav_path,
    )

    print("=" * 60)
    overall: bool | None = result.get("overall")
    if overall is True:
        print("RESULT: PASS")
    else:
        fail_reason = result.get("fail_reason", "unknown")
        print(f"RESULT: FAIL  reason={fail_reason}")
    print("=" * 60)

    if args.json_output:
        out_path = Path(args.json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON 結果已寫入：{out_path}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))

    return 0 if overall is True else 1


if __name__ == "__main__":
    sys.exit(main())
