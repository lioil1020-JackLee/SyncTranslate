from __future__ import annotations

import argparse
import json

from app.infra.audio.virtual_bridge_client import VirtualAudioBridgeClient


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify SyncTranslate bridge PCM loopback")
    parser.add_argument("--bridge-path", default="", help="Path to sync_audio_bridge executable")
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--duration-ms", type=int, default=20)
    parser.add_argument("--tone-hz", type=float, default=440.0)
    parser.add_argument("--timeout-ms", type=int, default=200)
    args = parser.parse_args()

    client = VirtualAudioBridgeClient(bridge_path=str(args.bridge_path or ""))
    try:
        result = client.verify_pcm_loopback(
            sample_rate=int(args.sample_rate),
            duration_ms=int(args.duration_ms),
            tone_hz=float(args.tone_hz),
            timeout_ms=int(args.timeout_ms),
        )
        payload = {
            "ok": bool(result.ok),
            "frames_written": int(result.frames_written),
            "stats_delta_frames": int(result.stats_delta_frames),
            "shared_memory_frames": int(result.shared_memory_frames),
            "event_signaled": bool(result.event_signaled),
            "latency_ms": float(result.latency_ms),
            "error": str(result.error or ""),
        }
        print(json.dumps(payload, ensure_ascii=False))
        return 0 if result.ok else 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
