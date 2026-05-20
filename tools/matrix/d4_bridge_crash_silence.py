from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from app.bootstrap.external_runtime import configure_external_ai_runtime
from app.infra.audio.sinks import VirtualMicrophoneSink


class _FlakyBridge:
    def __init__(self) -> None:
        self.calls = 0
        self.primary_failed = 0
        self.silence_accept = 0

    def write_virtual_microphone(self, audio, *, sample_rate: int) -> None:
        del sample_rate
        self.calls += 1
        payload = np.asarray(audio, dtype=np.float32)
        if self.calls == 1:
            self.primary_failed += 1
            raise RuntimeError("simulated_bridge_crash")
        if np.allclose(payload, 0.0):
            self.silence_accept += 1
            return
        raise RuntimeError("expected_silence_payload")

    def flush_virtual_microphone(self) -> None:
        return


def run_d4_bridge_crash_silence(out_json: str | None = None) -> dict[str, object]:
    configure_external_ai_runtime()
    bridge = _FlakyBridge()
    sink = VirtualMicrophoneSink(bridge)

    t = np.arange(960, dtype=np.float32) / 48000.0
    tone = (np.sin(2.0 * np.pi * 440.0 * t) * 0.2).reshape((-1, 1))

    raised = ""
    try:
        sink.play(tone, sample_rate=48000)
    except Exception as exc:
        raised = str(exc)

    stats = sink.diagnostic_stats()
    pass_d4 = bool(
        not raised
        and int(stats.get("write_failures", 0) or 0) >= 1
        and int(stats.get("silence_fallback_writes", 0) or 0) >= 1
        and bridge.silence_accept >= 1
    )

    result = {
        "kind": "d4_bridge_crash_silence",
        "pass_d4": pass_d4,
        "raised": raised,
        "bridge_calls": int(bridge.calls),
        "bridge_primary_failed": int(bridge.primary_failed),
        "bridge_silence_accept": int(bridge.silence_accept),
        "sink_diagnostic": stats,
        "note": "此腳本驗證 app sink 在 bridge 寫入失敗時會改寫 silence，避免 session 中斷。",
    }

    if out_json:
        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="D4 bridge crash silence fallback verifier.")
    parser.add_argument("--out", default="logs/session_reports/d4_bridge_crash_silence.json")
    args = parser.parse_args()

    result = run_d4_bridge_crash_silence(args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if bool(result.get("pass_d4", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
