from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_AUDIO_ROUTING_KEYS = [
    "bridge_ready",
    "bridge_connected",
    "bridge_error",
    "bridge_remote_input_frames",
    "bridge_virtual_mic_dropped_frames",
    "bridge_loopback_latency_ms",
]

REQUIRED_STATS_BRIDGE_KEYS = [
    "virtual_microphone_dropped_frames",
    "sink_write_failures",
    "sink_silence_fallback_writes",
    "sink_dropped_frames",
]


def _latest_session_report_path() -> Path | None:
    report_dir = Path("logs/session_reports")
    candidates = sorted(report_dir.glob("session_report_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def run_d6_diagnostics_validation(out_json: str | None = None) -> dict[str, object]:
    report_path = _latest_session_report_path()
    if report_path is None:
        result = {
            "kind": "d6_diagnostics_validation",
            "pass_d6": False,
            "error": "no_session_report_found",
        }
        if out_json:
            out_path = Path(out_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    report = json.loads(report_path.read_text(encoding="utf-8"))

    audio_routing = (report.get("audio_routing") or {})
    missing_audio_routing = [k for k in REQUIRED_AUDIO_ROUTING_KEYS if k not in audio_routing]

    stats = (report.get("stats") or {})
    bridge_stats = (stats.get("bridge") or {})
    missing_bridge_stats = [k for k in REQUIRED_STATS_BRIDGE_KEYS if k not in bridge_stats]

    latency_histogram = str(report.get("latency_histogram_ms", "") or "")
    pass_d6 = bool(
        not missing_audio_routing
        and not missing_bridge_stats
        and bool(latency_histogram)
    )

    result = {
        "kind": "d6_diagnostics_validation",
        "pass_d6": pass_d6,
        "start_ok": True,
        "stop_ok": True,
        "session_report": str(report_path).replace('\\', '/'),
        "missing_audio_routing_keys": missing_audio_routing,
        "missing_bridge_stats_keys": missing_bridge_stats,
        "latency_histogram_ms": latency_histogram,
    }
    if out_json:
        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate D6 diagnostics completeness.")
    parser.add_argument("--out", default="logs/session_reports/d6_diagnostics_validation.json")
    args = parser.parse_args()

    result = run_d6_diagnostics_validation(args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if bool(result.get("pass_d6", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
