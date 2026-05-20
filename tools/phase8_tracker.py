#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PLAN_PATH = Path("docs/修改計畫.md")
STATE_PATH = Path("logs/phase8_tracker.json")


@dataclass(frozen=True)
class Gate:
    key: str
    title: str
    required_markers: tuple[str, ...]
    optional: bool = False


GATES = (
    Gate(
        key="S1",
        title="Self-use scope is documented",
        required_markers=("免費自用", "Test Mode", "不依賴 Voicemeeter"),
    ),
    Gate(
        key="S2",
        title="App and bridge package is documented",
        required_markers=("package_audio_bridge.ps1", "package_onedir.ps1", "sync_audio_bridge.exe"),
    ),
    Gate(
        key="S3",
        title="Test-signed driver path is documented",
        required_markers=("test-signed", "SyncTranslateVirtualAudioDriver.msi", "Administrator"),
    ),
    Gate(
        key="S4",
        title="Release artifacts are documented",
        required_markers=("SyncTranslate-onedir-windows.zip", ".sha256", "GitHub Release"),
    ),
    Gate(
        key="S5",
        title="Local safety scan is documented",
        required_markers=("Defender", "defender_scan.json"),
    ),
    Gate(
        key="F1",
        title="Formal paid signing path is explicitly deferred",
        required_markers=("EV", "WHQL", "暫停"),
        optional=True,
    ),
)


def _safe_text(value: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    try:
        value.encode(encoding)
        return value
    except UnicodeEncodeError:
        return value.encode(encoding, errors="replace").decode(encoding, errors="replace")


def emit(value: str = "") -> None:
    print(_safe_text(value))


def marker_status(plan_text: str, gate: Gate) -> str:
    normalized_plan = plan_text.casefold()
    present = sum(1 for marker in gate.required_markers if marker.casefold() in normalized_plan)
    if present == len(gate.required_markers):
        return "OPTIONAL" if gate.optional else "READY"
    if present:
        return "PARTIAL"
    return "MISSING"


def main() -> int:
    if not PLAN_PATH.exists():
        emit(f"[FAIL] Missing plan: {PLAN_PATH}")
        return 1

    plan_text = PLAN_PATH.read_text(encoding="utf-8")
    rows = []
    emit("=" * 88)
    emit("SyncTranslate Self-Use Readiness Tracker")
    emit("=" * 88)
    emit(f"Plan: {PLAN_PATH}")
    emit(f"Checked: {datetime.now().isoformat(timespec='seconds')}")
    emit("")

    blocking_missing = False
    for gate in GATES:
        status = marker_status(plan_text, gate)
        if status == "MISSING" and not gate.optional:
            blocking_missing = True
        rows.append(
            {
                "key": gate.key,
                "title": gate.title,
                "status": status,
                "optional": gate.optional,
            }
        )
        note = "future formal-release option" if gate.optional else "self-use gate"
        emit(f"[{status:8}] {gate.key}  {gate.title} ({note})")

    emit("")
    emit("Current scope:")
    emit("- Free self-use / GitHub friends testing")
    emit("- Test-signed driver and Windows Test Mode are expected")
    emit("- EV, WHQL, attestation signing, and SmartScreen reputation are not current blockers")

    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps(
            {
                "timestamp": datetime.now().isoformat(),
                "plan": str(PLAN_PATH),
                "scope": "self_use_test_signed",
                "gates": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    emit("")
    emit(f"[OK] Tracker state written to {STATE_PATH}")
    return 1 if blocking_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
