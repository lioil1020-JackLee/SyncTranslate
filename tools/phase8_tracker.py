#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PLAN_PATH = Path("docs/修改計畫_v2_產品化重構版.md")
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
        title="Meeting mode no-driver scope is documented",
        required_markers=("會議監聽模式", "no-driver", "不檢查 driver"),
    ),
    Gate(
        key="S2",
        title="Driver/bridge protocol v2 boundary is documented",
        required_markers=("protocol version 2", "PCM16", "48000"),
    ),
    Gate(
        key="S3",
        title="Dialogue driver healthcheck is documented",
        required_markers=("dialogue", "driver health", "bridge"),
    ),
    Gate(
        key="S4",
        title="Installer flow is documented",
        required_markers=("installer", "driver", "healthcheck"),
    ),
    Gate(
        key="S5",
        title="Diagnostic report is documented",
        required_markers=("diagnostic", "report", "runtime/model"),
    ),
    Gate(
        key="F1",
        title="Production signing requirement is explicit",
        required_markers=("production-signed", "正式簽章", "Test Mode"),
        optional=True,
    ),
)


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
        print(f"[FAIL] Missing plan: {PLAN_PATH}")
        return 1

    plan_text = PLAN_PATH.read_text(encoding="utf-8")
    rows = []
    print("=" * 88)
    print("SyncTranslate v2 Self-Use Readiness Tracker")
    print("=" * 88)
    print(f"Plan: {PLAN_PATH}")
    print(f"Checked: {datetime.now().isoformat(timespec='seconds')}")
    print("")

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
        print(f"[{status:8}] {gate.key}  {gate.title} ({note})")

    print("")
    print("Current scope:")
    print("- Meeting mode is no-driver and portable.")
    print("- Dialogue mode requires driver, bridge, healthcheck, and installer flow.")
    print("- Test-signed MSI is suitable for VM/lab/self-use validation.")
    print("- Production signing/WHQL is required before normal-user release.")

    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps(
            {
                "timestamp": datetime.now().isoformat(),
                "plan": str(PLAN_PATH),
                "scope": "self_use_test_signed_rc",
                "gates": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("")
    print(f"[OK] Tracker state written to {STATE_PATH}")
    return 1 if blocking_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
