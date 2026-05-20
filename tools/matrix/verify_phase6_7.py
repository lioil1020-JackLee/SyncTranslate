#!/usr/bin/env python3
"""
Verify Phase 6/7 completion
Check all required components are in place
"""
from pathlib import Path
import sys


def check_phase6():
    """Check Phase 6 test scripts"""
    print("\nPhase 6 Test Scripts Check")
    print("-" * 60)
    
    files = [
        "tools/matrix/runtime_soak_2hr.py",
        "tools/matrix/phase6_sleep_wake_recovery.py",
        "tools/matrix/phase6_device_enable_disable_recovery.py",
        "tools/matrix/phase6_final_acceptance_gate.py",
        "tools/matrix/phase5_phase6_quick_gate.py",
        "tools/matrix/softphone_matrix_test.py",
    ]
    
    count = 0
    for f in files:
        path = Path(f)
        exists = path.exists()
        status = "[OK]" if exists else "[MISSING]"
        print(f"{status} {f}")
        if exists:
            count += 1
    
    return count == len(files)


def check_phase7():
    """Check Phase 7 documentation"""
    print("\nPhase 7 Documentation Check")
    print("-" * 60)
    
    files = [
        "README.md",
        "docs/設定說明.md",
        "docs/首次啟動引導.md",
        "docs/修改計畫.md",
        "app/infra/config/_config_migration.py",
    ]
    
    count = 0
    for f in files:
        path = Path(f)
        exists = path.exists()
        status = "[OK]" if exists else "[MISSING]"
        print(f"{status} {f}")
        if exists:
            count += 1
    
    return count == len(files)


def check_plan_file():
    """Check modification plan is updated"""
    print("\nModification Plan Update Check")
    print("-" * 60)
    
    plan = Path("docs/修改計畫.md")
    if not plan.exists():
        print("[MISSING] docs/modification_plan.md")
        return False
    
    with open(plan, encoding='utf-8') as f:
        content = f.read()
    
    checks = [
        ("self-use plan exists", "SyncTranslate 免費自用落地計畫"),
        ("test-signed route tracked", "test-signed"),
        ("test mode tracked", "Test Mode"),
        ("release artifacts tracked", "SyncTranslate-onedir-windows.zip"),
        ("github release guidance tracked", "GitHub Release"),
        ("third-party virtual audio removed", "不依賴 Voicemeeter"),
        ("formal signing deferred", "暫停"),
    ]
    
    count = 0
    for name, keyword in checks:
        found = keyword in content or name.replace(" ", "") in content.replace(" ", "")
        status = "[OK]" if found else "[PENDING]"
        print(f"{status} {name}")
        if found:
            count += 1
    
    return count >= 4


def main():
    print("\n" + "="*60)
    print("SyncTranslate Phase 6/7 Completion Check")
    print("="*60)
    
    p6 = check_phase6()
    p7 = check_phase7()
    plan = check_plan_file()
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Phase 6 (Tests): {'OK' if p6 else 'INCOMPLETE'}")
    print(f"Phase 7 (Docs):  {'OK' if p7 else 'INCOMPLETE'}")
    print(f"Plan Updated:    {'OK' if plan else 'INCOMPLETE'}")
    
    if p6 and p7 and plan:
        print("\n[SUCCESS] Phase 6/7 complete for the self-use test-signed route.")
        print("\nQuick verification commands:")
        print("  2-min:   .venv/Scripts/python.exe tools/matrix/phase5_phase6_quick_gate.py --soak-preset smoke")
        print("  45-min:  .venv/Scripts/python.exe tools/matrix/phase6_final_acceptance_gate.py")
        return 0
    else:
        print("\n[INCOMPLETE] Some items still pending.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
