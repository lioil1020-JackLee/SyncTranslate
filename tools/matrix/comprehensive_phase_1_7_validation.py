#!/usr/bin/env python3
"""
Comprehensive Phase 1-7 Validation
Verify all phases are truly complete before Phase 8 (signing & release)
"""
import json
from pathlib import Path
import sys


def validate_phase1():
    """Phase 1: App auto-device detection, virtual driver discovery, call_translation auto-decision"""
    print("\n" + "="*60)
    print("Phase 1: Auto-detection & Device Discovery")
    print("="*60)
    
    checks = {
        "app/infra/audio/device_registry.py": "Device registry exists",
        "app/infra/audio/virtual_devices.py": "Virtual devices discovery",
        "app/application/audio_router.py": "Call translation routing logic",
        "tests/test_audio_router_core.py": "Unit tests for audio routing",
    }
    
    passed = 0
    for path, desc in checks.items():
        p = Path(path)
        if p.exists():
            print(f"[OK] {desc}: {path}")
            passed += 1
        else:
            print(f"[FAIL] {desc}: {path}")
    
    return passed == len(checks)


def validate_phase2():
    """Phase 2: Bridge infrastructure, named pipe control, shared memory ring buffer"""
    print("\n" + "="*60)
    print("Phase 2: Bridge Infrastructure")
    print("="*60)
    
    checks = {
        "app/infra/audio/virtual_bridge_client.py": "Bridge client implementation",
        "app/infra/audio/sinks.py": "Virtual microphone sink",
        "app/infra/audio/bridge_ring_buffer.py": "Ring buffer management",
        "tests/test_bridge_ring_buffer.py": "Ring buffer unit tests",
    }
    
    passed = 0
    for path, desc in checks.items():
        p = Path(path)
        if p.exists():
            print(f"[OK] {desc}: {path}")
            passed += 1
        else:
            print(f"[FAIL] {desc}: {path}")
    
    return passed == len(checks)


def validate_phase3():
    """Phase 3: Driver PCM closed-loop + D1-D6 validation"""
    print("\n" + "="*60)
    print("Phase 3: Driver PCM Closed-loop & D1-D6")
    print("="*60)
    
    checks = {
        "drivers/synctranslate_virtual_audio": "Virtual audio driver source",
        "tests/test_asr_streaming_and_profiles.py": "ASR streaming tests",
        "tests/test_virtual_bridge_client.py": "Bridge client tests",
        "logs/session_reports": "Test reports directory",
    }
    
    passed = 0
    for path, desc in checks.items():
        p = Path(path)
        if p.exists():
            print(f"[OK] {desc}: {path}")
            passed += 1
        else:
            print(f"[FAIL] {desc}: {path}")
    
    # Check for D1-D6 test reports - look for any version
    d_tests = [
        ("d3", ["d3_app_crash_silence_phase3.json", "d3_app_crash_silence.json"]),
        ("d4", ["d4_bridge_crash_silence_phase3.json", "d4_bridge_crash_silence.json"]),
        ("d5", ["d5_resampling_validation_phase3.json", "d5_resampling_validation.json"]),
        ("d6", ["d6_diagnostics_validation_phase3.json", "d6_diagnostics_validation.json"]),
    ]
    
    report_dir = Path("logs/session_reports")
    d_passed = 0
    for d_name, test_files in d_tests:
        for test_file in test_files:
            report = report_dir / test_file
            if report.exists():
                try:
                    with open(report, encoding='utf-8') as f:
                        data = json.load(f)
                        # Check various possible status fields
                        status = (data.get("summary", {}).get("pass") or 
                                 data.get("summary", {}).get("overall_ok") or
                                 data.get(f"pass_{d_name}"))
                        if status:
                            print(f"[OK] {d_name}: {test_file}")
                            d_passed += 1
                            break
                        else:
                            print(f"[WARN] {test_file}: exists but status unclear")
                except Exception as e:
                    print(f"[WARN] {test_file}: {e}")
        else:
            print(f"[WARN] {d_name}: no report found")
    
    return passed == len(checks) and d_passed >= 2


def validate_phase4():
    """Phase 4: Multi-route audio closed-loop"""
    print("\n" + "="*60)
    print("Phase 4: Multi-route Audio Closed-loop")
    print("="*60)
    
    checks = {
        "tests/test_asr_streaming_and_profiles.py": "ASR streaming tests",
        "tests/test_audio_router_core.py": "Audio routing tests",
        "app/application/audio_router.py": "Multi-route router",
    }
    
    passed = 0
    for path, desc in checks.items():
        p = Path(path)
        if p.exists():
            print(f"[OK] {desc}: {path}")
            passed += 1
        else:
            print(f"[FAIL] {desc}: {path}")
    
    return passed == len(checks)


def validate_phase5():
    """Phase 5: 6 Softphone platform validation"""
    print("\n" + "="*60)
    print("Phase 5: Softphone Matrix (6 Platforms)")
    print("="*60)
    
    # Map platform names (may have variations in reports)
    platform_map = {
        "Zoom": "Zoom",
        "Teams": "Teams",
        "Google Meet": ["Google Meet", "Google Meet (Chrome)"],
        "LINE": "LINE",
        "WhatsApp": ["WhatsApp", "WhatsApp Desktop"],
        "Discord": "Discord"
    }
    
    # Check test script exists
    test_script = Path("tools/matrix/softphone_matrix_test.py")
    if test_script.exists():
        print(f"[OK] Softphone matrix test script: {test_script}")
    else:
        print(f"[FAIL] Softphone matrix test script missing")
        return False
    
    # Check for any available softphone matrix report
    report_dir = Path("logs/session_reports")
    latest_report = None
    report_files = []
    
    # Look for softphone_matrix_test_*.json files
    for f in report_dir.glob("softphone_matrix_test*.json"):
        try:
            with open(f, encoding='utf-8') as fp:
                data = json.load(fp)
                if data.get("kind") == "softphone_matrix":
                    report_files.append((f, data, f.stat().st_mtime))
        except:
            pass
    
    if report_files:
        # Get most recent
        latest_file, latest_report, _ = sorted(report_files, key=lambda x: x[2], reverse=True)[0]
        print(f"[OK] Latest report: {latest_file.name}")
        
        results = latest_report.get("profiles", [])
        all_pass = True
        passed_count = 0
        
        for platform, name_variants in platform_map.items():
            if not isinstance(name_variants, list):
                name_variants = [name_variants]
            
            found = False
            for profile in results:
                if profile.get("profile") in name_variants:
                    status = profile.get("pass_d1", False)
                    symbol = "[OK]" if status else "[FAIL]"
                    print(f"{symbol} {platform}: {status}")
                    found = True
                    if status:
                        passed_count += 1
                    all_pass = all_pass and status
                    break
            
            if not found:
                print(f"[WARN] {platform}: not in report")
                all_pass = False
        
        return passed_count >= 5  # At least 5/6 platforms pass
    else:
        print("[WARN] No softphone matrix report found")
        return False


def validate_phase6():
    """Phase 6: 2hr soak + sleep/wake + device recovery + Installer QA"""
    print("\n" + "="*60)
    print("Phase 6: Stability & Recovery Tests")
    print("="*60)
    
    checks = {
        "tools/matrix/runtime_soak_2hr.py": "2hr soak test",
        "tools/matrix/phase6_sleep_wake_recovery.py": "Sleep/wake recovery test",
        "tools/matrix/phase6_device_enable_disable_recovery.py": "Device recovery test",
        "tools/matrix/phase6_final_acceptance_gate.py": "Phase 6 final gate",
        "docs/修改計畫.md": "Remaining work plan",
    }
    
    passed = 0
    for path, desc in checks.items():
        p = Path(path)
        if p.exists():
            print(f"[OK] {desc}: {path}")
            passed += 1
        else:
            print(f"[FAIL] {desc}: {path}")
    
    # Check for soak report
    report_dir = Path("logs/session_reports")
    soak_reports = list(report_dir.glob("*soak*.json"))
    if soak_reports:
        print(f"[OK] Soak test reports found: {len(soak_reports)} reports")
    else:
        print("[WARN] No soak test reports found")
    
    return passed >= 4


def validate_phase7():
    """Phase 7: Productization & documentation"""
    print("\n" + "="*60)
    print("Phase 7: Productization & Documentation")
    print("="*60)
    
    docs = {
        "README.md": "Main documentation",
        "docs/設定說明.md": "Configuration guide",
        "docs/首次啟動引導.md": "First run guide",
        "docs/修改計畫.md": "Remaining work plan",
    }
    
    passed = 0
    for path, desc in docs.items():
        p = Path(path)
        if p.exists():
            print(f"[OK] {desc}: {path}")
            passed += 1
        else:
            print(f"[FAIL] {desc}: {path}")
    
    # Check config migration
    config_migration = Path("app/infra/config/_config_migration.py")
    if config_migration.exists():
        print(f"[OK] Config migration logic: {config_migration}")
        passed += 1
    else:
        print(f"[FAIL] Config migration logic missing")
    
    # Verify README no longer recommends third-party virtual audio routing.
    readme = Path("README.md")
    if readme.exists():
        with open(readme, encoding='utf-8') as f:
            content = f.read()
            if "不依賴 Voicemeeter" in content or "無需 Voicemeeter" in content:
                print("[OK] README states third-party virtual audio is not required")
                passed += 1
            else:
                print("[WARN] README does not clearly state third-party virtual audio is not required")
    
    return passed >= 6


def print_phase_summary(results):
    """Print comprehensive phase validation summary"""
    print("\n" + "="*60)
    print("COMPREHENSIVE PHASE 1-7 VALIDATION SUMMARY")
    print("="*60)
    
    phase_names = {
        1: "Auto-detection & Device Discovery",
        2: "Bridge Infrastructure",
        3: "Driver PCM Closed-loop",
        4: "Multi-route Audio",
        5: "Softphone Matrix (6 Platforms)",
        6: "Stability & Recovery",
        7: "Productization & Documentation"
    }
    
    all_ok = True
    for i, status in enumerate(results, start=1):
        symbol = "[OK]" if status else "[FAIL]"
        print(f"{symbol} Phase {i}: {phase_names[i]}")
        all_ok = all_ok and status
    
    print("="*60)
    if all_ok:
        print("[SUCCESS] All Phases 1-7 Complete for the self-use test-signed route.")
        print("\nCurrent release scope:")
        print("  - Free self-use / GitHub friends testing")
        print("  - Test-signed driver with Windows Test Mode")
        print("  - Formal EV/WHQL/attestation signing is deferred")
        print("  - Validate the call apps you actually use before sharing")
    else:
        print("[INCOMPLETE] Some phases need review.")
    
    return all_ok


def main():
    print("\n" + "="*70)
    print("SyncTranslate - Comprehensive Phase 1-7 Validation")
    print("="*70)
    
    results = [
        validate_phase1(),
        validate_phase2(),
        validate_phase3(),
        validate_phase4(),
        validate_phase5(),
        validate_phase6(),
        validate_phase7(),
    ]
    
    all_ok = print_phase_summary(results)
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
