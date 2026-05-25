from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "hyperv" / "run_driver_vm_validation.ps1"
README = ROOT / "tools" / "hyperv" / "README.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def test_hyperv_validation_script_exists_and_has_required_parameters() -> None:
    text = _read(SCRIPT)
    for token in (
        "VMName",
        "GuestUser",
        "GuestPassword",
        "Credential",
        "RepoPath",
        "GuestWorkDir",
        "BuildInGuest",
        "SkipBuild",
        "EnableTestSigning",
        "CreateCheckpoint",
        "IncludeRuntimes",
        "OutputDir",
        "NoReboot",
        "CollectDiagnostics",
    ):
        assert token in text


def test_hyperv_validation_script_contains_host_and_guest_flow() -> None:
    text = _read(SCRIPT)
    for token in (
        "Checkpoint-VM",
        "Copy-VMFile",
        "PowerShell Direct",
        "New-PSSession -VMName",
        "Invoke-Command -Session",
        "bcdedit /set testsigning on",
        "install_test_driver.ps1",
        "verify_driver_format.ps1",
        "preflight_release_check.py --mode dialogue",
        "validate_windows_audio_runtime.py",
        "audio_smoke_test.py dialogue-passthrough --duration 0.1",
        "export_diagnostics_bundle.py",
        "Copy-Item -FromSession",
        "Get-Command uv",
        "IncludeRuntimeAssets",
        "runtimes",
        "SysVAD samples already present",
        "fetch_sysvad.ps1 skipped",
    ):
        assert token in text


def test_hyperv_validation_script_has_safety_language() -> None:
    text = _read(SCRIPT)
    for token in (
        "disposable VM",
        "TESTSIGNING is for test VMs only",
        "No certificates, private keys, or passwords are stored",
        "Copy-VMFile is not available",
        "elevated Hyper-V host PowerShell session",
        "shared folder",
        "git clone",
    ):
        assert token in text


def test_hyperv_validation_script_does_not_close_guest_session_for_skips() -> None:
    text = _read(SCRIPT)
    assert "fetch_sysvad.ps1 skipped" in text
    assert not re.search(r"fetch_sysvad\.ps1 skipped\.[\s\S]{0,120}exit 0", text)
    assert not re.search(r"install skipped because -SkipBuild[\s\S]{0,160}exit 0", text)


def test_hyperv_readme_documents_vm_requirements_and_outputs() -> None:
    text = _read(README)
    for token in (
        "disposable VM",
        "Test-signed drivers",
        "TESTSIGNING",
        "WDK",
        "driver signing",
        "PowerShell Direct",
        "driver_format.json",
        "preflight_dialogue.json",
        "windows_audio_runtime.json",
        "diagnostics_bundle",
    ):
        assert token in text
