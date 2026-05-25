from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "drivers" / "synctranslate_virtual_audio" / "scripts"


def _read(name: str) -> str:
    return (SCRIPTS / name).read_text(encoding="utf-8", errors="ignore")


def test_driver_readiness_scripts_exist() -> None:
    for name in (
        "check_wdk_environment.ps1",
        "build_driver_package.ps1",
        "install_test_driver.ps1",
        "uninstall_test_driver.ps1",
        "verify_driver_format.ps1",
    ):
        assert (SCRIPTS / name).exists(), name


def test_wdk_environment_script_checks_required_tools() -> None:
    text = _read("check_wdk_environment.ps1")
    for token in ("msbuild", "infverif", "stampinf", "signtool", "PASS", "WARN", "FAIL"):
        assert token in text


def test_build_and_install_scripts_expose_safe_driver_flow_keywords() -> None:
    build = _read("build_driver_package.ps1")
    install = _read("install_test_driver.ps1")
    install_package = _read("install_driver_package.ps1")
    uninstall = _read("uninstall_test_driver.ps1")
    uninstall_package = _read("uninstall_driver_package.ps1")
    verify = _read("verify_driver_format.ps1")

    assert "apply_sysvad_overlay.ps1" in build
    assert "build_driver_poc.ps1" in build
    assert "removing stale SysVAD intermediate outputs" in _read("build_driver_poc.ps1")
    assert "Test-InfHasTextContent" in build
    assert "using WDK-generated package INF" in build
    assert "do not ship template-repaired INFs" in build
    assert "limited endpoint tables to SyncTranslate virtual speaker + virtual microphone" in _read("apply_sysvad_overlay.ps1")
    assert "bcdedit" in install
    assert "Test Mode" in install
    assert "pnputil" in uninstall
    assert "devcon remove" in uninstall
    assert "stale SyncTranslate device instance" in uninstall
    assert "stale MMDevice endpoint cache" in uninstall
    assert "AudioEndpointBuilder" in uninstall
    assert "/remove-device" in uninstall_package
    assert "stale media device instance" in uninstall_package
    assert "SyncTranslateClearStaleMmDeviceCache" in uninstall_package
    assert "/RU SYSTEM" in uninstall_package
    assert "oem\\d+\\.inf" in uninstall_package
    assert "Block" in uninstall_package
    assert "InstallComponentInfs" in install_package
    assert "[switch]$InstallComponentInfs = $true" not in install_package
    assert "Invoke-NativeForDriverLog" in install_package
    assert "Test-SyncTranslateMediaDeviceBound" in install_package
    assert "Remove-StaleSyncTranslateMediaDevices" in install_package
    assert "/remove-device" in install_package
    assert "MEDIA driver did not bind successfully" in install_package
    assert "Device node created" not in install_package
    assert "StageComponentInfs" in install_package
    assert "reconciling driver store binding with pnputil" in install_package
    assert "devcon already created the root device" in install_package
    assert "create_root_device.ps1" in install_package
    assert "SetupAPI fallback" in install_package
    assert "binding staged driver to the SetupAPI-created root device with pnputil" in install_package
    assert "installing componentized INF with pnputil" in install_package
    assert "without /install" in install_package
    assert "skipping optional componentized APO/extension INFs" in install_package
    assert r"\boem\d+\.inf\b" in uninstall
    assert "componentizedaudiosampleextension" in uninstall
    assert "componentizedaposample" in uninstall
    assert "AllowExistingSyncTranslateDevices" not in install
    assert "[SyncTranslate] Virtual audio driver package installed." in install_package
    assert "ComponentizedApoSample.inf" in install_package
    assert "ComponentizedAudioSampleExtension.inf" in install_package
    assert "exit 0" in install_package
    assert "SyncTranslate" in uninstall
    assert "48000Hz PCM16 2ch" in verify
    assert "-JsonOutput" in verify
    assert "Get-PnpDevice -PresentOnly -Class AudioEndpoint" in verify
    assert "virtual_speaker_endpoint" in verify
    assert "virtual_microphone_endpoint" in verify
    assert "single_sync_translate_device" in verify
    assert "no_problem_sync_translate_devices" in verify
    assert "endpoint_device_format" in verify


def test_driver_msi_uses_one_click_install_package_action() -> None:
    wxs = (SCRIPTS.parent / "installer" / "Package.wxs").read_text(encoding="utf-8", errors="ignore")
    preflight = _read("preflight_driver_install.ps1")
    package_script = _read("package_driver_msi.ps1")

    assert 'Id="InstallDriverPackage"' in wxs
    assert 'Id="CleanStaleDriverPayload"' in wxs
    assert "Remove-Item -LiteralPath $p -Recurse -Force" in wxs
    assert "install_driver_package.ps1" in wxs
    assert "-AllowHostInstall" in wxs
    assert "-SkipPreflight" in wxs
    assert 'Return="check"' in wxs
    assert "create_root_device.ps1" not in wxs
    assert "create_root_device.ps1" in package_script
    assert "devcon-free MSI path" in package_script
    assert "preflight_driver_install.ps1" in package_script
    assert "msi_has_install_custom_action" in preflight
    assert "one-click elevated install custom action" in preflight


def test_control_device_dispatch_forwards_portcls_irps() -> None:
    control = (SCRIPTS.parent / "overlay" / "synctranslate_control.cpp").read_text(encoding="utf-8", errors="ignore")

    assert "g_syncTranslatePreviousCreate" in control
    assert "g_syncTranslatePreviousClose" in control
    assert "g_syncTranslatePreviousCleanup" in control
    assert "SyncTranslateForwardIrp(g_syncTranslatePreviousCreate" in control
    assert "SyncTranslateForwardIrp(g_syncTranslatePreviousClose" in control
    assert "SyncTranslateForwardIrp(g_syncTranslatePreviousCleanup" in control
    assert "SyncTranslateForwardIrp(g_syncTranslatePreviousDeviceControl" in control
