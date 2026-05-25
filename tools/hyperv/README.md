# Hyper-V Driver Validation Harness

This folder contains host-side tooling for validating the SyncTranslate virtual
audio driver inside a Hyper-V Windows VM.

Use a disposable VM. Driver install, TESTSIGNING, WDK builds, and virtual audio
endpoint tests should not be run first on a daily-use machine.

## Requirements

- Hyper-V enabled on the host.
- Hyper-V PowerShell module installed.
- A running Windows guest VM.
- PowerShell Direct access to the VM.
- Guest account with Administrator rights.
- Python and `uv` installed in the guest.
- WDK / Visual Studio Build Tools in the guest if using `-BuildInGuest`.
- SyncTranslate driver signing material is not provided by this script. Do not
  commit certificates or private keys. Test-signed drivers are for VM/lab use
  only; production release requires production driver signing.

## Example

```powershell
$cred = Get-Credential
powershell -ExecutionPolicy Bypass -File tools/hyperv/run_driver_vm_validation.ps1 `
  -VMName SyncTranslateDriverVM `
  -Credential $cred `
  -RepoPath E:\py\SyncTranslate `
  -GuestWorkDir C:\SyncTranslate `
  -BuildInGuest `
  -IncludeRuntimes `
  -EnableTestSigning `
  -CreateCheckpoint `
  -CollectDiagnostics
```

If TESTSIGNING is enabled, the VM normally needs a reboot. Omit `-NoReboot` for
the harness to reboot and reconnect through PowerShell Direct.

## Flow

The host script:

1. Checks the Hyper-V module.
2. Checks that the VM exists and is running.
3. Optionally creates a checkpoint named `SyncTranslate-driver-test-YYYYMMDD-HHMMSS`.
4. Opens a PowerShell Direct session using `New-PSSession -VMName`.
5. Checks that `Copy-VMFile` is available, then copies a zipped repo snapshot to
   the VM and expands it under `GuestWorkDir`.
   By default the snapshot excludes host `runtimes/` and preserves any guest
   `runtimes/` already present. Pass `-IncludeRuntimes` when you want dialogue
   preflight to validate the same ASR/LLM/bridge runtime assets as the host.
6. Runs guest inventory checks: Windows version, user/admin state, `python --version`, `uv --version`.
7. Runs `drivers/synctranslate_virtual_audio/scripts/check_wdk_environment.ps1`.
8. Optionally runs `fetch_sysvad.ps1` and `build_driver_package.ps1`.
9. Optionally runs `bcdedit /set testsigning on` and reboots.
10. Runs `install_test_driver.ps1 -AllowHostInstall`.
11. Runs `verify_driver_format.ps1`.
12. Runs dialogue release preflight, Windows audio runtime validation, dialogue
    passthrough smoke, and optional diagnostics bundle export.
13. Copies `downloads/validation/` back to the host output directory.

If `Copy-VMFile` is unavailable, use a shared folder or clone the repo inside
the VM, then rerun with `-GuestWorkDir` pointing at that path.

## Output

Default output directory:

```text
downloads/validation/hyperv
```

Expected artifacts include:

- `host_driver_vm_validation_*.log`
- `guest_driver_vm_validation_*.log`
- `summary_*.json`
- copied guest validation files such as:
  - `driver_format.json`
  - `preflight_dialogue.json`
  - `windows_audio_runtime.json`
  - `diagnostics_bundle_*.zip` when `-CollectDiagnostics` is used

## Safety

- Do not pass or store plaintext passwords in files.
- Prefer `-Credential (Get-Credential)`.
- Do not commit driver certificates or private keys.
- The script never deletes the VM.
- It performs one install attempt per run and does not retry indefinitely.
- Meeting mode does not require the driver; this harness is for dialogue-mode
  driver/bridge validation only.
