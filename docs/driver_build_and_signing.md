# Driver Build And Signing

SyncTranslate v2 has two product modes:

- meeting mode does not require the virtual audio driver, bridge, Test Mode, or WDK.
- dialogue mode requires SyncTranslate Virtual Speaker, SyncTranslate Virtual Microphone, and the bridge.

The development driver is test-signed. It is for disposable VM/lab validation only. A production release needs production driver signing; normal users should not be asked to manually enable Windows Test Mode.

## Requirements

- Windows 10/11
- Visual Studio 2022 Build Tools
- Windows SDK
- Windows Driver Kit
- WDK tools on PATH or installed under `C:\Program Files (x86)\Windows Kits\10`
- Optional WiX tooling for MSI packaging

Check the environment:

```powershell
powershell -ExecutionPolicy Bypass -File drivers/synctranslate_virtual_audio/scripts/check_wdk_environment.ps1
```

## Build Package

Fetch SysVAD if needed, then build:

```powershell
powershell -ExecutionPolicy Bypass -File drivers/synctranslate_virtual_audio/scripts/fetch_sysvad.ps1
powershell -ExecutionPolicy Bypass -File drivers/synctranslate_virtual_audio/scripts/build_driver_package.ps1
```

The fixed output path is:

```text
artifacts/driver/synctranslate_virtual_audio/package
```

`build_driver_package.ps1` applies the SyncTranslate SysVAD overlay before build. The overlay enforces the v2 virtual endpoint boundary: 48000Hz / PCM16 / 2ch.

## Test Mode

Development test-signed packages require Windows Test Mode:

```powershell
bcdedit /set testsigning on
```

Reboot after changing Test Mode. The install script reports the command but does not force a reboot.

## Install Test Driver

Use an Administrator PowerShell in a disposable VM/lab:

```powershell
powershell -ExecutionPolicy Bypass -File drivers/synctranslate_virtual_audio/scripts/install_test_driver.ps1 -AllowHostInstall
```

The script checks Administrator rights, Test Mode, stages/installs the package, and runs format verification.

For SysVAD-based packages, installing only the root MEDIA INF into the Driver
Store is not enough. The product installer creates a `Root\SyncTranslateVirtualAudio`
device node and then runs `pnputil /add-driver ... /install` again so Windows
binds the staged driver to that root device. If `devcon.exe` is unavailable, the
MSI uses the bundled `create_root_device.ps1` SetupAPI fallback.

The APO/extension INFs are optional and are not required for the product path.

## Verify Devices And Format

```powershell
powershell -ExecutionPolicy Bypass -File drivers/synctranslate_virtual_audio/scripts/verify_driver_format.ps1 -JsonOutput downloads/validation/driver_format.json
```

Expected endpoint format:

```text
48000Hz / PCM16 / 2ch / interleaved stereo
```

Device Manager must show more than the MEDIA root device. A correct dialogue-mode install has:

- `SyncTranslate Virtual Audio Device` under **Sound, video and game controllers** with no warning icon.
- `SyncTranslate Virtual Speaker` under **Audio inputs and outputs**.
- `SyncTranslate Virtual Microphone` under **Audio inputs and outputs**.

If only `SyncTranslate Virtual Audio Device` appears under **Sound, video and game controllers**, the kernel driver loaded but Windows did not create MMDevice endpoints yet. Treat that as a failed dialogue-mode driver validation and collect `downloads/validation/driver_format.json` plus `setupapi.dev.log`.

`verify_driver_format.ps1` checks the endpoint device format stored in
`PKEY_AudioEngine_DeviceFormat`. Windows shared-mode `IAudioClient.GetMixFormat`
may still report 48 kHz float32 because the Windows audio engine mixes in float;
that shared mix value is recorded in JSON for diagnostics but is not the v2
driver boundary. The v2 boundary remains 48000Hz / PCM16 / 2ch.

## Uninstall

```powershell
powershell -ExecutionPolicy Bypass -File drivers/synctranslate_virtual_audio/scripts/uninstall_test_driver.ps1
```

The uninstall script enumerates SyncTranslate candidates and uses `pnputil` only on matching SyncTranslate packages. It must not delete non-SyncTranslate drivers.
