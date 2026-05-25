# SyncTranslate Virtual Audio Driver

## v2 virtual audio boundary

The v2 driver/app boundary is fixed to 48000 Hz / PCM16 / 2ch interleaved
stereo. `IOCTL_SYNCTRANSLATE_AUDIO_WRITE_PCM` accepts raw PCM16 stereo bytes,
and its input length must be aligned to 4 bytes per stereo frame. The kernel
ring buffer stores and reports counts in stereo frames.

App internal audio remains float32 `AudioFrame`; conversion to PCM16 stereo 48k
happens only at the virtual audio boundary. Meeting mode does not require this
driver or the bridge. Dialogue mode requires the virtual speaker/microphone and
the bridge.

See `format_contract_v2.md` for the detailed contract.

這個目錄放的是 SyncTranslate 自有虛擬音效驅動的 build、overlay、package、安裝與驗證工具。

目前目標是取代 Voicemeeter / VB-CABLE 的「TTS 送進會議麥克風」用途。

目前安全架構：

```text
SyncTranslate App / TTS
  -> \\.\SyncTranslateVirtualAudioControl
  -> IOCTL_SYNCTRANSLATE_AUDIO_WRITE_PCM
  -> driver NonPagedPool PCM ring buffer
  -> SyncTranslate Virtual Microphone
  -> meeting app
```

## 目前狀態

已完成：

- SysVAD overlay build。
- Driver package build。
- MSI wrapper package。
- Kernel-side nonpaged PCM ring buffer。
- Driver control device：

```text
\\.\SyncTranslateVirtualAudioControl
```

- IOCTL：

```text
IOCTL_SYNCTRANSLATE_AUDIO_WRITE_PCM
IOCTL_SYNCTRANSLATE_AUDIO_FLUSH
IOCTL_SYNCTRANSLATE_AUDIO_GET_STATS
```

- Python writer：

```text
app/infra/audio/synctranslate_driver_client.py
```

- VM smoke tool：

```text
tools/runtime_smoke/virtual_audio_driver_ioctl_smoke.py
```

- Driver Verifier 設定 script：

```text
drivers/synctranslate_virtual_audio/scripts/setup_driver_verifier.ps1
```

- VM 完整驗證 orchestration：

```text
drivers/synctranslate_virtual_audio/scripts/vm_smoke_sequence.ps1
```

目前驗證狀態：

- VM 可 build driver package。
- MSI 可在無 `devcon.exe` 的主機上透過 SetupAPI fallback 建立並 bind root device。
- 主機安裝驗證已確認 virtual speaker / virtual microphone endpoint 會出現在 Windows **Audio inputs and outputs**。
- `verify_driver_format.ps1` 已確認 endpoint device format 為 48000Hz PCM16 2ch。
- 仍需在 disposable VM / lab 環境長時間執行 Driver Verifier、真實通訊軟體與多版本 Windows smoke test。

## 重要安全規則

過去舊版 driver 曾造成 Windows stop screen，失敗模組為：

```text
tabletaudiosample.sys
```

請勿重新啟用舊危險路徑：

```c
#define SYNC_BRIDGE_KERNEL_ENABLED 0
#define SYNC_TTS_INJECTION_ENABLED 0
```

這兩個 flag 必須保持 `0`。

目前允許的安全路徑是：

```c
#define SYNC_DRIVER_PCM_RING_ENABLED 1
```

禁止事項：

- 不要在 DPC/audio timer/capture callback 裡開 named pipe。
- 不要在 DPC/audio timer/capture callback 裡 map user shared memory。
- 不要在 DPC/audio timer/capture callback 裡等待 named event。
- 不要直接在開發主機安裝未經 VM 驗證的 driver。

## 目錄說明

```text
drivers/synctranslate_virtual_audio/
  driver_contract.md
  README.md
  installer/
  overlay/
  scripts/
```

重點檔案：

```text
overlay/EndpointsCommon/minwavertstream.cpp
overlay/EndpointsCommon/synctranslate_pcm_ring.h
overlay/EndpointsCommon/synctranslate_pcm_ring.cpp
overlay/synctranslate_control.h
overlay/synctranslate_control.cpp
scripts/apply_sysvad_overlay.ps1
scripts/build_driver_poc.ps1
scripts/package_driver_msi.ps1
scripts/preflight_driver_install.ps1
scripts/install_driver_package.ps1
scripts/verify_driver_install.ps1
scripts/uninstall_driver_package.ps1
scripts/enable_test_mode.ps1
scripts/setup_driver_verifier.ps1
scripts/vm_smoke_sequence.ps1
```

## Build 前置需求

需要：

- Visual Studio 2022 C++ build tools
- Windows SDK
- Windows Driver Kit
- WiX command-line tool

檢查環境：

```powershell
powershell -ExecutionPolicy Bypass -File drivers/synctranslate_virtual_audio/scripts/check_prereqs.ps1
```

## Build driver package

```powershell
powershell -ExecutionPolicy Bypass -File drivers/synctranslate_virtual_audio/scripts/build_driver_poc.ps1
```

輸出：

```text
artifacts/driver/synctranslate_virtual_audio/package
```

## 重包 MSI

目前 MSI 版本為 `2.1.9`。

```powershell
powershell -ExecutionPolicy Bypass -File drivers/synctranslate_virtual_audio/scripts/package_driver_msi.ps1 `
  -PackageDir artifacts/driver/synctranslate_virtual_audio/package `
  -CertificatePath artifacts/driver/synctranslate_virtual_audio/SyncTranslateVirtualAudioTest.cer `
  -OutputMsi artifacts/driver/synctranslate_virtual_audio/SyncTranslateVirtualAudioDriver.msi `
  -ProductVersion 2.1.9
```

輸出：

```text
artifacts/driver/synctranslate_virtual_audio/SyncTranslateVirtualAudioDriver.msi
```

## Preflight

```powershell
powershell -ExecutionPolicy Bypass -File drivers/synctranslate_virtual_audio/scripts/preflight_driver_install.ps1 `
  -PackageDir artifacts/driver/synctranslate_virtual_audio/package
```

目前在一般主機上可能會看到 blocking：

```text
administrator
cat_signature:sysvad.cat
test_signing_enabled
```

這表示：

- 不是 elevated PowerShell。
- test certificate 尚未被信任。
- Windows Test Mode 尚未啟用。

這些是安裝環境 gate，不代表 driver build 失敗。

## 安裝策略

正式測試建議先在 disposable Windows VM 上完成：啟用 Test Mode、雙擊 MSI、確認 `SyncTranslate Virtual Speaker` / `SyncTranslate Virtual Microphone` 出現，再做 smoke test。

## VM 驗證流程

在 disposable Windows VM 中（elevated PowerShell）：

```powershell
# Step 1：啟用 Test Mode（需重啟）
powershell -ExecutionPolicy Bypass -File drivers/synctranslate_virtual_audio/scripts/enable_test_mode.ps1

# Step 2：重啟 VM 後，執行完整 smoke sequence
powershell -ExecutionPolicy Bypass -File drivers/synctranslate_virtual_audio/scripts/vm_smoke_sequence.ps1 -Phase all
```

`vm_smoke_sequence.ps1 -Phase all` 會依序做 preflight、install、verify、ioctl、reboot gate、verifier 與 uninstall。

結果 JSON 寫入 `logs/vm_smoke_sequence/`。

只有上述全部通過後，才考慮主機安裝。
## AudioEndpoint enumeration note

The one-click MSI runs `install_driver_package.ps1`, creates a clean
`Root\SyncTranslateVirtualAudio` device, removes stale SyncTranslate `ROOT\MEDIA`
instances, and installs the base virtual audio endpoint package. If `devcon.exe`
is unavailable, the MSI uses `create_root_device.ps1` as a SetupAPI fallback and
then runs `pnputil /add-driver ... /install` again to bind the staged driver.
APO/extension INFs are optional and are not installed by default for the product
path.

After install, reboot Windows so AudioEndpointBuilder refreshes the MMDevice
default format. A healthy install shows exactly one healthy SyncTranslate MEDIA
device plus two present AudioEndpoint devices:

- `Speaker (SyncTranslate Virtual Audio Device)` or localized `喇叭 (...)`
- `SyncTranslate Virtual Microphone (SyncTranslate Virtual Audio Device)`

If Device Manager shows only `SyncTranslate Virtual Audio Device` under
**Sound, video and game controllers**, but no SyncTranslate devices under
**Audio inputs and outputs**, reinstall with the current MSI and reboot the VM.

`verify_driver_format.ps1` validates `PKEY_AudioEngine_DeviceFormat` as
48000Hz PCM16 2ch. WASAPI shared-mode mix can still appear as 48k float32; that
is Windows Audio Engine behavior and is recorded separately in validation JSON.
