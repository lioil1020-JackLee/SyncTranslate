# SyncTranslate Virtual Audio Driver

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

尚未完成（需在 disposable VM 中執行）：

- VM 安裝驗證。
- IOCTL sine injection 錄音驗證。
- Driver Verifier gate。
- 主機安裝驗證。

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

目前 MSI 版本為 `2.1.2`。

```powershell
powershell -ExecutionPolicy Bypass -File drivers/synctranslate_virtual_audio/scripts/package_driver_msi.ps1 `
  -PackageDir artifacts/driver/synctranslate_virtual_audio/package `
  -CertificatePath artifacts/driver/synctranslate_virtual_audio/SyncTranslateVirtualAudioTest.cer `
  -OutputMsi artifacts/driver/synctranslate_virtual_audio/SyncTranslateVirtualAudioDriver.msi `
  -ProductVersion 2.1.2
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

不要先在開發主機安裝。

正確順序：

1. 建立 disposable Windows VM。
2. 建立 VM snapshot。
3. 在 VM 啟用 Test Mode。
4. 在 VM 信任測試憑證。
5. 在 VM 安裝 MSI 或 driver package。
6. 確認裝置管理員沒有驚嘆號。
7. 確認有：
   - `SyncTranslate Virtual Speaker`
   - `SyncTranslate Virtual Microphone`
8. 重啟 VM 三次。
9. 執行 IOCTL sine injection smoke test。
10. 再做 Driver Verifier。

## VM 驗證流程

在 disposable Windows VM 中（elevated PowerShell）：

```powershell
# Step 1：啟用 Test Mode（需重啟）
powershell -ExecutionPolicy Bypass -File drivers/synctranslate_virtual_audio/scripts/enable_test_mode.ps1

# Step 2：重啟 VM 後，執行完整 smoke sequence
powershell -ExecutionPolicy Bypass -File drivers/synctranslate_virtual_audio/scripts/vm_smoke_sequence.ps1 -Phase all
```

`vm_smoke_sequence.ps1 -Phase all` 會依序執行：

1. **preflight** — 環境 gate（administrator / test signing / certificate）
2. **install** — 安裝 `SyncTranslateVirtualAudioDriver.msi`
3. **verify** — 確認 `SyncTranslate Virtual Speaker` / `SyncTranslate Virtual Microphone` 出現
4. **ioctl** — Python IOCTL smoke test（WRITE_PCM / GET_STATS / FLUSH / 錄音驗證）
5. **reboot gate** — 提示手動重啟三次（每次重啟後重跑 verify + ioctl）
6. **verifier** — 啟用 Driver Verifier，再次 smoke
7. **uninstall** — 解除安裝，確認裝置清除

結果 JSON 寫入 `logs/vm_smoke_sequence/`。

只有上述全部通過後，才考慮主機安裝。
