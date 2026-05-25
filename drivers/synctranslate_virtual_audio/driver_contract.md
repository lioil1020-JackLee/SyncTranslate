# SyncTranslate 虛擬音效驅動合約

## 產品目標

SyncTranslate virtual audio driver 的目標，是取代 Voicemeeter / VB-CABLE 在會議翻譯流程中的用途。

主要 endpoint：

- `SyncTranslate Virtual Microphone`：給 Zoom / Teams / Meet / Discord 選用的 Windows capture endpoint。

目前安全里程碑：

```text
SyncTranslate App / TTS
  -> DeviceIoControl
  -> driver NonPagedPool PCM ring buffer
  -> virtual microphone capture stream
```

後續可選里程碑：

```text
Windows app render
  -> SyncTranslate Virtual Speaker
  -> driver ring
  -> SyncTranslate Virtual Microphone
```

獨立於 App/bridge 的 kernel-only virtual cable loopback 不是目前 v2 產品主線。
目前產品主線是 App/bridge 以 protocol v2 PCM16 stereo 48k 將翻譯 TTS 或 direct passthrough 音訊送進
`SyncTranslate Virtual Microphone`，並由 healthcheck 驗證 speaker/microphone endpoint 與 driver boundary。

## 目前實作狀態

目前 driver overlay 相關檔案：

```text
overlay/EndpointsCommon/minwavertstream.cpp
overlay/EndpointsCommon/synctranslate_pcm_ring.h
overlay/EndpointsCommon/synctranslate_pcm_ring.cpp
overlay/synctranslate_control.h
overlay/synctranslate_control.cpp
```

目前 control device：

```text
Kernel name:   \Device\SyncTranslateVirtualAudioControl
User path:     \\.\SyncTranslateVirtualAudioControl
Symbolic link: \DosDevices\SyncTranslateVirtualAudioControl
```

目前 guard flags：

```c
#define SYNC_BRIDGE_KERNEL_ENABLED 0
#define SYNC_TTS_INJECTION_ENABLED 0
#define SYNC_DRIVER_PCM_RING_ENABLED 1
```

`SYNC_DRIVER_PCM_RING_ENABLED` 是目前允許使用的安全路徑。其他兩個 flag 必須維持 `0`，除非已經在 disposable VM 中完成驗證並明確更新設計。

## 已淘汰的危險 kernel 路徑

以下路徑曾經造成主機在 `tabletaudiosample.sys` stop screen / 重啟，不可隨意恢復：

- DPC/audio timer path 讀寫 user shared memory。
- DPC/audio timer path 開啟或 signal named event。
- DPC/audio timer path 呼叫 named pipe / heartbeat 邏輯。
- 舊版 kernel TTS injection global loopback buffer。

過去看過的錯誤包含：

```text
SYSTEM_THREAD_EXCEPTION_NOT_HANDLED (0x7E)
DRIVER_IRQL_NOT_LESS_OR_EQUAL (0xD1)
KMODE_EXCEPTION_NOT_HANDLED (0x1E)
```

## IOCTL 介面

定義位置：

```text
drivers/synctranslate_virtual_audio/overlay/synctranslate_control.h
```

目前 IOCTL：

```text
IOCTL_SYNCTRANSLATE_AUDIO_WRITE_PCM
IOCTL_SYNCTRANSLATE_AUDIO_FLUSH
IOCTL_SYNCTRANSLATE_AUDIO_GET_STATS
```

### WRITE_PCM

目前第一版實作輸入是 raw PCM：

```text
float32 mono samples
```

目前沒有 header。input buffer 長度必須是非零，且必須是 `sizeof(FLOAT)` 的倍數。

限制：

- sample format：`float32`
- channels：`mono`
- 主要目標 sample rate：`48000 Hz`
- app 端負責 resample、downmix、clip

Python 寫入端：

```text
app/infra/audio/synctranslate_driver_client.py
```

Python writer 目前會：

- 將輸入 audio 轉成 `float32`
- 多聲道時 downmix 成 mono
- clip 到 `[-1.0, 1.0]`
- 呼叫 `IOCTL_SYNCTRANSLATE_AUDIO_WRITE_PCM`

未來協定注意事項：

如果未來需要 multi-channel、sample-rate metadata、timestamp 或 versioning，請新增 versioned IOCTL 或 header-based IOCTL。不要直接改掉目前 raw `WRITE_PCM` ABI，否則 app/driver 版本會不相容。

### FLUSH

清空 kernel PCM ring buffer。

### GET_STATS

回傳：

```c
typedef struct _SYNCTRANSLATE_PCM_RING_STATS
{
    ULONGLONG capacityFrames;
    ULONGLONG bufferedFrames;
    ULONGLONG totalWrittenFrames;
    ULONGLONG totalReadFrames;
    ULONGLONG droppedFrames;
    ULONGLONG underrunFrames;
} SYNCTRANSLATE_PCM_RING_STATS;
```

## Kernel Ring Buffer

定義位置：

```text
overlay/EndpointsCommon/synctranslate_pcm_ring.h
overlay/EndpointsCommon/synctranslate_pcm_ring.cpp
```

目前行為：

- 使用 `ExAllocatePool2(POOL_FLAG_NON_PAGED, ...)`
- 預設容量：`48000 * 5` frames
- writer：IOCTL path
- reader：virtual microphone capture path
- synchronization：一個 `KSPIN_LOCK`
- overrun：丟棄最舊 frame，遞增 `droppedFrames`
- underrun：輸出 silence，遞增 `underrunFrames`

reader 行為：

- 若有 frame，轉換並寫入 capture DMA buffer。
- 若沒有 frame，寫入 silence。

writer 行為：

- 若有空間，append frames。
- 若已滿，先推進 read pointer 丟棄舊 frames，再 append。

## Build 產物

driver package：

```text
artifacts/driver/synctranslate_virtual_audio/package
```

目前 MSI：

```text
artifacts/driver/synctranslate_virtual_audio/SyncTranslateVirtualAudioDriver.msi
```

目前 MSI 版本：

```text
2.1.7
```

MSI 目前是使用者安裝包。正式使用時直接雙擊 MSI 即可；若需要手動驗證，先完成 Test Mode。

## 驗證 Gate

實驗 driver package 不應直接安裝在開發主機。必須先在 disposable VM 通過：

1. build 成功。
2. INF verification 通過。
3. CAT signature / test certificate setup 已確認。
4. elevated VM PowerShell 中 preflight 通過。
5. VM install 建立預期 endpoint。
6. `\\.\SyncTranslateVirtualAudioControl` 可成功開啟。
7. `WRITE_PCM` 後 `totalWrittenFrames` 增加。
8. 透過 IOCTL 注入 sine，能從 `SyncTranslate Virtual Microphone` 錄到。
9. VM 重啟三次不出現 stop screen。
10. VM uninstall 能清除所有 SyncTranslate devices / driver packages。
11. Driver Verifier 只針對 SyncTranslate driver 執行並通過 smoke workload。

## 主機規則

開發主機可以：

- build driver package
- 檢查 driver package
- 重包 MSI
- 跑 app 端測試
- 讀 crash dump / log

開發主機不應自動安裝實驗 driver。

主機安裝必須是 VM 驗證後的人為明確決策。除非操作者在看過目前 preflight 結果後，明確要求「現在安裝到主機」，否則 Codex / Copilot 不應自動執行 host install。

## 下一個工程任務

建立 VM smoke tool：

```text
tools/runtime_smoke/virtual_audio_driver_ioctl_smoke.py
```

建議檢查：

- 開啟 `\\.\SyncTranslateVirtualAudioControl`
- 呼叫 `GET_STATS`
- 用 `WRITE_PCM` 寫入 1 kHz float32 mono sine
- 確認 `totalWrittenFrames` 增加
- 從 `SyncTranslate Virtual Microphone` 錄音
- 驗證 RMS / frequency / frame count
- 呼叫 `FLUSH`
- 確認 buffered frames 下降或歸零
