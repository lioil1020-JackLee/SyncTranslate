# SyncTranslate v2 Windows 驗收清單

本文件用於 Windows 實機或 Hyper-V VM 的產品化驗收。請先在 repo root 執行命令，例如：

```powershell
Set-Location E:\py\SyncTranslate
```

## A. Meeting Mode 驗收

- 不安裝 SyncTranslate virtual driver 也能啟動 meeting mode。
- `system input` 可從麥克風或任意 Windows input device 收音。
- `system output loopback` 可從喇叭、耳機或任意 Windows output device 做 WASAPI loopback 收音。
- ASR 會產生原文字幕。
- LLM 會產生翻譯字幕。
- TTS off 時字幕與翻譯記錄仍會更新。
- 不要求 bridge ready。
- 不送任何音訊到 SyncTranslate Virtual Microphone。

建議命令：

```powershell
uv run python main.py --check
uv run python .\tools\validation\preflight_release_check.py --mode meeting
uv run python .\tools\validation\audio_smoke_test.py meeting-input --duration 3 --wav .\downloads\validation\meeting_input.wav
uv run python .\tools\validation\audio_smoke_test.py meeting-loopback --duration 3 --wav .\downloads\validation\meeting_loopback.wav
```

## B. Dialogue Mode 驗收

- 已安裝 SyncTranslate virtual audio driver。
- 通訊軟體輸出選 `SyncTranslate Virtual Speaker`。
- 通訊軟體輸入選 `SyncTranslate Virtual Microphone`。
- remote -> local TTS 正常送到本地喇叭。
- local -> remote TTS 正常送到 virtual microphone。
- remote voice = `none` 時，remote -> local 走 direct passthrough。
- local voice = `none` 時，local -> remote 走 direct passthrough。
- direct passthrough 不產生字幕、不翻譯、不寫 transcript。
- 聲音無機械音、無明顯爆音，延遲可接受。

建議命令：

```powershell
powershell -ExecutionPolicy Bypass -File .\drivers\synctranslate_virtual_audio\scripts\verify_driver_format.ps1 -JsonOutput .\downloads\validation\driver_format.json
uv run python .\tools\validation\preflight_release_check.py --mode dialogue
uv run python .\tools\validation\validate_windows_audio_runtime.py --no-capture-probe --no-bridge-probe
uv run python .\tools\validation\audio_smoke_test.py dialogue-passthrough --duration 0.1
```

## C. Driver / Endpoint 驗收

健康的安裝狀態應看到：

- **Sound, video and game controllers**: `SyncTranslate Virtual Audio Device`，沒有驚嘆號。
- **Audio inputs and outputs**: `SyncTranslate Virtual Speaker`。
- **Audio inputs and outputs**: `SyncTranslate Virtual Microphone`。

如果只看到 `Sound, video and game controllers` 裡的 root device，但沒有 Audio inputs and outputs endpoint，dialogue mode 尚未通過驗收。

目前開發 MSI 路徑：

```text
artifacts/driver/synctranslate_virtual_audio/SyncTranslateVirtualAudioDriver.msi
```

測試簽章 driver 僅供開發 VM 或實驗室使用。正式給一般使用者的一鍵安裝 MSI 需要 production driver signing / WHQL 或等價 Microsoft 認可簽章流程，不能要求一般使用者手動開 Test Mode。

## D. Format 驗收

- Windows / driver / bridge external boundary 是 `48000Hz / PCM16 / 2ch / interleaved stereo`。
- app internal audio 是 float32 `AudioFrame`，保留 sample rate、channels、source metadata。
- ASR branch 是 `16000Hz mono float32`。
- passthrough branch 保留 stereo，只有在 sink/bridge/driver boundary 才轉成 2ch PCM16。

`verify_driver_format.ps1` 驗證 `PKEY_AudioEngine_DeviceFormat` 作為 v2 driver boundary。WASAPI shared-mode mix 可能顯示 48 kHz float32，這是 Windows audio engine 的 shared mix 格式，只作診斷資訊，不代表 driver boundary 失敗。

## E. UI 驗收

- ASR 語言選單只有 `zh-TW`、`en`、`ja`、`ko`、`th`，沒有 `auto`。
- meeting/dialogue 分區清楚。
- meeting mode 隱藏或禁用 virtual mic、remote channel、TTS voice、direct passthrough 控制。
- dialogue mode 顯示 remote/local 雙通道。
- running 時可切換 ASR language、translation target、TTS voice。
- running 時切 session mode 採一致策略：要求停止後切換。

## F. Release / Packaging 驗收

- `downloads/validation/`、`logs/`、cache、runtimes/model 大檔不提交 Git。
- release zip 保留 `runtimes/shared`、`runtimes/faster_whisper`、ASR model、LLM GGUF、audio bridge。
- no-driver Windows 電腦可以開啟 meeting mode。
- driver missing 時 dialogue mode 顯示清楚警告或 disabled，不 crash。
- diagnostics bundle 可匯出並遮罩敏感資訊。

建議命令：

```powershell
python -m compileall -q app tests tools
uv run pytest -q
uv run python main.py --check
uv run python .\tools\validation\preflight_release_check.py --mode meeting
uv run python .\tools\validation\preflight_release_check.py --mode dialogue
uv run python .\tools\validation\export_diagnostics_bundle.py
```

## G. 目前 RC 狀態

- Driver package 可在 Hyper-V VM 以 WDK build。
- 本機可用 WiX 包裝 MSI。
- 開發 MSI 已可一鍵安裝並建立 render/capture endpoints。
- endpoint device format 已可驗證為 `48000Hz / PCM16 / 2ch`。
- 本機無 WDK 時，build tools 顯示 WARN 是正確行為；meeting mode 不受影響。
- production signing / WHQL 與多版本 Windows + 通訊軟體驗收仍需人工實機完成。
