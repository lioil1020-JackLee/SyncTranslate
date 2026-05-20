# SyncTranslate

## 重要安全狀態

目前 SyncTranslate 自製 Windows 音訊 driver 僅可在一次性 Windows VM / lab 環境測試，不建議安裝到日常使用的主機。曾觀察到 `tabletaudiosample.sys` 觸發 `SYSTEM_THREAD_EXCEPTION_NOT_HANDLED (0x7E)` 藍畫面，因此 MSI 已改為只封裝檔案，不會自動安裝 driver 或要求重開機；`install_driver_package.ps1` 也會預設阻擋主機安裝，除非明確傳入 `-AllowHostInstall`。

**Windows 本地 AI 通話翻譯系統** — 透過自研虛擬音訊設備實現無需 Voicemeeter、無需外接硬體的全雙向通話翻譯。

> 目前版本定位為個人自用 / GitHub 朋友測試版。虛擬音訊 driver 是 test-signed，因此需要 Windows Test Mode；不需要購買 EV 憑證、WHQL 或 Microsoft attestation signing。若要給一般使用者無痛安裝，未來才需要正式 driver 簽章。

SyncTranslate 提供 `SyncTranslate Virtual Speaker` 與 `SyncTranslate Virtual Microphone`，使 Zoom、Teams、Google Meet、LINE、WhatsApp、Discord 等任意通訊軟體的音訊流能直接接入本地 ASR、翻譯與 TTS 管線，實現 full-duplex 實時字幕與雙向翻譯。

**技術棧**：
- ASR：faster-whisper/CTranslate2（支援中文自動識別、low-latency 與 high-accuracy 模式）
- 翻譯：本地 llama.cpp in-process GGUF 模型（不依賴外部 API）
- TTS：Edge TTS（可配置為 passthrough 本地輸出）
- GUI：PySide6 + 實時字幕與診斷面板
- 驅動：SyncTranslate Audio Driver（test-signed）

![即時字幕](image/即時字幕.png)

![設定](image/設定.png)

## 快速開始

### 前提條件
- Windows 10/11（64-bit）
- 已安裝 SyncTranslate Audio Driver（test-signed，自用版需 Windows Test Mode；見下方「驅動安裝」）
- Python 3.12 / uv（用於開發環境）

### 驅動安裝（首次只需一次）
```powershell
# 1. 以系統管理員身份開啟 PowerShell
# 2. 啟用 Windows Test Mode（需重新開機一次）
Set-Location E:\py\SyncTranslate
.\drivers\synctranslate_virtual_audio\scripts\enable_test_mode.ps1
# 重新開機後...

# 3. 安裝 MSI（需要 Administrator 權限）
msiexec /i "artifacts/driver/synctranslate_virtual_audio/SyncTranslateVirtualAudioDriver.msi"
```

安裝完成後，可在「聲音設定」或「音訊設備管理員」中看到 SyncTranslate 虛擬端點。不同 Windows 語系可能顯示為 `SyncTranslate Virtual Speaker`，或類似 `喇叭 (SyncTranslate Virtual Audio Device)`；麥克風通常顯示為 `SyncTranslate Virtual Microphone`。

### 首次運行應用
```powershell
# 開發環境
uv sync
uv run python main.py

# 檢查配置與 runtime 就緒
uv run python main.py --check
```

首次運行時，若 bridge process（`sync_audio_bridge.exe`）未啟動，App 會嘗試自動啟動；若需手動啟動或重啟 bridge，可由 UI 設定面板觸發。

### 基本使用流程
1. 在 Zoom/Teams/Meet 等通訊軟體中，將麥克風輸入設為 `SyncTranslate Virtual Microphone`
2. 將揚聲器/耳機設為 `SyncTranslate Virtual Speaker`
3. 在 SyncTranslate UI 中選擇：
   - 遠端語言（會議/遠端音訊語言）
   - 本地語言（麥克風輸入語言）
   - 翻譯方向（例：EN ↔ ZH-TW）
4. 開始通話，雙路字幕與翻譯會實時顯示在 SyncTranslate UI 上

## 核心功能

### 通話翻譯
- **雙路字幕**：`remote` 對應會議/遠端音訊，`local` 對應本地麥克風輸入
- **自動語言識別**：中文/非中文自動切換識別引擎，支援中英混合對話
- **實時翻譯**：本地 llama.cpp GGUF 模型直譯，無外部 API 依賴
- **全雙向通話翻譯**：透過 SyncTranslate Virtual Speaker/Microphone 實現 full-duplex

### ASR/翻譯管線
- **ASR v2**：`SourceRuntimeV2` 管理音訊前端、VAD endpoint、partial/final decode、queue 背壓與 hallucination 過濾
- **語言路由**：
  - 中文家族（中文 ASR）→ `large-v3-turbo`（預設）
  - 非中文 + `auto` 模式 → `large-v3-turbo`（預設）
  - 備選：`belle-zh-ct2`（可於設定 UI 選擇）
- **音訊前端**：支援 AGC/high-pass，中文模式停用頻譜增強器以減少誤識別
- **背壓保護**：queue 預設 256/channel，worker 在壓力下合併更多 chunk、暫停 partial、延後昂貴 decode，避免掉音
- **字幕緩衝**：去除相鄰 final 的重疊前綴，翻譯字幕不顯示 `[final]` 標記

### 準確度控制
- `low_latency`：關閉 final rescue 與中文 fallback，優先降低延遲
- `balanced`：低信心 final 才重辨識（預設模式）
- `high_accuracy`：允許更積極的 final rescue 與中文 fallback

### 軟體相容性
支援以下通訊軟體的自動音訊路由（通過 matrix 測試）：
- Zoom
- Microsoft Teams
- Google Meet
- LINE
- WhatsApp
- Discord

### 虛擬音訊驅動
- **SyncTranslate Virtual Speaker**：接收 SyncTranslate 的本地翻譯輸出
- **SyncTranslate Virtual Microphone**：向通訊軟體呈現 SyncTranslate 的虛擬麥克風
- **Bridge Process**：原生 C++ audio bridge（`sync_audio_bridge.exe`），負責 48kHz 48-bit ring buffer 與 IPC
- **無需第三方虛擬音訊軟體**：不依賴 Voicemeeter、VB-CABLE 或其他虛擬音訊工具

### 診斷與監測
- **UI 診斷面板**：即時顯示 ASR queue、dropped chunks、當前模型/profile/enhancement、capture 狀態
- **詳細報告**：session report、healthcheck、runtime smoke test、ASR benchmark
- **Soak 測試**：支援 120s~7200s 可配置測試，驗證 2hr 長時間穩定性
- **度量追蹤**：virtual_mic_dropped_frames、sink_backpressure_flushes、sink_dropped_frames 等細粒度度量

## 專案結構

```text
main.py                         # GUI / CLI 入口
app/bootstrap/                  # runtime path、app factory、DI bundle
app/application/                # AudioRouter、session、diagnostics、transcript、translation dispatcher
app/domain/                     # domain model、runtime state、常數
app/infra/asr/                  # ASR v2 backend、worker、endpointing、frontend、language profiles
app/infra/audio/                # sounddevice capture/playback、device registry、routing
app/infra/config/               # AppConfig schema、YAML parser、migration、serialization
app/infra/translation/          # local llama translation provider、prompt、parser、stitcher
app/infra/tts/                  # TTS queue、voice policy、Edge TTS integration
app/local_ai/                   # local AI healthcheck 與 subprocess worker
app/ui/                         # PySide6 UI pages
tests/                          # pytest 測試
tools/asr_benchmark/            # streaming ASR benchmark 與報表
tools/runtime_setup/            # runtime 下載、搬移、onedir 打包輔助
tools/runtime_smoke/            # runtime smoke tests
tools/youtube_srt/              # YouTube 字幕解析工具
```

## 安裝

建議使用 Python 3.12 與 `uv`。

```powershell
uv sync
```

準備外部 AI runtime：

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\runtime_setup\prepare_external_runtimes.ps1
```

這會準備 `runtimes/shared`、`runtimes/faster_whisper` 與相關模型/runtime 檔案。實際可執行 runtime 會被 PyInstaller onedir 使用。

## 執行

開發環境：

```powershell
uv run python main.py
```

指定設定：

```powershell
uv run python main.py --config config.yaml
```

檢查設定與 runtime：

```powershell
uv run python main.py --check
```

onedir runtime 直接執行原始入口：

```powershell
.\dist\SyncTranslate-onedir\runtimes\shared\Scripts\python.exe .\main.py
```

## 設定

主要設定檔為 `config.yaml`，範本為 `config.example.yaml`。設定會被解析成 `app/infra/config/schema.py` 的 `AppConfig`。

常用 runtime key：

- `runtime.local_asr_language` / `runtime.remote_asr_language`
- `runtime.local_translation_target` / `runtime.remote_translation_target`
- `runtime.local_tts_voice` / `runtime.remote_tts_voice`
- `runtime.asr_profile_local` / `runtime.asr_profile_remote`
- `runtime.asr_queue_maxsize_local` / `runtime.asr_queue_maxsize_remote`
- `runtime.asr_frontend_enabled`
- `runtime.asr_enhancement_enabled`
- `runtime.display_partial_strategy`
- `runtime.max_pipeline_latency_ms`
- `runtime.asr_accuracy_mode`
- `runtime.asr_final_rescue_enabled`
- `runtime.asr_chinese_fallback_enabled`

ASR v3 準確度控制採用「partial 快速顯示、final 低信心才救援」策略：
- `low_latency`：關閉 final rescue 與中文 fallback，優先降低延遲。
- `balanced`：低信心 final 才重辨識，為預設模式。
- `high_accuracy`：允許更積極的 final rescue 與中文 fallback。

診斷摘要會顯示 queue、dropped chunks、實際模型、fallback 模型、final priority、rescue 次數與 fallback 次數。

ASR regression corpus 優先使用本機 benchmark 素材建立：

```powershell
uv run --with datasets==2.18.0 --with librosa --with soundfile --with numpy python tools/asr_benchmark/prepare_local_benchmark_corpus.py --force
uv run python tools/asr_benchmark/run_regression_corpus.py --manifest downloads/asr_regression_local/manifest.yaml --models turbo --modes meeting,dialogue --quick
```

此工具會建立 35 筆 curated stable baseline：20 筆中文、10 筆英文、5 筆中英混合。它保留雙模式穩定通過的本機 benchmark 段落，並補入已實測穩定的 FLEURS/ASCEND 樣本；音檔、字幕切片與報告位於 `downloads/asr_regression_local/` 與 `downloads/benchmark_results/`，不提交 Git；可提交的是工具、manifest 範本、測試與文件。

常用 audio key：

- `audio.meeting_in`
- `audio.microphone_in`
- `audio.speaker_out`
- `audio.meeting_out`

SyncTranslate 不再提供路由級增益/音量控制，請直接使用 Windows 系統音量與通訊軟體音量控制。

完整說明見 [設定說明](docs/設定說明.md)。

## 虛擬音訊驅動安裝

SyncTranslate 需要 Windows 將以下兩個音訊端點提供給應用程式：

```text
SyncTranslate Virtual Speaker   ← 本地翻譯輸出
SyncTranslate Virtual Microphone ← 虛擬麥克風（通訊軟體看得到）
```

這兩個端點**不能**只靠 App 一般啟動建立，必須安裝 Windows audio driver。SyncTranslate 提供 test-signed driver 與 MSI 安裝程式。

目前不做正式簽章，因此這是自用/內測路線：

- 需要 Administrator 權限安裝 driver。
- 需要 Windows Test Mode。
- Secure Boot 可能需要關閉。
- 未簽章 App/Driver 可能被 SmartScreen 或防毒提醒。
- 公司電腦、受管裝置或不願意開 Test Mode 的朋友不適合使用目前版本。

### 驅動組件

驅動代碼位於：
```text
drivers/synctranslate_virtual_audio/
```

預建 MSI 位於：
```text
artifacts/driver/synctranslate_virtual_audio/SyncTranslateVirtualAudioDriver.msi
```

### 安裝步驟

**重要**：需使用 **Administrator** 權限。

#### 步驟 1：啟用 Windows Test Mode

test-signed driver 需要 Test Mode 才能載入。使用 Administrator PowerShell 執行：

```powershell
Set-Location E:\py\SyncTranslate
.\drivers\synctranslate_virtual_audio\scripts\enable_test_mode.ps1
```

**系統會重新開機一次**。重新開機後 BIOS 開機畫面會看到「底部有 Test Mode 標籤」。

**安裝注意事項**：
- 若出現「值受安全開機原則保護」→ 需進 BIOS 關閉 Secure Boot
- 若有啟用 BitLocker → 先暫停 BitLocker 或保存修復金鑰

#### 步驟 2：安裝 MSI

Test Mode 啟用且重新開機後，以 Administrator 身份執行：

```powershell
msiexec /i "artifacts/driver/synctranslate_virtual_audio/SyncTranslateVirtualAudioDriver.msi"
```

#### 步驟 3：驗證安裝

重新開機後，開啟「聲音設定」或「音訊設備管理員」，應能看到：
- `SyncTranslate Virtual Speaker`，或類似 `喇叭 (SyncTranslate Virtual Audio Device)` 的 speaker endpoint
- `SyncTranslate Virtual Microphone`

### 診斷

若驅動安裝失敗或 Secure Boot 問題，使用診斷腳本：

```powershell
Set-Location E:\py\SyncTranslate
.\drivers\synctranslate_virtual_audio\scripts\diagnose_test_signing.ps1
```

### 開發者：自行編譯驅動

若要修改驅動代碼或自行編譯，工作區在：

```text
drivers/synctranslate_virtual_audio/
```

檢查建置環境：

```powershell
drivers/synctranslate_virtual_audio/scripts/check_prereqs.ps1
```

本專案可自行安裝 local .NET SDK / WiX 5，不需要 Administrator：

```powershell
drivers/synctranslate_virtual_audio/scripts/install_build_prereqs.ps1 -InstallDotNetSdk -InstallWix
```

WDK、DriverKit build tools 需要系統層級安裝：

```powershell
drivers/synctranslate_virtual_audio/scripts/install_build_prereqs.ps1 -InstallWdk -LaunchElevatedWdkInstaller
```

工具齊全後產出 MSI：

```powershell
drivers/synctranslate_virtual_audio/scripts/build_driver_msi.ps1
```

安裝 test-signed MSI 前需先開啟 Windows Test Mode 並重新開機。Administrator PowerShell 通常會從 `C:\Windows\System32` 開啟，請先切到專案根目錄再執行相對路徑：
```powershell
Set-Location E:\py\SyncTranslate
.\drivers\synctranslate_virtual_audio\scripts\enable_test_mode.ps1
```

也可以直接使用完整路徑：
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "E:\py\SyncTranslate\drivers\synctranslate_virtual_audio\scripts\enable_test_mode.ps1"
```

若畫面出現「值受安全開機原則保護，因此無法修改或刪除」，代表 Secure Boot 阻擋 Test Mode。請先進 BIOS/UEFI 關閉 Secure Boot；若有啟用 BitLocker，請先暫停 BitLocker 或確認已保存修復金鑰，再重新執行上面的指令並重開機。這是目前免費自用版的限制。

若 BIOS 畫面看起來已關閉，但 Windows 仍報安全開機原則保護，請用系統管理員 PowerShell 跑診斷：
```powershell
Set-Location E:\py\SyncTranslate
.\drivers\synctranslate_virtual_audio\scripts\diagnose_test_signing.ps1
```

以診斷輸出的 `UEFISecureBootEnabled` / `Confirm-SecureBootUEFI` 為準；只要 Windows 仍回報啟用，`TESTSIGNING` 就會被擋。

重新開機後再執行 MSI。若安裝失敗，安裝腳本會寫入：
```text
C:\ProgramData\SyncTranslate\Logs\driver-install.log
```

## 測試

基本檢查：

```powershell
uv run python -m compileall -q app main.py tools tests
uv run pytest
```

ASR v2 重點測試：

```powershell
uv run pytest tests/test_asr_streaming_and_profiles.py tests/test_asr_frontend_v2.py tests/test_asr_v2_endpointing.py
```

runtime smoke：

```powershell
uv run python .\tools\runtime_smoke\run_runtime_smoke.py --config config.yaml
```

onedir 打包後 smoke：

```powershell
uv run python .\tools\runtime_smoke\run_runtime_smoke.py --config config.yaml --packaged-onedir .\dist\SyncTranslate-onedir
```

ASR benchmark：

```powershell
uv run python tools/asr_benchmark/run_multi_benchmark.py --only-lang zh --output-dir downloads/benchmark_results/manual_zh
```

ASR regression corpus baseline：

```powershell
uv run --with datasets==2.18.0 --with librosa --with soundfile --with numpy python tools/asr_benchmark/prepare_local_benchmark_corpus.py --force
uv run python tools/asr_benchmark/run_regression_corpus.py --manifest downloads/asr_regression_local/manifest.yaml --models turbo --modes meeting,dialogue --speed 16 --quick --output-dir downloads/benchmark_results/asr_regression_local_baseline
```

公開資料集 fallback corpus：
```powershell
uv run --with datasets==2.18.0 --with librosa --with soundfile --with numpy python tools/asr_benchmark/prepare_regression_corpus.py --force
uv run python tools/asr_benchmark/run_regression_corpus.py --manifest downloads/asr_regression/manifest.yaml --models turbo --modes meeting,dialogue --speed 16 --quick --output-dir downloads/benchmark_results/asr_regression_baseline
```

完整說明見 [測試說明](docs/測試說明.md)。

## 打包

準備 runtime：

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\runtime_setup\prepare_external_runtimes.ps1
```

建立 PyInstaller onedir：

```powershell
uv run pyinstaller SyncTranslate-onedir.spec --noconfirm
powershell -ExecutionPolicy Bypass -File .\tools\runtime_setup\relocate_ai_runtime_artifacts.ps1
```

壓縮 onedir：

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\runtime_setup\package_onedir.ps1 -Source .\dist\SyncTranslate-onedir -Output .\dist\SyncTranslate-onedir-windows.zip
```

## 文件

- [架構說明](docs/架構說明.md)
- [設定說明](docs/設定說明.md)
- [測試說明](docs/測試說明.md)
- [音訊裝置建議配置](docs/音訊裝置建議配置.md)
