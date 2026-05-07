# SyncTranslate

SyncTranslate 是一個 Windows 本地即時字幕、ASR、翻譯與 TTS runtime。專案目前以 PySide6 GUI 為主，使用 faster-whisper/CTranslate2 做 ASR，使用本地 llama.cpp in-process GGUF 模型做翻譯，並可透過 Edge TTS 或 passthrough 輸出聲音。

![即時字幕](image/即時字幕.png)

![設定](image/設定.png)

## 目前能力

- 雙路字幕：`remote` 對應會議/遠端音訊，`local` 對應本地麥克風或本機播放路由。
- ASR v2：`SourceRuntimeV2` 管理音訊前端、VAD endpoint、partial/final decode、queue 背壓與 hallucination 過濾。
- 語言路由：中文 ASR 預設走 `asr_channels.local` 的 `large-v3-turbo`，`belle-zh-ct2` 保留為設定 UI 可選備用；非中文與 `auto` 預設走 `asr_channels.remote` 的 `large-v3-turbo`。
- 中文 ASR 前端：保留 AGC/high-pass，但中文 profile 會停用頻譜增強器，降低 belle-zh-ct2 對降噪失真的誤識別。
- 翻譯：本地 `hy-mt1.5-7b.gguf` 透過 `llama-cpp-python` in-process provider 執行。
- 字幕緩衝：ASR final 會先去除相鄰 final 的重疊前綴，再進入翻譯，翻譯字幕不顯示 `[final]` 標記。
- 背壓保護：ASR queue 預設 local/remote 為 `256`，worker 在 queue 壓力下會合併更多 chunk、暫停 partial、並延後昂貴的 forced final decode，避免越忙越掉音訊。
- 診斷：UI 會顯示 ASR queue、dropped chunks、實際模型/profile/enhancement 與 local/remote capture 疑似同源狀態；也可輸出 session report、healthcheck、runtime smoke 與 ASR benchmark 結果。
- 中文 benchmark：`tools/asr_benchmark/run_zh_preset_matrix.py` 可比較 `meeting/dialogue × belle/turbo`，目前壓力測試顯示 `large-v3-turbo` 平均準確率較高且 dropped chunks 較穩定。

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

常用 audio key：

- `audio.meeting_in`
- `audio.microphone_in`
- `audio.speaker_out`
- `audio.meeting_out`
- `audio.meeting_in_gain`
- `audio.microphone_in_gain`
- `audio.speaker_out_volume`
- `audio.meeting_out_volume`

完整說明見 [設定說明](docs/設定說明.md)。

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
