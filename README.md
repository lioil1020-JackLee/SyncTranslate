# SyncTranslate

## 畫面截圖

![即時字幕畫面](image/即時字幕.png)

![設定畫面](image/設定.png)

SyncTranslate 是一個 Windows 桌面即時雙向字幕、翻譯與語音輸出工具。它可以同時處理會議遠端音訊與本機麥克風，將兩邊語音轉成字幕，必要時使用本地 LLM 翻譯，再透過 TTS 或音訊旁路輸出到指定裝置。

目前專案以 ASR v2、本地 llama.cpp in-process 翻譯、Edge TTS 與 PySide6 UI 為主線。舊的 ASR worker、未接入 UI 的 controller 抽離檔、重複 metrics/dispatcher shim 已移除，程式碼結構以實際 runtime 路徑為準。

## 功能

- 雙通道即時字幕：`remote` 代表會議/系統音訊，`local` 代表本機麥克風。
- ASR 語言分流：中文使用中文 profile，非中文與 auto 使用 faster-whisper profile。
- 本地翻譯：透過 `llama-cpp-python` 載入 GGUF 模型，不依賴遠端翻譯 API。
- TTS / 旁路 / 只顯示字幕：依每個通道的輸出模式決定。
- 音訊路由：支援麥克風、會議音訊、喇叭、虛擬裝置與 Windows endpoint volume 控制。
- 診斷與匯出：包含 healthcheck、session report、diagnostics export。

## 專案結構

```text
main.py                         # GUI / CLI 入口
app/bootstrap/                  # 啟動、外部 runtime、DI container
app/application/                # 管線協調、字幕、翻譯 dispatch、設定服務、診斷匯出
app/domain/                     # 共用模型、狀態、常數、文字工具
app/infra/asr/                  # ASR v2 runtime、endpointing、frontend、faster-whisper adapter
app/infra/audio/                # 音訊擷取、播放、裝置列舉、音量控制
app/infra/config/               # YAML schema、載入、儲存、遷移
app/infra/translation/          # 本地 LLM 翻譯 provider、prompt、parser、stitcher
app/infra/tts/                  # Edge TTS、播放 queue、voice policy
app/local_ai/                   # 本地 AI healthcheck 與 subprocess worker
app/ui/                         # PySide6 主視窗與頁面
tests/                          # 單元與可控整合測試
tools/asr_benchmark/            # ASR benchmark 與報表工具
tools/runtime_setup/            # 外部 runtime / 模型 / 打包輔助工具
tools/runtime_smoke/            # 實機 runtime smoke test
tools/youtube_srt/              # YouTube 字幕比對工具
```

## 安裝

建議使用 Python 3.11 與 `uv`。

```powershell
uv sync
```

準備外部 AI runtime 與模型：

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\runtime_setup\prepare_external_runtimes.ps1
```

這個步驟會準備 `runtimes/shared`、`runtimes/faster_whisper` 與必要模型。外部 runtime 不由 `pyproject.toml` 直接管理，目的是讓 GUI app、打包與 AI 套件隔離。

## 啟動

```powershell
uv run python main.py
```

指定設定檔：

```powershell
uv run python main.py --config config.yaml
```

檢查設定與 runtime：

```powershell
uv run python main.py --check
```

## 設定

主要設定檔是 `config.yaml`，範本是 `config.example.yaml`。啟動時會載入 YAML 並轉成 `AppConfig`；UI 修改會先套用到記憶體，按下儲存後才寫回設定檔。

重點欄位：

- `runtime.local_asr_language` / `runtime.remote_asr_language`
- `language.local_target` / `language.meeting_target`
- `runtime.local_tts_voice` / `runtime.remote_tts_voice`
- `audio.*_device`
- `asr_profiles.*`
- `llm_profiles.*`
- `tts.*`

更多細節見 [docs/設定說明.md](docs/設定說明.md)。

## 測試

快速檢查：

```powershell
uv run python -m compileall -q app main.py tools tests
uv run pytest
```

只跑核心測試：

```powershell
uv run pytest tests/test_audio_router_core.py tests/test_translation_stitcher.py tests/test_asr_v2_endpointing.py
```

實機 runtime smoke：

```powershell
uv run python .\tools\runtime_smoke\run_runtime_smoke.py --config config.yaml
```

更多測試策略見 [docs/測試說明.md](docs/測試說明.md)。

## 打包

先準備 runtime，再執行 PyInstaller spec：

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\runtime_setup\prepare_external_runtimes.ps1
uv run pyinstaller SyncTranslate-onedir.spec --noconfirm
powershell -ExecutionPolicy Bypass -File .\tools\runtime_setup\relocate_ai_runtime_artifacts.ps1
```

壓縮 onedir：

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\runtime_setup\package_onedir.ps1 -Source .\dist\SyncTranslate-onedir -Output .\dist\SyncTranslate-onedir-windows.zip
```

## 文件

- [docs/架構說明.md](docs/架構說明.md)
- [docs/設定說明.md](docs/設定說明.md)
- [docs/測試說明.md](docs/測試說明.md)
- [docs/音訊裝置建議配置.md](docs/音訊裝置建議配置.md)

## 維護原則

- ASR runtime 以 `app/infra/asr/worker_v2.py`、`endpointing_v2.py`、`backend_v2.py` 為主。
- UI 行為以 `app/ui/main_window.py` 與 `app/ui/pages/` 為準。
- 管線協調以 `app/application/audio_router.py` 為主。
- 不再保留只為舊 API 或未接入抽離設計存在的 shim。
- 文件只描述目前可執行的架構，不保留一次性 PR summary 或流水帳式歷史文件。
