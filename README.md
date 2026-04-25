# SyncTranslate

SyncTranslate 是一個以 Windows 桌面為主的即時雙向字幕 / 翻譯 / 語音輸出工具。它可以同時處理本地與遠端兩條音訊通道，將音訊送進 ASR、翻譯與 TTS，並在 UI 中即時顯示原文與譯文。

目前專案的主線已切換到 `ASR v2`。新的 ASR 架構重點是：

- 中文 ASR 自動使用 `faster-whisper + silero-vad`
- 非中文與 `auto` 自動使用 `faster-whisper`
- `none` 會直接停用該通道 ASR
- 模型採懶載入與共享 registry，避免啟動時重複初始化
- diagnostics / session report 會輸出每通道實際使用的 backend、effective device、初始化狀態
- 依 ASR 語言套用 `language_profiles.py` 的辨識參數，中文、英文、日文、韓文、泰文會使用不同 VAD / endpoint / prompt / no-speech 設定
- 即時串流路徑會使用 segment-local VAD 統計與短尾端幻聽過濾，降低空白尾音被辨識成 `You`、`Thank you`、片尾訂閱提示等文字的機率

專案內部現在只維護單一路徑 `ASR v2`。舊版執行路徑已移除，僅保留極小的相容 helper 供測試與舊介面引用。

## 目前功能

- 雙通道音訊路由
  - `remote` 與 `local` 可同時存在
  - 可對應 VoiceMeeter / 實體裝置 / 虛擬裝置
- 即時 ASR
  - 中文走 faster-whisper
  - 非中文走 faster-whisper
  - 每通道可獨立指定 ASR 語言
- 即時翻譯
  - 目前以 LM Studio 為主要本地 LLM backend
- TTS / passthrough 輸出
  - 每通道可獨立設定翻譯目標與輸出語音
- 診斷與匯出
  - runtime stats
  - session report
  - 字幕原文 / 譯文匯出

## 架構摘要

主要資料流如下：

1. `AudioCapture` 收到本地或遠端音訊
2. `AudioRouter` 將 chunk 分流到 ASR / translation / TTS
3. `ASRManagerV2` 建立每通道 v2 runtime 與 backend pair
4. `EndpointingRuntime` 執行 VAD / endpointing
5. `SourceRuntimeV2` 負責 partial / final / validator / diagnostics
6. 翻譯與 TTS 消費 final transcript
7. `MainWindow` / `LiveCaptionPage` 顯示字幕與 diagnostics

重要模組：

- `app/application/`
  - `audio_router.py`
  - `transcript_service.py`
  - `transcript_postprocessor.py`（Phase 1）
  - `asr_event_processor.py`（Phase 4）
  - `translation_dispatcher.py`（Phase 4）
  - `tts_dispatcher.py`（Phase 4）
  - `pipeline_metrics.py`（Phase 4）
- `app/domain/`
  - `glossary.py`（Phase 1）
  - `metrics.py`（Phase 1）
- `app/infra/asr/`
  - `factory.py`
  - `manager_v2.py`
  - `worker_v2.py`
  - `backend_resolution.py`
  - `backend_v2.py`
  - `endpointing_v2.py`
  - `faster_whisper_adapter.py`
  - `streaming_policy.py`（Phase 2）
  - `endpoint_profiles.py`（Phase 2）
  - `audio_pipeline/`（Phase 3）
- `app/infra/logging/`（Phase 1）
- `app/infra/translation/`
- `app/infra/tts/`
- `app/ui/`
  - `controllers/`（Phase 4）
- `tools/asr_benchmark/`（Phase 3，開發工具）

更完整內容請看：

- [docs/架構說明.md](docs/架構說明.md)
- [docs/設定說明.md](docs/設定說明.md)
- [docs/測試說明.md](docs/測試說明.md)
- [docs/ASR重構藍圖.md](docs/ASR重構藍圖.md)

## 外部 AI 運行時架構

本專案採用**外部化運行時**設計，將重量級 AI 依賴（PyTorch / faster-whisper）與主程式解耦：

### 目標

- **輕量化主包**：`SyncTranslate.exe` 與 UI 依賴獨立，不含 AI 套件（~200MB 而非 2GB+）
- **模組化管理**：兩層隔離運行時（shared CUDA、faster-whisper）
- **靈活部署**：可選離線/CUDA，支援獨立升級各 AI 套件
- **超大資產支援**：GitHub Release 自動分片超過 2GB 的打包結果

### 最終目錄結構（發行版）

```text
SyncTranslate-onedir/
  SyncTranslate.exe                 # 主應用
  config.yaml
  _internal/                        # PyInstaller 主程式依賴
    app/
    PySide6/
    ...（無 torch/faster-whisper）
  
  runtimes/                         # 外部 AI 運行時（從開發環境複製）
    shared/Lib/site-packages/       # torch / torchaudio / onnxruntime
    faster_whisper/Lib/site-packages/  # faster-whisper / ctranslate2 / tiktoken
  
  models/                           # 可選：離線模型快取
```

### 啟動時的自動掛載

應用啟動時，`app/bootstrap/external_runtime.py` 會：
1. 掃描 `runtimes/{shared,faster_whisper}/Lib/site-packages`
2. 將路徑加入 `sys.path`
3. 自動註冊 Windows DLL 目錄（torch/lib、onnxruntime/capi、nvidia CUDA bins）
4. 設定模型快取環境變數（`MODELSCOPE_CACHE` / `HF_HOME`）

這樣 ASR 初始化時能直接找到外部運行時中的套件。

### 核心程式碼改動

| 檔案 | 用途 |
|---|---|
| `app/bootstrap/external_runtime.py` | 啟動期外部運行時掛載 |
| `main.py` | 在 import `app_factory` 前呼叫 `configure_external_ai_runtime()` |
| `SyncTranslate-onedir.spec` | 排除 AI 套件，改由 relocate 腳本複製 |
| `app/infra/asr/faster_whisper_adapter.py` | 錯誤訊息導向外部運行時路徑 |
| `pyproject.toml` | 移除 AI 套件依賴（輕量化 .venv） |
| `conftest.py` | pytest 自動配置外部運行時 |

### 風險與對策

| 風險 | 對策 |
|---|---|
| 外部 runtime 缺套件 | 重跑 `prepare_external_runtimes.ps1` 或補齊對應套件 |
| CUDA DLL 未被解析 | 確認 `runtimes/shared/Lib/site-packages/torch/lib` 與 nvidia 路徑存在 |
| Release 資產超過 2GB | 工作流程自動分片為 `.part001/.part002...`，使用者可用附帶腳本重組 |

## 安裝與執行

建議使用 `uv`：

```powershell
# 1. 安裝主程式依賴（輕量級，不含 torch/faster-whisper 等）
uv sync --locked

# 2. 準備外部 AI 運行時（torch, faster-whisper 等）
powershell -ExecutionPolicy Bypass -File .\tools\runtime_setup\prepare_external_runtimes.ps1
```

啟動程式：

```powershell
uv run python .\main.py
```

執行健康檢查：

```powershell
uv run python .\main.py --check
```

執行測試（透過 `conftest.py` 自動配置外部運行時）：

```powershell
uv run pytest -q
```

執行實際入口 smoke（更接近直接開 app 後按健康檢查）：

```powershell
uv run python .\tools\runtime_smoke\run_runtime_smoke.py --config config.yaml
```

如果要打包 onedir：

```powershell
# 1. 確保外部運行時已準備
powershell -ExecutionPolicy Bypass -File .\tools\runtime_setup\prepare_external_runtimes.ps1

# 2. 安裝 build 工具
uv sync --locked --group build

# 3. 構建 onedir
uv run pyinstaller .\SyncTranslate-onedir.spec --noconfirm --clean

# 4. 複製外部運行時到 dist 根目錄
powershell -ExecutionPolicy Bypass -File .\tools\runtime_setup\relocate_ai_runtime_artifacts.ps1
```

## 設定重點

目前最重要的 runtime 設定包括：

- `runtime.asr_pipeline`：預設 `v2`
- `runtime.local_asr_language` / `runtime.remote_asr_language`：決定 ASR backend
- `runtime.asr_v2_endpointing`
- `runtime.asr_profile_local` / `runtime.asr_profile_remote`：endpoint profile（預設 `meeting_room`）
- `runtime.early_final_enabled`：預設 `false`
- `runtime.stable_partial_min_repeats`：預設 `3`
- `runtime.partial_stability_max_delta_chars`：預設 `6`
- `runtime.asr_partial_min_audio_ms`：預設 `360`
- `runtime.enable_postprocessor` / `runtime.enable_partial_stabilization`：ASR 後處理
- `runtime.glossary_enabled` / `runtime.glossary_path`：術語表套用
- `runtime.degradation_policy_enabled`：streaming 降級保護
- `runtime.enable_structured_logging`：jsonl 結構化日誌
- `runtime.asr_queue_maxsize_local` / `runtime.asr_queue_maxsize_remote`

### ASR 辨識率調校狀態（2026-04-25）

目前離線辨識已達穩定水準，主要不再調整離線 decoder；後續優化重點放在即時串流。

- 離線英文 benchmark：normalized WER 約 `0.8%`
- 離線中文 benchmark：normalized CER 約 `6.8%`
- 即時英文 benchmark：尾端幻聽過濾後 normalized accuracy 約 `99.3%`
- 即時中文長故事 benchmark：overlay / 片尾過濾與 final correction 關閉後約 `86.3%`，主要瓶頸仍是背景音與即時切段波動

使用時請優先確認 `runtime.local_asr_language` / `runtime.remote_asr_language` 與實際說話語言一致。語言鎖錯時，辨識率會明顯下降。

UI 內建三種快速模式：

- `超穩定會議字幕`
  - ASR 使用 `meeting_room`，LLM / TTS 也偏穩定；適合長句字幕、會議監聽、降低 partial / final 抖動
- `低延遲對話（穩定 ASR）`
  - ASR 使用 `meeting_room`，LLM / TTS 使用低延遲對話設定；建議中文雙向對話優先使用
- `低延遲雙向對話`
  - ASR 使用 `turn_taking`，偏向短句往返、提早切段、降低對話等待感；中文準確率可能低於穩定 ASR 模式

ASR 路由規則：

- `zh` / `zh-TW` / `zh-CN` / `cmn` / `yue` -> `faster_whisper_v2`
- 其他語言 -> `faster_whisper_v2`
- `auto` -> `faster_whisper_v2`
- `none` -> disabled

## Diagnostics 觀測

diagnostics / runtime stats / 匯出目前可直接看到：

- `resolved_backend`
- `device_effective`
- `endpoint_signal.pause_ms`
- `speech_started_count / soft_endpoint_count / hard_endpoint_count`
- validator / post-processor 的 `rejected_count`
- 最近一次 `last_rejection_reason`

這些欄位能直接幫助判斷是：

- endpoint 太敏感
- final 切太碎
- validator 正在擋垃圾輸出
- 或 queue / degradation 已開始影響品質

## 目前已知限制

- 若實際 `effective device` 為 `CPU`，雙通道同時中文發話時仍可能因吞吐量不足而增加延遲
- speaker diarization 目前仍屬實驗性功能，預設關閉
- `runtime.asr_v2_backend` 目前主要作為相容欄位保留；實際 backend 以通道語言解析結果為準
- `stream_worker.py` 只剩 compatibility helpers；不再承載實際 ASR 執行流程

## 開發工具

### ASR Benchmark（`tools/asr_benchmark/`）

離線 ASR 品質量測工具，不打包進發行版：

```powershell
# 跑 benchmark（輸出 jsonl）
uv run python tools/asr_benchmark/run_benchmark.py \
  --audio <音訊檔> --profile default --reference <參考文字> \
  --chunk-ms 40 --output downloads/benchmark_results/result.jsonl

# 產生報表（table / csv / json）
uv run python tools/asr_benchmark/report.py \
  --input downloads/benchmark_results/result.jsonl --format table
```

### YouTube SRT 工具（`tools/youtube_srt/`）

下載 YouTube 字幕並與 ASR 結果做 CER/WER 比對，可用於評估辨識品質：

```powershell
# 下載中文字幕參考集
uv run python tools/youtube_srt/download_chinese.py

# 下載英文字幕參考集
uv run python tools/youtube_srt/download_english.py

# 對指定 YouTube 影片進行 ASR benchmark
uv run python tools/youtube_srt/benchmark_against_yt.py \
  --url "https://www.youtube.com/watch?v=<VIDEO_ID>" \
  --source remote --lang en --profile default \
  --out-dir downloads/benchmark \
  --output downloads/benchmark_results/result.json
```

字幕下載結果存於 `downloads/chinese_srt/` 與 `downloads/english_srt/`。  
Benchmark 結果存於 `downloads/benchmark_results/`。

## 文件索引

- [docs/架構說明.md](docs/架構說明.md)
- [docs/設定說明.md](docs/設定說明.md)
- [docs/測試說明.md](docs/測試說明.md)
- [docs/音訊裝置建議配置.md](docs/音訊裝置建議配置.md)
- [docs/快速安裝手冊.md](docs/快速安裝手冊.md)
- [docs/更新紀錄.md](docs/更新紀錄.md)
- [docs/ASR重構藍圖.md](docs/ASR重構藍圖.md)


