# SyncTranslate

SyncTranslate 是一個以 Windows 桌面為主的即時雙向字幕 / 翻譯 / 語音輸出工具。它可以同時處理本地與遠端兩條音訊通道，將音訊送進 ASR、翻譯與 TTS，並在 UI 中即時顯示原文與譯文。

目前專案的主線已切換到 `ASR v2`。新的 ASR 架構重點是：

- 中文 ASR 自動使用 `FunASR + FSMN-VAD`
- 非中文與 `auto` 自動使用 `faster-whisper`
- `none` 會直接停用該通道 ASR
- 模型採懶載入與共享 registry，避免啟動時重複初始化
- diagnostics / session report 會輸出每通道實際使用的 backend、effective device、初始化狀態

舊版 `legacy` ASR 仍暫時保留作為 fallback，但已停止擴充。後續功能與調整都以 `ASR v2` 為主。

## 目前功能

- 雙通道音訊路由
  - `remote` 與 `local` 可同時存在
  - 可對應 VoiceMeeter / 實體裝置 / 虛擬裝置
- 即時 ASR
  - 中文走 FunASR
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
3. `ASRManagerV2` 依通道語言解析 backend
4. `EndpointingRuntime` 執行 VAD / endpointing
5. `SourceRuntimeV2` 負責 partial / final 事件發送
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
  - `funasr_registry.py`
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

## 安裝與執行

建議使用 `uv`：

```powershell
uv sync --locked --extra local
```

啟動程式：

```powershell
uv run python .\main.py
```

執行系統檢查：

```powershell
uv run python .\main.py --check
```

執行測試（全套 319 tests）：

```powershell
uv run pytest -q
```

如果要打包 onedir：

```powershell
uv sync --locked --extra local --group build
uv run pyinstaller .\SyncTranslate-onedir.spec --noconfirm --clean
```

## 設定重點

目前最重要的 runtime 設定包括：

- `runtime.asr_pipeline`：預設 `v2`
- `runtime.local_asr_language` / `runtime.remote_asr_language`：決定 ASR backend
- `runtime.asr_v2_endpointing`
- `runtime.asr_profile_local` / `runtime.asr_profile_remote`：endpoint profile（default / meeting_room / headset / noisy_environment / max_accuracy / low_latency）
- `runtime.enable_postprocessor` / `runtime.enable_partial_stabilization`：ASR 後處理
- `runtime.glossary_enabled` / `runtime.glossary_path`：術語表套用
- `runtime.degradation_policy_enabled`：streaming 降級保護
- `runtime.enable_structured_logging`：jsonl 結構化日誌
- `runtime.asr_queue_maxsize_local` / `runtime.asr_queue_maxsize_remote`

ASR 路由規則：

- `zh` / `zh-TW` / `zh-CN` / `cmn` / `yue` -> `funasr_v2`
- 其他語言 -> `faster_whisper_v2`
- `auto` -> `faster_whisper_v2`
- `none` -> disabled

## 目前已知限制

- 若實際 `effective device` 為 `CPU`，雙通道同時中文發話時仍可能因吞吐量不足而增加延遲
- speaker diarization 目前仍屬實驗性功能，預設關閉
- `runtime.asr_v2_backend` 目前主要作為相容欄位保留；實際 backend 以通道語言解析結果為準
- legacy ASR 尚未完全刪除，但不再作為主要維護路線

## 文件索引

- [docs/架構說明.md](docs/架構說明.md)
- [docs/設定說明.md](docs/設定說明.md)
- [docs/測試說明.md](docs/測試說明.md)
- [docs/音訊裝置建議配置.md](docs/音訊裝置建議配置.md)
- [docs/快速安裝手冊.md](docs/快速安裝手冊.md)
- [docs/更新紀錄.md](docs/更新紀錄.md)
