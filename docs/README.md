# SyncTranslate

SyncTranslate 是一個 Windows 桌面即時口譯工具。專案目前的實作已固定為「雙向、雙來源、雙輸出」模式：本地麥克風與遠端會議音訊永遠同時存在，各自做 ASR、翻譯、字幕更新，必要時再送到對向輸出。

這份 README 以目前程式與測試行為為準，不再沿用舊版單向 session 或簡化流程圖的說法。

## 1. 核心模型

- `local` / `remote` 在程式裡先代表「輸入來源 source」。
- `local` / `remote` 在 TTS manager 裡也代表「輸出 channel」；這裡的意思不同，要分開看。
- `remote` source = 會議音訊輸入，對應 `audio.meeting_in`
- `local` source = 本地麥克風輸入，對應 `audio.microphone_in`
- `local` output channel = 本地喇叭輸出，對應 `audio.speaker_out`
- `remote` output channel = 遠端會議輸出，對應 `audio.meeting_out`
- 四個字幕面板固定為：
  - `meeting_original`
  - `meeting_translated`
  - `local_original`
  - `local_translated`

## 2. 實際雙向音源流程

目前實作的核心在 [`app/application/audio_router.py`](/e:/py/SyncTranslate/app/application/audio_router.py)。

### 2.1 啟動時

- `AudioRouter.start()` 會先把模式強制設成 `bidirectional`
- 兩條 source 都視為 ASR source
- 因此即使外部傳入舊模式名稱，例如 `meeting_to_local`，實際上仍會以雙向模式啟動
- 測試已驗證這件事：[`tests/test_audio_router_policy.py`](/e:/py/SyncTranslate/tests/test_audio_router_policy.py)

### 2.2 每個 audio chunk 的處理

#### `remote` source 進來時

1. 從 `audio.meeting_in` 擷取遠端會議音訊。
2. 先嘗試把原始音訊送到 `local` output channel 的 passthrough。
3. 如果 runtime state 允許 ASR，再送到 `remote` ASR。
4. ASR 結果寫入 `meeting_original`。
5. 若 `remote_translation_enabled = true`，交給 `remote` LLM 翻譯後寫入 `meeting_translated`。
6. 若 `remote_translation_enabled = false`，`meeting_translated` 會直接回填 ASR 原文，不經 LLM。
7. 若最終輸出模式是 `tts`，則把要說出的文字送到 `local` output channel，也就是播放到本地喇叭。

#### `local` source 進來時

1. 從 `audio.microphone_in` 擷取本地麥克風音訊。
2. 先嘗試把原始音訊送到 `remote` output channel 的 passthrough。
3. 如果 runtime state 允許 ASR，再送到 `local` ASR。
4. ASR 結果寫入 `local_original`。
5. 若 `local_translation_enabled = true`，交給 `local` LLM 翻譯後寫入 `local_translated`。
6. 若 `local_translation_enabled = false`，`local_translated` 會直接回填 ASR 原文，不經 LLM。
7. 若最終輸出模式是 `tts`，則把要說出的文字送到 `remote` output channel，也就是播放到會議輸出裝置。

## 3. 正確的方向對應

這裡是最容易在文件裡寫反的地方。

| 輸入來源 | 原文面板 | 翻譯面板 | TTS / passthrough 目標輸出 |
| --- | --- | --- | --- |
| `remote` | `meeting_original` | `meeting_translated` | `local` channel -> `audio.speaker_out` |
| `local` | `local_original` | `local_translated` | `remote` channel -> `audio.meeting_out` |

也就是說：

- 遠端聽到的內容，最終若要播出，是播到本地喇叭
- 本地說的內容，最終若要播出，是播到會議輸出裝置

## 4. `tts`、`subtitle_only`、`passthrough` 的實際語意

每個輸出 channel 都有自己的 output mode，由 UI 推導後呼叫 `audio_router.set_output_mode(channel, mode)`。

- `tts`
  - 該方向有翻譯
  - 且該方向有選 TTS voice
  - final 結果才會排入 TTS queue
- `subtitle_only`
  - 該方向有翻譯
  - 但 TTS voice = `none`
  - 仍會更新字幕，但不播音
- `passthrough`
  - 該方向翻譯目標 = `none`
  - 原始音訊直接走對向播放
  - 注意：passthrough 不會停掉 ASR；README 舊版這點寫得太含糊

實作上：

- `remote` source 的 passthrough 會送到 `local` output channel
- `local` source 的 passthrough 會送到 `remote` output channel
- output mode 是設定在輸出 channel，不是設定在輸入 source

## 5. 翻譯與語音的 fallback 行為

目前流程不是「沒翻譯就完全沒有 translated panel」。

- 若某個 source 的翻譯關閉，translated panel 仍會更新
- 寫入的是 ASR 原文
- 若該方向輸出模式仍是 `tts`，final ASR 原文也可以直接拿去播

這點可在 [`app/application/audio_router.py`](/e:/py/SyncTranslate/app/application/audio_router.py) 與 [`tests/test_audio_router_policy.py`](/e:/py/SyncTranslate/tests/test_audio_router_policy.py) 看見。

## 6. 專案結構

- `app/bootstrap/`: 啟動流程、依賴注入、runtime path
- `app/application/`: `AudioRouter`、Session / Settings / Export 等協調服務
- `app/domain/`: runtime state、事件、字幕模型
- `app/infra/audio/`: 裝置列舉、capture、playback、routing
- `app/infra/asr/`: Faster-Whisper 串流辨識
- `app/infra/translation/`: LM Studio 翻譯 provider 與 stitcher
- `app/infra/tts/`: edge-tts、queue、voice policy
- `app/local_ai/`: system check
- `app/ui/`: 主視窗與各頁面
- `tests/`: 單元與整合測試

## 7. 主要設定位置

常用設定可從這幾區看起：

- 音訊路由
  - `audio.meeting_in`
  - `audio.microphone_in`
  - `audio.speaker_out`
  - `audio.meeting_out`
- ASR
  - `asr_channels.remote`
  - `asr_channels.local`
  - `runtime.remote_asr_language`
  - `runtime.local_asr_language`
- 翻譯
  - `llm_channels.remote`
  - `llm_channels.local`
  - `runtime.remote_translation_enabled`
  - `runtime.local_translation_enabled`
  - `runtime.remote_translation_target`
  - `runtime.local_translation_target`
- TTS
  - `meeting_tts`
  - `local_tts`
  - `tts_channels.local`
  - `tts_channels.remote`
  - `runtime.remote_tts_voice`
  - `runtime.local_tts_voice`

## 8. 安裝與啟動

1. 建議先建立虛擬環境。
2. 安裝依賴：
   - `pip install -r requirements.txt`
   - 或使用 `uv` 安裝專案依賴
3. 啟動 UI：
   - `uv run python .\main.py`
4. 僅執行 system check：
   - `uv run python .\main.py --check`

## 9. 測試

- 全部測試：`pytest -q`
- 建議先看這幾個檔案理解流程：
  - [`tests/test_pipeline_integration.py`](/e:/py/SyncTranslate/tests/test_pipeline_integration.py)
  - [`tests/test_audio_router_policy.py`](/e:/py/SyncTranslate/tests/test_audio_router_policy.py)
  - [`tests/test_multilingual_channel_policy.py`](/e:/py/SyncTranslate/tests/test_multilingual_channel_policy.py)
  - [`tests/test_audio_routing_virtual_devices.py`](/e:/py/SyncTranslate/tests/test_audio_routing_virtual_devices.py)

## 10. 注意事項

- 啟動 warmup 已移除，現在改成 system check。
- `direction.mode` 已固定為 `bidirectional`。
- `edge-tts` 是目前預設 TTS 路徑。
- runtime log 與健康檢查輸出會寫到系統暫存目錄 `SyncTranslate`。

## 11. 延伸閱讀

- [架構文件](./ARCHITECTURE.md)
- [配置文件](./CONFIGURATION.md)
- [測試文件](./TESTING.md)
- [更新日誌](./CHANGELOG.md)
