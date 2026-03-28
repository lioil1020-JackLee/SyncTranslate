# Architecture

## Runtime flow

目前 runtime 的核心由 [`app/application/audio_router.py`](/e:/py/SyncTranslate/app/application/audio_router.py)、[`app/bootstrap/dependency_container.py`](/e:/py/SyncTranslate/app/bootstrap/dependency_container.py)、[`app/application/session_service.py`](/e:/py/SyncTranslate/app/application/session_service.py) 組成。

整體流程如下：

1. `SessionController.start()` 一律以 `bidirectional` 啟動 router。
2. `AudioInputManager` 同時管理兩個輸入 source：
   - `remote` -> `meeting_in`
   - `local` -> `microphone_in`
3. 每個 source 的 audio chunk 都先進 `AudioRouter`。
4. Router 會先嘗試把 chunk 送去對向 output channel 的 passthrough。
5. 若該 source 允許 ASR，再把 chunk 送進該方向的 ASR pipeline。
6. ASR 事件先寫入 original panel：
   - `remote` -> `meeting_original`
   - `local` -> `local_original`
7. 若該方向翻譯啟用，再送入對應 LLM provider。
8. 翻譯結果寫入 translated panel：
   - `remote` -> `meeting_translated`
   - `local` -> `local_translated`
9. 若翻譯關閉，translated panel 仍會更新，但內容回填 ASR 原文。
10. 若 output mode 是 `tts`，final 結果會送到對向輸出播放。

## Direction model

專案已不再支援真正的單向 runtime。

舊名稱例如：

- `remote_only`
- `local_only`
- `meeting_to_local`
- `local_to_meeting`

最多只會出現在舊設定或舊測試語境中；runtime 仍會強制落到雙向模式。

## Source 與 output channel 的關係

這是架構上最重要的對照：

| source | capture device | original panel | translated panel | output channel | output device |
| --- | --- | --- | --- | --- | --- |
| `remote` | `audio.meeting_in` | `meeting_original` | `meeting_translated` | `local` | `audio.speaker_out` |
| `local` | `audio.microphone_in` | `local_original` | `local_translated` | `remote` | `audio.meeting_out` |

所以：

- `remote` source 的 TTS / passthrough 都是往本地播放
- `local` source 的 TTS / passthrough 都是往遠端會議輸出

## Output modes

TTS manager 對每個 output channel 維護獨立模式：

- `tts`
- `subtitle_only`
- `passthrough`

行為：

- `tts`: 播放 final 語音
- `subtitle_only`: 只更新字幕，不播放
- `passthrough`: 直接播放原始音訊 chunk

補充：

- mode 是設定在 output channel，不是 source
- `passthrough` 不會關閉 ASR

## Model strategy

ASR 與 LLM 都是 direction-specific：

- ASR
  - `asr_channels.local`
  - `asr_channels.remote`
- LLM
  - `llm_channels.local`
  - `llm_channels.remote`

這是 runtime 的實際來源，而不是單純文件約定。

## TTS strategy

TTS manager 由 [`app/infra/tts/playback_queue.py`](/e:/py/SyncTranslate/app/infra/tts/playback_queue.py) 負責：

- 管理 `local` / `remote` 兩個輸出 queue
- 播放 TTS
- 播放 passthrough 音訊
- 在切換 output mode 時清空該 channel 相關 pending queue

實際 voice 選擇會再經過 [`app/infra/tts/voice_policy.py`](/e:/py/SyncTranslate/app/infra/tts/voice_policy.py) 依目標語言挑選或修正。

## Health model

舊 warmup / preheat 流程已移除。

目前只保留：

- system check

system check 會驗證：

- ASR availability
- LM Studio connectivity
- edge-tts availability

## Package layout

- `app/application/`: router、session、settings、export 等應用層服務
- `app/bootstrap/`: startup wiring、pipeline bundle、runtime paths
- `app/domain/`: runtime state、events、transcript models
- `app/infra/audio/`: capture、playback、device registry、routing
- `app/infra/asr/`: Faster-Whisper adapter 與 streaming pipeline
- `app/infra/config/`: schema、settings store、migration logic
- `app/infra/translation/`: translation provider、stitcher、LM Studio adapter
- `app/infra/tts/`: edge-tts engine、playback queue、voice policy
- `app/local_ai/`: system-check worker
- `app/ui/`: main window 與各 UI page
- `tests/`: 單元與整合測試
