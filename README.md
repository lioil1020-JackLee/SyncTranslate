# SyncTranslate 文件導覽

## 畫面預覽

![即時字幕畫面](./image/即時字幕.png)

![設定畫面](./image/設定.png)

SyncTranslate 是一個在 Windows 上運作的雙向即時翻譯桌面工具，負責：

1. 擷取遠端會議音訊與本地麥克風音訊
2. 分別送入方向獨立的 ASR 與 LLM
3. 在 UI 顯示原文與翻譯字幕
4. 依每個方向的 ASR / 翻譯 / TTS 設定決定是否顯示字幕、翻譯文字或輸出語音

## 主要畫面

- `即時字幕`
  - 顯示遠端原文、遠端翻譯、本地原文、本地翻譯
  - 每個方向都可獨立設定 `ASR語言`、`翻譯目標`、`TTS語音`
  - `ASR語言 = 無` 代表關閉該方向辨識
  - `翻譯目標 = 無` 代表只顯示原文，不送翻譯
  - `TTS語音 = 無` 代表只顯示文字，不播放譯文語音
  - 音量控制已移到 `設定 > 音訊裝置`，四個 slider 直接控制真實裝置音量
- `設定`
  - 音訊裝置
  - 本地模型設定
  - 系統檢查摘要

## 最近整理

- 即時字幕頁已改成以 `channel` 為中心管理 UI 狀態。
- `remote` / `local` 共享一致的字幕控制邏輯與標籤刷新邏輯。
- `ASR語言 / 翻譯目標 / TTS語音` 都支援獨立關閉，不再依賴額外模式切換。
- `TTS = 無` 時仍可保留字幕與翻譯文字，不會再被自動補回預設聲線。
- passthrough 已改為非同步播放佇列，避免卡住 ASR callback。
- ASR 預設調整為穩定優先，降低 partial 與 final hallucination。
- ASR partial 預設改為較輕量的 live decode 組合，降低即時辨識抖動。
- ASR / LLM 新增 session 內自適應調參，會依最近語句長度與延遲微調 live 策略，但不會自動改寫 `config.yaml`。
- `設定 > 音訊裝置` 的四個音量 slider 以百分比 `0% ~ 100%` 顯示真實裝置音量。

## 核心概念

- `remote` source
  - 來源：`audio.meeting_in`
  - 輸出：`audio.speaker_out`
- `local` source
  - 來源：`audio.microphone_in`
  - 輸出：`audio.meeting_out`

字幕面板固定為：

- `meeting_original`
- `meeting_translated`
- `local_original`
- `local_translated`

## 目前內建預設

- ASR：`large-v3`
- ASR partial beam：`1`
- ASR partial history：`2s`
- 執行取樣率：`48000`
- chunk：`40ms`
- pre-roll：`220ms`
- 內部播放增益：固定 `100%`
- 字幕 profile：`live_caption_fast`
- 語音 profile：`speech_output_natural`
- runtime adaptive：`ASR / LLM = enabled`

## 方向控制規則

- 遠端方向：
  - 來源：`audio.meeting_in`
  - 文字輸出：遠端原文 / 遠端翻譯
  - 語音輸出：`audio.speaker_out`
- 本地方向：
  - 來源：`audio.microphone_in`
  - 文字輸出：本地原文 / 本地翻譯
  - 語音輸出：`audio.meeting_out`
- 建議用法：
  - 會議紀錄單向模式：`本地 ASR語言 = 無`
  - 只看字幕不播報：`TTS語音 = 無`
  - 關閉翻譯只看原文：`翻譯目標 = 無`

## 常用命令

```powershell
uv run python .\main.py
uv run python .\main.py --check
uv run python -m pytest -q
```

## 相關文件

- [架構說明](./docs/架構說明.md)
- [設定說明](./docs/設定說明.md)
- [快速安裝手冊](./docs/快速安裝手冊.md)
- [測試說明](./docs/測試說明.md)
- [更新紀錄](./docs/更新紀錄.md)
- [音訊裝置建議配置](./docs/音訊裝置建議配置.md)
## 2026-03-30 音量控制更新

- `即時字幕` 頁面的共用 `輸出音量` 已移除。
- `設定 > 音訊裝置` 現在提供四個獨立的真實裝置音量控制：`遠端輸入`、`遠端輸出`、`本地輸入`、`本地輸出`。
- 這四個 slider 直接控制 Windows endpoint volume，包含一般硬體裝置與 VoiceMeeter 虛擬裝置。
- slider 顯示為百分比 `0% ~ 100%`，不是 SyncTranslate 內部倍數增益。
- SyncTranslate 內部的 capture / playback gain 現在固定為 `100%`，不再模擬裝置音量。
## 2026-03-30 TTS Queue Update

- TTS queue now separates synthesis and playback, so the next sentence can be synthesized while the current sentence is still playing.
- TTS now speaks the final LLM translation text directly and no longer generates a separate speech-profile rewrite.
- Default behavior no longer speaks stable partial text; ASR, LLM, and TTS stay on a simple final-result pipeline.
- `僅 partial 使用上下文` 開啟時，只有 partial 翻譯會參考前文；final 翻譯仍只看當句，避免同一句 ASR 出現不同譯文。

## 2026-04-01 Runtime Adaptive Tuning

- ASR partial 預設改為 `beam_size = 1`、`partial_history_seconds = 2`，先降低 live partial decode 負擔。
- ASR 會依最近語句長度、partial latency、final latency 與 queue 壓力，動態調整 `partial_interval_ms`、`min_silence_duration_ms`、`soft_final_audio_ms`。
- LLM 會依最近片段長度、翻譯延遲與失敗率，動態調整 partial trigger、最小 partial 間隔、context 長度，必要時暫時從 `live_caption_fast` 切到 `live_caption_stable`。
- 自適應僅在目前 session 生效，不會把學到的值自動寫回 `config.yaml`。
