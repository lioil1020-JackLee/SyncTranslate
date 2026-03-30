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
  - `輸出音量` 是共用主音量，會同時影響原音直通與 TTS
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
- `輸出音量` slider 已改成固定樣式，圓形 handle 會對齊 `0.1x` 刻度。

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
- 執行取樣率：`48000`
- chunk：`40ms`
- pre-roll：`220ms`
- 預設輸出增益：`1.4x`
- 字幕 profile：`live_caption_fast`
- 語音 profile：`speech_output_natural`

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
