# Configuration

## Main file

本機設定使用 `config.yaml`。

範例檔在 [`config.example.yaml`](../config.example.yaml)。

## 1. 先分清楚兩層鍵名

這個專案有兩套名稱：

- 外部 YAML 鍵
  - 寫在 `config.yaml`
  - 由 [`app/infra/config/settings_store.py`](/e:/py/SyncTranslate/app/infra/config/settings_store.py) 讀寫與轉換
- 內部 dataclass 欄位
  - 定義在 [`app/infra/config/schema.py`](/e:/py/SyncTranslate/app/infra/config/schema.py)
  - 這是 runtime 真正使用的欄位

最常見的差異是 `language`：

- 外部 YAML 會寫：
  - `language.remote_translation_target`
  - `language.local_translation_target`
- 載入後會轉成內部欄位：
  - `language.meeting_target`
  - `language.local_target`

所以如果你在看程式碼，會常看到 `meeting_target` / `local_target`；如果你在看 `config.yaml`，則會看到 `remote_translation_target` / `local_translation_target`。

## 2. 固定行為

以下行為目前被程式固定，不是可自由切換的 session 模式：

- `direction.mode` 會被強制成 `bidirectional`
- 啟動 warmup 已移除
- ASR 與 LLM 一律採 direction-specific 設定
- 舊的單向模式名稱就算傳進 runtime，也只會被當作雙向模式處理

## 3. Audio routing

`audio` 區塊包含四個裝置選擇：

- `meeting_in`
- `microphone_in`
- `speaker_out`
- `meeting_out`

方向對應如下：

- `meeting_in`: 遠端會議音訊輸入
- `microphone_in`: 本地麥克風輸入
- `speaker_out`: 本地喇叭輸出
- `meeting_out`: 遠端會議輸出

`*_gain` 與 `*_volume` 欄位仍存在，但目前會被 UI / runtime 固定為 `1.0`。

## 4. Language 與翻譯控制

### 外部 YAML 常用鍵

- `language.remote_translation_target`
- `language.local_translation_target`
- `runtime.remote_translation_enabled`
- `runtime.local_translation_enabled`
- `runtime.remote_translation_target`
- `runtime.local_translation_target`

### 實際 runtime 行為

- `runtime.remote_translation_enabled`
  - 控制 `remote` source 是否進 LLM
- `runtime.local_translation_enabled`
  - 控制 `local` source 是否進 LLM
- 若翻譯關閉：
  - `*_translated` 面板仍會更新
  - 寫入的是 ASR 原文

注意：

- `language.remote_translation_target` / `language.local_translation_target` 是 YAML 對外表示
- `runtime.remote_translation_target` / `runtime.local_translation_target` 是 UI 直接控制與 runtime 判斷用欄位
- 載入與儲存時兩套資訊會由 settings store 協調

## 5. ASR

共享預設在：

- `asr`

方向專用覆寫在：

- `asr_channels.local`
- `asr_channels.remote`

每個方向可以各自設定：

- model
- device
- compute type
- beam size
- VAD
- streaming 參數

### ASR 語言

不是只有 auto。

可用欄位：

- `runtime.remote_asr_language`
- `runtime.local_asr_language`

行為如下：

- 若欄位值是 `auto`，runtime 會用 ASR 偵測語言
- 若欄位值是固定語言，例如 `en`、`zh-TW`、`ja`，runtime 會用該語言作為 source language

## 6. LLM

共享預設在：

- `llm`

方向專用覆寫在：

- `llm_channels.local`
- `llm_channels.remote`

因此 `local` 與 `remote` 可以用不同的：

- LM Studio model
- base URL
- decoding 參數
- caption profile
- speech profile

## 7. TTS

TTS 目前預設走 `edge-tts`。

相關區塊：

- `tts`
- `meeting_tts`
- `local_tts`
- `tts_channels.local`
- `tts_channels.remote`
- `runtime.remote_tts_voice`
- `runtime.local_tts_voice`
- `runtime.tts_output_mode`

### TTS profile 的語意

- `meeting_tts`
  - 比較接近中文 / 會議方向的基底設定
- `local_tts`
  - 比較接近英文 / 本地方向的基底設定
- `tts_channels.local`
  - 對 `local` output channel 的 override
- `tts_channels.remote`
  - 對 `remote` output channel 的 override

實際播哪個 voice，仍會再經過 [`app/infra/tts/voice_policy.py`](/e:/py/SyncTranslate/app/infra/tts/voice_policy.py) 依 target language 做修正。

### output mode

每個 output channel 最終模式來自 UI 推導：

- `tts`
- `subtitle_only`
- `passthrough`

補充：

- `runtime.tts_output_mode` 是整體摘要欄位
- 真正套到 router 的，是每個 channel 各自的 mode
- `passthrough` 不會停用 ASR

## 8. 設定檔相容與 migration

[`app/infra/config/settings_store.py`](/e:/py/SyncTranslate/app/infra/config/settings_store.py) 會處理舊版鍵名，例如：

- 舊 `remote_translation_target` / `local_translation_target` 映到內部 `meeting_target` / `local_target`
- 舊 `chinese_tts` / `english_tts` 轉為 `meeting_tts` / `local_tts`
- 舊 queue 欄位會轉成 `*_local` / `*_remote`
- 舊單向 session 設定會被吃掉，最後仍轉成雙向模式

## 9. Runtime artifacts

程式會把 runtime 產物寫到系統暫存資料夾 `SyncTranslate`，包含：

- crash log
- runtime event log
- system-check snapshots
## 10. Quick Presets

設定頁上方的「快速設定」不是只改 UI 顯示，會同步帶動一組進階參數。

### 效能預設

`低延遲`
- ASR：`large-v3-turbo`
- Beam：`1`
- Partial interval：`240ms`
- Partial / Final history：`2s / 4s`
- VAD：偏快切段
  - min speech `120ms`
  - min silence `180ms`
  - speech pad `120ms`
  - max speech `12s`
  - rms threshold `0.020`
  - no speech threshold `0.72`
- Queue：
  - ASR `64`
  - LLM `16`
  - TTS `16`
- Runtime：
  - chunk `30ms`
  - ASR pre-roll `220ms`
  - translation cache `128`
  - prefix delta `10`
  - llm streaming tokens `32`
  - max pipeline latency `1400ms`
- TTS：
  - max wait `1500ms`
  - max chars `120`
  - drop backlog threshold `3`
  - cancel policy `older_only`
- LLM max tokens：
  - local `96`
  - remote `128`

`平衡（建議）`
- ASR：`large-v3-turbo`
- Beam：`2`
- Partial interval：`320ms`
- Partial / Final history：`3s / 6s`
- VAD：平衡切段
  - min speech `140ms`
  - min silence `240ms`
  - speech pad `180ms`
  - max speech `16s`
  - rms threshold `0.018`
  - no speech threshold `0.65`
- Queue：
  - ASR `96`
  - LLM `24`
  - TTS `24`
- Runtime：
  - chunk `40ms`
  - ASR pre-roll `350ms`
  - translation cache `256`
  - prefix delta `6`
  - llm streaming tokens `16`
  - max pipeline latency `2200ms`
- TTS：
  - max wait `2200ms`
  - max chars `160`
  - drop backlog threshold `4`
  - cancel policy `older_only`
- LLM max tokens：
  - local `128`
  - remote `160`

`高準確`
- ASR：`large-v3`
- Beam：`4`
- Partial interval：`420ms`
- Partial / Final history：`4s / 8s`
- VAD：偏穩定與完整句
  - min speech `180ms`
  - min silence `320ms`
  - speech pad `220ms`
  - max speech `20s`
  - rms threshold `0.015`
  - no speech threshold `0.55`
- Queue：
  - ASR `128`
  - LLM `32`
  - TTS `32`
- Runtime：
  - chunk `60ms`
  - ASR pre-roll `500ms`
  - translation cache `384`
  - prefix delta `4`
  - llm streaming tokens `12`
  - max pipeline latency `3200ms`
- TTS：
  - max wait `3500ms`
  - max chars `220`
  - drop backlog threshold `6`
  - cancel policy `all_pending`
- LLM max tokens：
  - local `128`
  - remote `192`

### 翻譯風格

`精準直譯`
- temperature `0.0`
- top_p `0.75`
- repeat_penalty `1.10`
- max tokens：local `96` / remote `128`
- sliding-window trigger tokens `20`
- max context items `4`

`平衡`
- temperature `0.12`
- top_p `0.88`
- repeat_penalty `1.06`
- max tokens：local `128` / remote `160`
- sliding-window trigger tokens `16`
- max context items `5`

`自然語氣`
- temperature `0.24`
- top_p `0.94`
- repeat_penalty `1.02`
- max tokens：local `160` / remote `224`
- sliding-window trigger tokens `14`
- max context items `6`

### TTS 朗讀策略

`清晰播報（建議）`
- base / channel sample rate `24000`
- noise_w `0.65`
- length_scale `1.0`
- noise_scale `0.60`
- tts max wait `2400ms`
- tts max chars `180`
- backlog threshold `4`
- cancel policy `all_pending`

`平衡`
- base / channel sample rate `22050`
- noise_w `0.75`
- length_scale `1.0`
- noise_scale `0.667`
- tts max wait `2200ms`
- tts max chars `160`
- backlog threshold `4`
- cancel policy `all_pending`

`輕快對話`
- base / channel sample rate `22050`
- noise_w `0.85`
- length_scale `1.03`
- noise_scale `0.70`
- tts max wait `2600ms`
- tts max chars `220`
- backlog threshold `5`
- cancel policy `older_only`

`快速回應`
- base / channel sample rate `24000`
- noise_w `0.55`
- length_scale `0.96`
- noise_scale `0.58`
- tts max wait `1400ms`
- tts max chars `120`
- backlog threshold `3`
- cancel policy `older_only`

### 備註

- 「快速設定」會直接覆蓋對應的進階欄位。
- 若你手動再改進階欄位，快速設定在語意上就應視為 `自訂`。
- 目前 `模型策略` 仍是固定為方向獨立，快速設定不會把 local / remote 合併成共享模型。
