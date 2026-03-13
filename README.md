# SyncTranslate

Windows 桌面即時口譯工具，專注於 **會議雙向翻譯** 與 **低延遲字幕 / 語音回放**。

SyncTranslate 將會議音訊與本地麥克風音訊分成兩條獨立管線，透過 **串流 ASR → LLM 翻譯 → TTS 播放** 的方式，把遠端語音翻成本地字幕 / 語音，也能把本地語音翻譯後送回遠端。整體以 **PySide6 GUI** 管理音訊路由、模型設定、即時字幕、健康檢查與診斷資訊。

---

## 功能亮點

- **三種工作模式**
  - `meeting_to_local`：遠端會議語音 → 本地字幕 / 本地播放
  - `local_to_meeting`：本地麥克風語音 → 遠端字幕 / 遠端播放
  - `bidirectional`：雙向同時執行
- **本地串流 ASR**
  - 以 `faster-whisper` 為核心
  - 可調整模型、裝置、精度、VAD、beam、partial / final 行為
  - 支援中文 / 英文分流設定
- **本地翻譯 LLM**
  - 目前以 **LM Studio** 為主要本地推理後端
  - 支援中翻英、英翻中分開指定模型與參數
  - 內建 sliding window / partial stitch / exact cache，降低切句造成的語意斷裂
- **TTS 播放**
  - 目前以 `edge-tts` 為主
  - 中文 / 英文可分開設定聲線、取樣率與佇列策略
  - 內建 backlog 控制、取消策略與防回授保護
- **音訊路由與診斷**
  - 支援本地輸入 / 輸出裝置選擇
  - 支援虛擬音訊裝置（例如 VB-CABLE）進行遠端路由
  - 可檢視 runtime stats、最近錯誤與健康檢查結果

---

## 系統架構

### 端到端流程

```text
遠端會議音訊 / 本地麥克風
        ↓
   AudioCapture / AudioInputManager
        ↓
      ASRManager
        ↓
 Streaming ASR + VAD + utterance/revision
        ↓
  TranslatorManager / TranslationStitcher
        ↓
    TranscriptBuffer / 即時字幕
        ↓
       TTSManager
        ↓
 本地播放 / 遠端虛擬輸出裝置
```

### 主要模組

- `main.py`：程式入口、參數解析、runtime crash logging、GUI 啟動
- `app/ui_main.py`：主視窗與頁面整合
- `app/audio_router.py`：整條音訊 / ASR / 翻譯 / TTS 管線協調
- `app/asr_manager.py`：來源導向 ASR stream 管理、utterance / revision 管理
- `app/translator_manager.py`：翻譯方向映射、caption / speech profile 管理
- `app/local_ai/translation_stitcher.py`：partial 節流、exact cache、上下文拼接
- `app/transcript_buffer.py`：字幕 upsert、partial / final 更新
- `app/tts_manager.py`：TTS 佇列、取消策略、播放控制
- `app/state_manager.py`：session state 與 echo guard 狀態管理
- `app/settings.py` / `app/schemas.py`：設定檔載入、schema、legacy migration

---

## 目標使用情境

SyncTranslate 適合這幾類場景：

- 中英雙向會議口譯
- 本地麥克風語音即時翻譯後送入會議軟體
- 將會議系統音訊翻成字幕與耳機播放
- 需要本地模型、不依賴雲端翻譯 API 的工作流

---

## 系統需求

### 作業系統

- Windows 10 / 11

### Python

- Python 3.11+

### 建議硬體

- **GPU 建議**：NVIDIA 顯示卡（執行 `faster-whisper` 與本地 LLM 時可顯著降低延遲）
- **CPU 可用**：可執行，但延遲通常較高，不建議雙向即時模式
- **音訊裝置**：需能被 PortAudio / WASAPI 正常列出
- **虛擬音訊裝置**：建議安裝 **VB-CABLE** 或相容方案，供遠端路由使用

### Python 依賴

專案的核心相依套件包含：

- `pyside6`
- `sounddevice`
- `soundcard`
- `edge-tts`
- `numpy`
- `pyyaml`
- `comtypes`
- `pycaw`
- `miniaudio`
- `faster-whisper`（optional extra: `local`）

---

## 安裝方式

### 1. 取得專案

```bash
git clone https://github.com/lioil1020-JackLee/SyncTranslate.git
cd SyncTranslate
```

### 2. 安裝依賴

建議使用 `uv`：

```bash
uv sync --extra local
```

如果不用 `uv`，可改用：

```bash
pip install -r requirements.txt
pip install faster-whisper
```

### 3. 建立設定檔

```powershell
Copy-Item .\config.example.yaml .\config.yaml
```

### 4. 啟動程式

```bash
uv run python .\main.py
```

### 5. 只做快速檢查

```bash
uv run python .\main.py --check
```

---

## 快速開始

1. 打開 **音訊路由與診斷** 頁面。
2. 確認：
   - 本地輸入裝置（麥克風）
   - 本地輸出裝置（耳機 / 喇叭）
   - 遠端輸入 / 輸出裝置（通常為 VB-CABLE）
3. 進入 **參數設定**：
   - 設定中文 / 英文 ASR 模型與 VAD
   - 設定中翻英 / 英翻中 LLM 模型
   - 設定中文 / 英文 TTS 聲線
4. 先執行健康檢查，確認 ASR / LLM / TTS 都可用。
5. 進入 **即時字幕** 頁面：
   - 選擇模式
   - 選擇語言方向
   - 需要時先測試本地輸出 TTS
6. 按下「開始」，啟動 session。

---

## 設定檔說明

主設定檔為 `config.yaml`。若不存在，程式會以 `config.example.yaml` 作為 fallback。

### `audio`

控制本地與遠端音訊裝置與音量倍率。

```yaml
audio:
  meeting_in: Windows WASAPI::CABLE Output (VB-Audio Virtual Cable)
  microphone_in: Windows WASAPI::耳機 (WH-1000XM5)
  speaker_out: Windows WASAPI::耳機 (WH-1000XM5)
  meeting_out: Windows WASAPI::CABLE Input (VB-Audio Virtual Cable)
  meeting_in_gain: 1.0
  microphone_in_gain: 1.0
  speaker_out_volume: 1.0
  meeting_out_volume: 1.0
```

### `direction`

```yaml
direction:
  mode: bidirectional
```

可用值：

- `meeting_to_local`
- `local_to_meeting`
- `bidirectional`

### `language`

控制雙向來源語言與目標語言。

```yaml
language:
  meeting_source: en
  meeting_target: zh-TW
  local_source: zh-TW
  local_target: en
```

### `asr` 與 `asr_channels`

- `asr`：共用預設值
- `asr_channels.chinese` / `asr_channels.english`：中文與英文獨立覆寫

常用參數：

```yaml
asr:
  engine: faster_whisper
  model: large-v3
  device: cuda
  compute_type: float16
  beam_size: 1
  condition_on_previous_text: true
  temperature_fallback: 0.0,0.2
  no_speech_threshold: 0.65
  vad:
    enabled: true
    min_speech_duration_ms: 160
    min_silence_duration_ms: 450
    max_speech_duration_s: 10.0
    speech_pad_ms: 300
    rms_threshold: 0.03
  streaming:
    partial_interval_ms: 250
    partial_history_seconds: 2
    final_history_seconds: 4
```

#### 建議做法

- 中文 ASR：偏穩定，可用 `large-v3`
- 英文 ASR：偏低延遲，可用 `distil-large-v3`
- 中文 / 英文的 `beam_size`、`temperature_fallback`、VAD 參數可分開調整

### `llm` 與 `llm_channels`

- `llm`：共用預設值
- `llm_channels.zh_to_en` / `llm_channels.en_to_zh`：翻譯方向獨立覆寫

```yaml
llm:
  backend: lm_studio
  base_url: http://127.0.0.1:1234
  model: hy-mt1.5-7b
  temperature: 0.05
  top_p: 0.9
  max_output_tokens: 128
  repeat_penalty: 1.05
  stop_tokens: '</target>,Translation:'
  request_timeout_sec: 12
  sliding_window:
    enabled: true
    trigger_tokens: 20
    max_context_items: 4
```

#### Profiles

支援不同翻譯風格：

- `live_caption_fast`
- `live_caption_stable`
- `speech_output_natural`
- `technical_meeting`

#### 建議做法

- 中翻英：`hy-mt1.5-7b`
- 英翻中：`hy-mt1.5-7b`
- 如果要更自然的口語輸出，可再為 speech profile 準備另一個模型

### `tts` 與 `tts_channels`

- `tts`：主設定
- `tts_channels.chinese` / `tts_channels.english`：中文 / 英文獨立設定

```yaml
tts:
  engine: edge_tts
  voice_name: zh-TW-HsiaoChenNeural
  length_scale: 0.95
  noise_scale: 0.667
  noise_w: 0.6
  sample_rate: 24000
```

#### 建議做法

- 中文：`zh-TW-HsiaoChenNeural`
- 英文：`en-US-JennyNeural`
- 即時模式建議縮小 TTS 佇列，避免語音 backlog

### `runtime`

控制整體 pipeline 行為。

```yaml
runtime:
  sample_rate: 24000
  chunk_ms: 40
  asr_queue_maxsize_chinese: 16
  asr_queue_maxsize_english: 12
  llm_queue_maxsize_zh_to_en: 4
  llm_queue_maxsize_en_to_zh: 4
  tts_queue_maxsize_chinese: 6
  tts_queue_maxsize_english: 6
  translation_exact_cache_size: 256
  translation_prefix_min_delta_chars: 6
  llm_streaming_tokens: 16
  max_pipeline_latency_ms: 2000
  tts_cancel_pending_on_new_final: true
  tts_cancel_policy: all_pending
  tts_max_wait_ms: 2500
  tts_max_chars: 140
  tts_drop_backlog_threshold: 4
  local_echo_guard_enabled: true
  local_echo_guard_resume_delay_ms: 300
  remote_echo_guard_resume_delay_ms: 300
  warmup_on_start: true
```

---

## UI 頁面說明

### 1. 音訊路由與診斷

用來：

- 選擇本地輸入 / 輸出裝置
- 驗證會議路由是否正確
- 檢查目前 session stats
- 檢視最近錯誤與健康檢查結果

### 2. 即時字幕

用來：

- 切換 session 模式
- 選擇語言方向
- 觀看四個字幕區塊
  - 遠端原文
  - 遠端翻譯
  - 本地原文
  - 本地翻譯
- 啟動 / 停止 session
- 測試本地 TTS

### 3. 參數設定

用來：

- 分別調整中文 / 英文 ASR
- 分別調整中翻英 / 英翻中 LLM
- 分別調整中文 / 英文 TTS
- 調整 queue、latency、streaming tokens、防回授等 runtime 參數

---

## 建議的模型與參數

### ASR

- 中文：`large-v3`
- 英文：`distil-large-v3`

### LLM

- 中翻英：`hy-mt1.5-7b`
- 英翻中：`hy-mt1.5-7b`

### TTS

- 中文：`zh-TW-HsiaoChenNeural`
- 英文：`en-US-JennyNeural`

### 低延遲平衡建議

- `chunk_ms: 40~50`
- 中文 `beam_size: 2`
- 英文 `beam_size: 1`
- `llm_streaming_tokens: 12~16`
- `max_pipeline_latency_ms: 2000~2500`
- `tts_drop_backlog_threshold: 3~4`

---

## 專案結構

```text
SyncTranslate/
├─ app/
│  ├─ local_ai/
│  ├─ asr_manager.py
│  ├─ audio_capture.py
│  ├─ audio_input_manager.py
│  ├─ audio_router.py
│  ├─ pages_audio_routing.py
│  ├─ pages_diagnostics.py
│  ├─ pages_live_caption.py
│  ├─ pages_local_ai.py
│  ├─ schemas.py
│  ├─ session_controller.py
│  ├─ settings.py
│  ├─ state_manager.py
│  ├─ transcript_buffer.py
│  ├─ translator_manager.py
│  ├─ tts_manager.py
│  └─ ui_main.py
├─ config.example.yaml
├─ main.py
├─ pyproject.toml
└─ README.md
```

---

## 故障排除

### 1. 第一個字常被吃掉

通常不是翻譯問題，而是 **VAD / streaming 切段** 問題。可優先調整：

- `min_speech_duration_ms`
- `min_silence_duration_ms`
- `speech_pad_ms`
- `chunk_ms`

### 2. 字幕延遲越來越大

通常是 queue 過大或 TTS backlog。可優先調整：

- `asr_queue_maxsize_*`
- `llm_queue_maxsize_*`
- `tts_queue_maxsize_*`
- `tts_drop_backlog_threshold`
- `max_pipeline_latency_ms`

### 3. 翻譯結果出現 prompt tag / XML tag

可檢查：

- `stop_tokens`
- prompt template
- 模型是否為翻譯專用模型

### 4. TTS 落後播放

可優先調整：

- `tts_cancel_pending_on_new_final`
- `tts_cancel_policy`
- `tts_max_wait_ms`
- `tts_max_chars`

### 5. 健康檢查失敗

請確認：

- LM Studio 已啟動且 API 位址正確
- TTS 可正常生成
- 音訊裝置名稱與 `config.yaml` 一致
- `faster-whisper` 模型已可在本機載入

---

## 開發與除錯

### 啟動檢查模式

```bash
uv run python .\main.py --check
```

### runtime crash log

程式啟動時會建立 `logs/runtime_crash.log`，用於記錄未處理例外與 thread exception。

---

## Roadmap

- 更完整的 prompt / profile 管理
- 更多語言支援
- 更細的 per-channel / per-language runtime 策略
- 更完整的 benchmark / replay / diagnostics 匯出
- 自動模式切換（低延遲 / 平衡 / 高準確）

---

## 授權

若尚未指定授權，請在專案中補上 LICENSE 後再更新本節。

---

## 致謝

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [edge-tts](https://github.com/rany2/edge-tts)
- [PySide6](https://doc.qt.io/qtforpython/)
- VB-CABLE / 虛擬音訊路由工具

