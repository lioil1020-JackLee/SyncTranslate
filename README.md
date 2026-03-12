# SyncTranslate

SyncTranslate 是一個 Windows 桌面即時口譯工具，專注在「會議雙向翻譯」場景。

它可以同時處理兩條音訊方向：
- 會議來音 (英文) -> 中文字幕 + 中文語音播到你的耳機
- 本地麥克風 (中文) -> 英文字幕 + 英文語音送回會議軟體

整體流程為 Audio Capture -> ASR -> LLM 翻譯 -> TTS -> Audio Playback，並提供 GUI 介面做路由設定、健康檢查、字幕監看與除錯。

## 目錄

- [核心功能](#核心功能)
- [系統需求](#系統需求)
- [快速開始](#快速開始)
- [執行方式](#執行方式)
- [操作流程](#操作流程)
- [設定檔說明](#設定檔說明)
- [專案架構](#專案架構)
- [執行架構與資料流](#執行架構與資料流)
- [健康檢查與診斷](#健康檢查與診斷)
- [常見問題](#常見問題)
- [開發者說明](#開發者說明)

## 核心功能

- 雙向翻譯管線
	- `meeting_to_local`: 會議音訊翻譯到本地
	- `local_to_meeting`: 麥克風音訊翻譯回會議
	- `bidirectional`: 雙向同時執行
- 即時字幕與最終字幕分離顯示
	- partial/final 事件分流，降低延遲與誤觸發
- 本地 ASR
	- 使用 `faster-whisper`，支援模型、裝置、VAD 參數調整
- 本地 LLM 翻譯
	- 支援 `Ollama` 與 `LM Studio` 相容路徑
	- Sliding Window 拼接上下文，減少句子切割帶來的語意斷裂
- 雙 TTS 引擎
	- `piper`: 離線語音
	- `edge_tts`: 雲端語音
- 音訊路由與系統音量整合
	- 裝置選擇、增益/音量控制、路由檢查、Voicemeeter 輔助
- 診斷能力
	- 子程序健康檢查 (ASR/LLM/TTS)
	- 輸出裝置語音測試
	- 錯誤事件紀錄與診斷匯出

## 系統需求

- OS: Windows 10/11
- Python: 3.11 以上
- 套件管理: `uv` (建議)
- 音訊需求:
	- `sounddevice` + PortAudio 可正常列出裝置
	- 若使用虛擬混音路由，建議搭配 Voicemeeter
- 建議硬體:
	- NVIDIA GPU (執行 ASR 會更順)
	- 足夠 CPU 與 RAM 以支援雙向即時處理

## 快速開始

1. 安裝依賴

```powershell
uv sync --extra local
```

2. 複製設定檔

```powershell
Copy-Item .\config.example.yaml .\config.yaml
```

3. 啟動程式

```powershell
uv run python .\main.py
```

4. 先跑快速檢查 (可選)

```powershell
uv run python .\main.py --check
```

## 執行方式

- 開啟 GUI:

```powershell
uv run python .\main.py
```

- 指定設定檔:

```powershell
uv run python .\main.py --config .\config.yaml
```

- 僅做設定與裝置檢查:

```powershell
uv run python .\main.py --check
```

## 操作流程

1. 開啟「音訊路由與診斷」
2. 選擇 `meeting_in`、`microphone_in`、`speaker_out`、`meeting_out`
3. 設定方向模式 (`meeting_to_local` / `local_to_meeting` / `bidirectional`)
4. 到「參數設定」調整 ASR、LLM、TTS
5. 執行健康檢查，確認三段狀態為正常
6. 到「即時字幕」按「開始」
7. 視需要在除錯頁觀察路由與 runtime stats

## 設定檔說明

主設定檔為 `config.yaml`。若不存在，程式會以 `config.example.yaml` 當 fallback。

### audio

- `meeting_in`: 會議來音的輸入裝置名稱
- `microphone_in`: 本地麥克風輸入裝置
- `speaker_out`: 翻譯後播放到本地耳機/喇叭的輸出裝置
- `meeting_out`: 翻譯後回送會議的輸出裝置
- `meeting_in_gain`: 會議輸入增益，預設 `1.0`
- `microphone_in_gain`: 麥克風輸入增益，預設 `1.0`
- `speaker_out_volume`: 本地播放音量比例，預設 `1.0`
- `meeting_out_volume`: 回送會議音量比例，預設 `1.0`

### direction

- `mode`: 
	- `meeting_to_local`
	- `local_to_meeting`
	- `bidirectional`

### language

- `meeting_source`: 會議來源語言 (例如 `en`)
- `meeting_target`: 會議翻譯目標語言 (例如 `zh-TW`)
- `local_source`: 本地來源語言
- `local_target`: 本地翻譯目標語言

### asr

- `engine`: 目前預設 `faster_whisper`
- `model`: 例如 `small`、`medium`、`large-v3`
- `device`: `cuda` / `cpu` / `auto`
- `compute_type`: `float16` / `int8` 等
- `beam_size`: Beam Search 寬度
- `condition_on_previous_text`: 是否參考前文

`asr.vad`:
- `enabled`: 啟用 VAD
- `min_speech_duration_ms`: 最短語音段長
- `min_silence_duration_ms`: 最短靜音段長
- `max_speech_duration_s`: 最長語音段
- `speech_pad_ms`: 語音前後補白
- `rms_threshold`: 音量門檻

`asr.streaming`:
- `partial_interval_ms`: partial 更新間隔
- `partial_history_seconds`: partial 保留秒數
- `final_history_seconds`: final 保留秒數

### llm

- `backend`: `ollama` 或 `lm_studio`
- `base_url`: 後端 API URL
- `model`: 模型名稱
- `temperature`: 溫度
- `top_p`: nucleus sampling
- `request_timeout_sec`: 逾時秒數

`llm.sliding_window`:
- `enabled`: 是否啟用上下文拼接
- `trigger_tokens`: 觸發翻譯最小 token 門檻
- `max_context_items`: 最多帶入歷史片段數

### meeting_tts / local_tts / tts

- `engine`: `piper` 或 `edge_tts`
- `executable_path`: piper 執行檔路徑
- `model_path`: piper 語音模型路徑
- `config_path`: piper 模型設定路徑
- `voice_name`: edge_tts voice 名稱
- `speaker_id`: piper speaker id
- `length_scale`: 語速控制
- `noise_scale`: 音色擾動
- `noise_w`: 發音節奏擾動
- `sample_rate`: TTS 輸出採樣率

說明:
- `meeting_tts` 用於「會議來音 -> 本地」方向
- `local_tts` 用於「本地麥克風 -> 會議」方向
- `tts` 為舊欄位相容；目前 UI 會以 `meeting_tts`、`local_tts` 為主

### runtime

- `sample_rate`: 全域擷取採樣率
- `chunk_ms`: 音訊切塊大小
- `asr_queue_maxsize`: ASR 佇列上限
- `llm_queue_maxsize`: LLM 佇列上限
- `tts_queue_maxsize`: TTS 佇列上限
- `warmup_on_start`: 啟動時是否自動預熱/健康檢查

### health_last_success

- `asr` / `llm` / `tts`: 最近一次成功狀態的摘要字串

## 專案架構

```text
SyncTranslate/
	main.py                        # 程式入口與 UI 啟動
	config.example.yaml            # 範例設定
	config.yaml                    # 本機設定 (不建議提交)
	app/
		ui_main.py                   # 主視窗、分頁與 session 控制
		session_controller.py        # 雙向 pipeline 啟停協調
		pipeline_direction.py        # 單方向處理管線
		audio_capture.py             # 音訊擷取
		audio_playback.py            # 音訊播放與 fallback
		audio_device_selection.py    # 裝置名稱比對/選擇
		device_manager.py            # 輸入輸出裝置查詢
		router.py                    # 路由檢查
		settings.py                  # YAML 設定載入/儲存
		schemas.py                   # 設定與資料結構
		transcript_buffer.py         # 字幕緩衝
		debug_panel.py               # 除錯分頁
		pages_*.py                   # 各 UI 子頁面
		windows_volume.py            # Windows 系統音量整合
		local_ai/
			faster_whisper_engine.py   # ASR 封裝
			streaming_asr.py           # 串流 ASR 管理
			vad_segmenter.py           # 語音切段
			translation_stitcher.py    # 翻譯上下文拼接
			ollama_client.py           # LLM 客戶端
			piper_tts.py               # Piper TTS 引擎
			tts_factory.py             # TTS 工廠
			healthcheck.py             # 健康檢查邏輯
			healthcheck_worker.py      # 子程序健康檢查入口
			runtime_paths.py           # 路徑解析工具
```

## 執行架構與資料流

每個方向都使用 `DirectionalPipeline`，流程如下：

1. `AudioCapture` 依 `chunk_ms` 擷取音訊
2. `StreamingAsr` 收 chunk，VAD 判斷切段
3. 產生 `AsrEvent` (partial/final)
4. `TranslationStitcher` 收斂語境後呼叫 LLM
5. 翻譯結果進入 `TranscriptBuffer`
6. 需播報時送入 TTS queue
7. TTS worker 合成語音後交給 `AudioPlayback`

`SessionController` 依 mode 控制兩條 pipeline 的啟停。UI 端會持續拉取狀態更新字幕與除錯面板。

## 健康檢查與診斷

- 健康檢查透過 `app.local_ai.healthcheck_worker` 子程序執行，避免主 UI 被阻塞
- 檢查項目:
	- ASR 模型可用性
	- LLM 後端可連通性
	- TTS 引擎可用性 (Piper 路徑、Edge 套件)
- 診斷匯出:
	- 產生 `diagnostics_YYYYMMDD_HHMMSS.txt`
	- 包含路由、模型、執行參數與最近錯誤

## 常見問題

### 1) 開始後沒有字幕

- 檢查 `meeting_in` / `microphone_in` 是否有選到可用輸入裝置
- 在「除錯」頁確認 route check 與 capture level
- 降低 `asr.vad.rms_threshold`，避免語音被當作靜音

### 2) 有字幕但沒有聲音

- 確認 `speaker_out` / `meeting_out` 指向可播放裝置
- 在「音訊路由與診斷」頁先做 TTS 測試
- 若是虛擬裝置，優先確認 host API 與系統裝置可用性

### 3) LLM 翻譯延遲高

- 降低 `llm.sliding_window.max_context_items`
- 提高模型推論資源或改用較小模型
- 適度提高 `llm.request_timeout_sec` 避免過早 timeout

### 4) ASR 反應慢

- 降低模型尺寸 (`large-v3` -> `small`)
- 使用 `cuda` + `float16`
- 拉大 `chunk_ms` 可能降低呼叫頻率，但會提高單次延遲

## 開發者說明

### 環境建置

```powershell
uv sync --extra local
```

### 快速檢查

```powershell
uv run python .\main.py --check
```

### 重要設計原則

- UI 與重運算隔離: 健康檢查、session 啟停都用背景 thread/subprocess
- 音訊穩定優先: `audio_playback.py` 含多層 fallback
- 設定即時套用: UI 變更透過 debounce 套用，降低誤觸與重啟成本

### 版本管理建議

- 不要提交 `config.yaml`（機器相關設定）
- 不要提交 `.venv/`、`.uv-cache/`、`logs/` 產物
- 大模型檔案建議放在 `models/` 並以外部發佈方式管理

