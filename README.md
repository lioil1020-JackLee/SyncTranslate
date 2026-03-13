# SyncTranslate

SyncTranslate 是一個 Windows 桌面即時口譯工具，聚焦在會議雙向翻譯場景。

目前版本的實際流程是：音訊擷取 -> 串流 ASR -> LLM 翻譯 -> Edge TTS -> 指定輸出裝置播放，並以 PySide6 GUI 管理音訊路由、模型參數、即時字幕與健康檢查。

## 核心功能

- 雙向翻譯模式
	- `meeting_to_local`: 遠端會議語音翻成本地字幕與本地播放
	- `local_to_meeting`: 本地麥克風語音翻成遠端字幕與遠端播放
	- `bidirectional`: 雙向同時執行
- 本地 ASR
	- 使用 `faster-whisper`
	- 支援模型、裝置、VAD 與 streaming 參數調整
- 本地 LLM 翻譯
	- 只支援 `LM Studio`
	- Sliding window 翻譯拼接，降低片段切分造成的語意斷裂
- TTS
	- 現行只保留 `edge-tts`
	- 支援主設定加通道覆寫：本機輸出與遠端輸出可用不同聲線/取樣率
- 音訊路由
	- 本地輸入與本地輸出可從 GUI 選擇
	- 遠端輸入與遠端輸出目前固定為 VB-CABLE 裝置
	- 支援 Windows 系統音量同步
- 診斷與除錯
	- 健康檢查：ASR / LLM / TTS
	- 即時 runtime stats 與最近錯誤
	- 匯出診斷資訊

## 系統需求

- Windows 10/11
- Python 3.11+
- 建議使用 `uv`
- `sounddevice` 與 PortAudio 可正常列出音訊裝置
- 若要走會議雙向路由，需先安裝並設定 VB-CABLE 或相容虛擬音訊裝置
- 若要順跑 `faster-whisper`，建議使用 NVIDIA GPU；CPU 也可運行但延遲較高

## 快速開始

1. 安裝依賴

```powershell
uv sync --extra local
```

2. 建立設定檔

```powershell
Copy-Item .\config.example.yaml .\config.yaml
```

3. 啟動 GUI

```powershell
uv run python .\main.py
```

4. 只做快速檢查

```powershell
uv run python .\main.py --check
```

## 操作流程

1. 到「音訊路由與診斷」確認本地輸入與本地輸出裝置。
2. 遠端路由使用固定 VB-CABLE 裝置；若系統名稱不同，先調整 config.example.yaml 或程式常數。
3. 到「參數設定」調整 ASR、LLM、TTS 主設定與通道覆寫。
4. 執行健康檢查，確認 ASR / LLM / TTS 都可用。
5. 到「即時字幕」選擇模式與語言方向。
6. 需要時先按「測試本地輸出TTS」。
7. 按「開始」啟動 session。

## 設定檔說明

主設定檔為 `config.yaml`。如果檔案不存在，程式會使用 `config.example.yaml` 作為 fallback。

### `audio`

- `meeting_in`: 遠端輸入裝置。現行 UI 固定為 `Windows WASAPI::CABLE Output (VB-Audio Virtual Cable)`。
- `microphone_in`: 本地麥克風輸入裝置。
- `speaker_out`: 本地播放裝置。
- `meeting_out`: 遠端輸出裝置。現行 UI 固定為 `Windows WASAPI::CABLE Input (VB-Audio Virtual Cable)`。
- `meeting_in_gain`: 遠端輸入增益。
- `microphone_in_gain`: 本地麥克風增益。
- `speaker_out_volume`: 本地輸出音量比例。
- `meeting_out_volume`: 遠端輸出音量比例。

### `direction`

- `mode`: `meeting_to_local`、`local_to_meeting`、`bidirectional`

### `language`

- `meeting_source` / `meeting_target`: 遠端方向語言
- `local_source` / `local_target`: 本地方向語言

### `asr`

- `engine`: 目前為 `faster_whisper`
- `model`: 例如 `small`、`medium`、`large-v3`
- `device`: `cuda`、`cpu`、`auto`
- `compute_type`: 通常由裝置推導，GUI 會自動處理
- `beam_size`: Beam Search 寬度
- `condition_on_previous_text`: 是否延續前文

`asr.vad`:

- `enabled`
- `min_speech_duration_ms`
- `min_silence_duration_ms`
- `max_speech_duration_s`
- `speech_pad_ms`
- `rms_threshold`

`asr.streaming`:

- `partial_interval_ms`
- `partial_history_seconds`
- `final_history_seconds`

### `llm`

- `backend`: 固定 `lm_studio`
- `base_url`: 後端 API 位址
- `model`: 模型名稱
- `temperature`
- `top_p`
- `request_timeout_sec`

`llm.sliding_window`:

- `enabled`
- `trigger_tokens`
- `max_context_items`

### `tts`

`tts` 是共用主設定，儲存兩個方向共用的語音參數。

- `engine`: 目前固定 `edge_tts`
- `voice_name`: 預設聲線
- `length_scale`
- `noise_scale`
- `noise_w`
- `sample_rate`

### `tts_channels`

用來覆寫單一輸出通道設定。

- `tts_channels.local`: 遠端來音翻譯後，播到本機喇叭的通道
- `tts_channels.remote`: 本地麥克風翻譯後，送往遠端會議的通道

可覆寫欄位包含：

- `engine`
- `voice_name`
- `sample_rate`
- `noise_w`

### `meeting_tts` / `local_tts`

這兩個欄位目前仍保留，主要用於相容既有設定與執行期展開後的有效設定。

### `runtime`

- `sample_rate`
- `chunk_ms`
- `asr_queue_maxsize`
- `llm_queue_maxsize`
- `tts_queue_maxsize`
- `warmup_on_start`

### `health_last_success`

- `asr`
- `llm`
- `tts`

## 目前架構

```text
SyncTranslate/
	main.py
	config.example.yaml
	config.yaml
	app/
		ui_main.py                 # 主視窗與整體流程控制
		session_controller.py      # session 啟停協調
		audio_router.py            # 新的雙向音訊路由中樞
		audio_input_manager.py     # 本地/遠端輸入管理
		asr_manager.py             # 串流 ASR 管理
		translator_manager.py      # 翻譯與 channel 對應
		tts_manager.py             # TTS 佇列與播放
		state_manager.py           # 執行中狀態與回授抑制
		audio_capture.py
		audio_playback.py
		audio_device_selection.py
		device_manager.py
		model_providers.py         # 目前只保留 EdgeTtsProvider
		schemas.py
		settings.py
		transcript_buffer.py
		debug_panel.py
		pages_audio_routing.py
		pages_diagnostics.py
		pages_io_control.py
		pages_live_caption.py
		pages_local_ai.py
		windows_volume.py
		local_ai/
			faster_whisper_engine.py
			streaming_asr.py
			vad_segmenter.py
			translation_stitcher.py
			lm_studio_client.py
			tts_factory.py
			healthcheck.py
			healthcheck_worker.py
			runtime_paths.py
```

## 執行資料流

目前不再使用舊的 `DirectionalPipeline`。新的流程由 `AudioRouter` 統一協調：

1. `AudioInputManager` 啟動本地或遠端音訊擷取。
2. `ASRManager` 為每個來源維護 `StreamingAsr` 實例。
3. `TranslatorManager` 根據來源方向呼叫對應的 `TranslationStitcher`。
4. `TranscriptBuffer` 同時保存 original / translated 字幕。
5. `TTSManager` 依通道將翻譯文字送到本地或遠端輸出。
6. `StateManager` 在遠端 TTS 播放期間暫停遠端 ASR，降低回授與自我收音。

## 健康檢查與診斷

- 健康檢查透過 `app.local_ai.healthcheck_worker` 子程序執行。
- 檢查項目：
	- ASR 模型是否可載入
	- LLM 後端是否可連線
	- `edge-tts` 套件是否可用
- 診斷頁可執行：
	- 儲存設定
	- 重新載入設定
	- 健康檢查
	- 預熱 + 檢查
	- 匯出診斷資訊

## 開發備註

- `requirements.txt` 是 `pyproject.toml` 的平面鏡像。
- 舊的 `piper_tts.py`、`pipeline_direction.py`、`router.py`、`asr_worker.py` 已移除。
- `model_providers.py` 目前只保留實際被使用的 Edge TTS provider，避免留下未接線的舊 provider 假象。
7. TTS worker 合成語音後交給 `AudioPlayback`

`SessionController` 依 mode 控制兩條 pipeline 的啟停。UI 端會持續拉取狀態更新字幕與除錯面板。

## 健康檢查與診斷

- 健康檢查透過 `app.local_ai.healthcheck_worker` 子程序執行，避免主 UI 被阻塞
- 檢查項目:
	- ASR 模型可用性
	- LLM 後端可連通性
	- TTS 引擎可用性 (Edge 套件)
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

