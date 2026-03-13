# SyncTranslate

SyncTranslate 是 Windows 即時口譯桌面工具。

系統會擷取本地與遠端音訊，做串流 ASR，透過 LM Studio 翻譯，再用 Edge TTS 播放翻譯語音。UI 以 PySide6 建置，提供音訊路由、即時字幕、參數設定與診斷工具。

## 核心能力

- 三種方向模式：遠端到本地、本地到遠端、雙向
- faster-whisper 串流 ASR（含 VAD）
- LM Studio 本地翻譯
- Edge TTS 本地與遠端雙輸出通道
- partial/final 字幕與同句 revision 更新
- session 啟停狀態機與診斷報告輸出

## 參數頁新架構

- 左上：語音辨識 ASR
  - 兩個模型欄位：中文、英文
  - 共用：引擎、裝置（僅 cuda/cpu）、精度、語言模式（Auto/鎖定）
  - 獨立：ASR 佇列、Beam、延續前文、中文 fallback / 英文 fallback、中文 No Speech / 英文 No Speech、VAD、串流間隔、歷史長度、RMS 門檻等其他參數
- 右上：翻譯模型 LLM
  - 兩個模型欄位：中翻英、英翻中
  - 共用：後端、服務位址、重新載入模型
  - 獨立：LLM 佇列、逾時、溫度、Top P、最大輸出 Token、Repeat Penalty、Stop Tokens、是否啟用上下文拼接、觸發 Token、上下文項數
  - 建議預設：中翻英 max tokens=96，英翻中 max tokens=160
  - 建議預設：中翻英 stop=`</target>,Translation:`，英翻中 stop=`</target>,翻譯:`
- 左下：系統執行
  - 新句取消舊語音
  - 取消策略（取消所有未播 / 只取消舊句）
  - TTS 最大等待(ms)
  - TTS 最大字數
  - Streaming tokens
  - 最大 pipeline 延遲(ms)
- 右側 LLM 下方：TTS 模型設定，再下方輸出通道覆寫（中文/英文共用欄位）
  - 中文欄位只顯示 zh- Edge 聲線，英文欄位只顯示 en- Edge 聲線
  - 獨立：TTS 佇列、引擎、聲線、取樣率、Noise W

## 語言方向到模型的自動映射

系統不再把本地/遠端硬綁到固定 ASR/LLM 設定，而是依語言與方向自動映射：

- ASR 映射
  - 來源語言是中文/非英文：使用中文 ASR 模型
  - 來源語言是英文：使用英文 ASR 模型
- LLM 映射
  - 中文到英文：使用中翻英 LLM 模型
  - 英文到中文：使用英翻中 LLM 模型

這表示本地或遠端都可以依當前語言方向，動態選到正確的 ASR 與 LLM profile。

## 設定檔儲存結構

主要設定檔是 config.yaml。

新版除保留相容欄位 asr 與 llm 之外，新增以下正式分流欄位：

- asr_channels
  - chinese：中文 ASR profile
  - english：英文 ASR profile
- llm_channels
  - zh_to_en：中翻英 LLM profile
  - en_to_zh：英翻中 LLM profile

runtime.config_schema_version 已升級到 3。

說明：

- asr_channels.chinese / asr_channels.english 對應中文/英文 ASR 設定
- llm_channels.zh_to_en / llm_channels.en_to_zh 對應中翻英/英翻中 LLM 設定
- tts_channels.chinese / tts_channels.english 對應中文/英文輸出通道覆寫
- chinese_tts / english_tts 為兩個通道的 fallback 設定（分別預設中文 / 英文）
- runtime.asr_queue_maxsize_chinese / runtime.asr_queue_maxsize_english 對應中文/英文 ASR 佇列
- runtime.llm_queue_maxsize_zh_to_en / runtime.llm_queue_maxsize_en_to_zh 對應中翻英/英翻中 LLM 佇列
- runtime.tts_queue_maxsize_chinese / runtime.tts_queue_maxsize_english 對應中文/英文 TTS 佇列
- runtime.tts_cancel_pending_on_new_final 控制新 final 句子是否取消舊語音
- runtime.tts_cancel_policy 控制取消策略（全部未播 / 僅舊句）
- runtime.tts_max_wait_ms 控制 TTS 佇列任務最長等待時間，超時會丟棄
- runtime.tts_max_chars 控制 TTS 最大句長，超過會自動切句
- runtime.llm_streaming_tokens 控制翻譯提早觸發的 token 門檻（降低延遲）
- runtime.max_pipeline_latency_ms 控制最大可接受延遲，超過會丟棄 backlog
- asr_channels.*.temperature_fallback 控制 Whisper temperature fallback
- asr_channels.*.no_speech_threshold 控制 Whisper no-speech 判定門檻
- llm_channels.*.max_output_tokens / repeat_penalty / stop_tokens 控制翻譯輸出長度與重複/停止詞

若舊版設定沒有 asr_channels 或 llm_channels，啟動時會自動遷移並補齊。

## 專案結構

- 啟動與主視窗
  - main.py
  - app/ui_main.py
- 組裝與執行期
  - app/app_bootstrap.py
  - app/runtime_facade.py
  - app/config_apply_service.py
  - app/diagnostics_service.py
- 即時管線
  - app/audio_router.py
  - app/audio_input_manager.py
  - app/asr_manager.py
  - app/translator_manager.py
  - app/tts_manager.py
  - app/state_manager.py
  - app/transcript_buffer.py
- 本地 AI 實作
  - app/local_ai/faster_whisper_engine.py
  - app/local_ai/streaming_asr.py
  - app/local_ai/translation_stitcher.py
  - app/local_ai/lm_studio_client.py
  - app/local_ai/healthcheck.py

## 需求

- Windows 10/11
- Python 3.11+
- 建議使用 uv
- 可用音訊裝置（sounddevice）
- LM Studio 啟動且已載入模型

## 開發啟動

安裝依賴：

```powershell
uv sync --extra local --group dev
```

啟動 GUI：

```powershell
uv run python .\main.py
```

快速自檢：

```powershell
uv run python .\main.py --check
```

## 診斷與記錄

- 執行事件：logs/runtime_events.log
- 崩潰記錄：logs/runtime_crash.log
- Session 報告：logs/session_reports

## 打包

```powershell
uv run pyinstaller .\SyncTranslate-onedir.spec --noconfirm --clean
uv run pyinstaller .\SyncTranslate-onefile.spec --noconfirm --clean
```

## 常見問題

- 字幕不更新：先確認音訊路由與輸入電平
- 有字幕沒聲音：檢查 speaker_out 與 meeting_out
- 翻譯延遲高：降低上下文項數或更換較小 LLM 模型
- ASR 負載高：改用較小模型，或確認使用 CUDA
