# ASR 重構藍圖

## 目前狀態

ASR 重構已進入收尾與實測調校階段：

- `ASR v2` 已建立並成為主要路線
- backend 已改成依通道語言自動路由
- 中文鏈路已接入 `faster-whisper + silero-vad`
- 非中文與 `auto` 仍走 `faster-whisper`
- lazy registry / runtime stats / diagnostics 已落地

也就是說，這份文件不再描述「是否要重構」，而是描述：

- 已完成什麼
- 還沒完成什麼
- 還有哪些觀測與實測工具要補強

## 已完成

### 1. 新入口

已完成：

- `app/infra/asr/factory.py`
- `app/infra/asr/manager_v2.py`
- `app/infra/asr/worker_v2.py`
- `app/infra/asr/contracts.py`

### 2. 語言自動路由

已完成：

- `backend_resolution.py`

規則：

- 中文 -> `faster_whisper_v2`
- 非中文 / `auto` -> `faster_whisper_v2`
- `none` -> disabled

### 3. 中文鏈路

已完成：

- `faster-whisper` backend
- `silero-vad`
- lazy registry
- device fallback
- runtime stats 暴露

### 4. Diagnostics

已完成輸出：

- `resolved_backend`
- `resolved_language_family`
- `backend_resolution_reason`
- `device_effective`
- `model_init_mode`
- `init_failure`
- `endpoint_signal.pause_ms`
- `speech_started_count / soft_endpoint_count / hard_endpoint_count`
- `postprocessor.final.rejected_count`
- `postprocessor.final.last_rejection_reason`

### 5. UI 同步

已完成：

- Local AI 頁與 Live Caption 頁的 backend 提示
- device / effective device 認知同步
- 舊式 engine 選項淡出

### 6. Phase 1-4 商品化重構（2026-04-16）

已完成：

**Phase 1：ASR 後處理與可觀測性**
- `TranscriptPostProcessor`（partial 穩定前綴、text normalization、glossary）
- `GlossaryStore` / `GlossaryLoader`（YAML / JSON 術語表）
- `RuntimeLogger`（背景 jsonl 結構化日誌）
- `UtteranceLatency` / `PipelineMetrics` / `MetricsCollector`

**Phase 2：StreamingPolicy 與 ASR Profiles**
- `StreamingPolicy`（normal / congested / degraded 三段降級保護）
- 6 個內建 endpoint profile（default / meeting_room / headset / noisy_environment / max_accuracy / low_latency）

**Phase 3：音訊前處理鏈與 Benchmark 工具**
- `AudioFrontendChain`（HighpassStage / LoudnessStage / NoiseReductionStage / MusicSuppressionStage）
- `tools/asr_benchmark/`（離線 CER / WER benchmark CLI）
- `tools/youtube_srt/`（YouTube SRT 基準測試工具）

**Phase 4：AudioRouter 責任分離 & UI Controller 化**
- `ASREventProcessor` / `TranslationDispatcher` / `TTSDispatcher` / `PipelineMetricsCollector`
- `SessionActionController` / `LiveCaptionRefreshController` / `ConfigHotApplyController` / `HealthcheckController`

**Config 擴充（全向後相容）**
- `enable_postprocessor` / `enable_partial_stabilization`
- `glossary_enabled` / `glossary_path` / `glossary_apply_on_partial` / `glossary_apply_on_final`
- `enable_structured_logging` / `runtime_log_format`
- `asr_profile_local` / `asr_profile_remote`
- `streaming_profile_local` / `streaming_profile_remote`
- `degradation_policy_enabled`

**faster-whisper 專屬參數化**
- 從共用 ASR 欄位拆出 `faster-whisper.*` 子區塊
- `use_itn`、`batch_size_s_offline`、`batch_size_s_online`、`online_encoder_chunk_look_back`、`online_decoder_chunk_look_back`

## 尚未完成

### 1. v2 實測回放工具

單一路徑 v2 已落地，但仍缺更方便的壞案例回放與比對工具。

下一步要做：

- 離線回放一段音訊並輸出 frontend / endpointing / partial / final / validator 節奏
- 更快重現「畫面體感差」但不容易定位的案例

### 2. 中文鏈路實測穩定度

目前仍需要更多真實驗收，尤其是：

- 中文新聞長句
- 雙通道同時中文發話
- VoiceMeeter B1 / B2 同源輸入
- CPU 與 CUDA 的差異

### 3. ASR final correction

相關欄位與模組已存在，但仍需持續驗證：

- final-only 校正是否真的比 raw ASR 更穩
- 是否有過度改寫
- 專有名詞與近音字是否有明顯改善

### 4. speaker diarization

目前仍視為實驗性功能：

- 先確保單一講者不亂跳
- 再決定是否重新啟用預設

## 單一路徑 v2 現況

目前實際 runtime 已統一為 `ASR v2`，剩下的只是不影響執行的相容 helper 與舊設定正規化。

## 下一階段建議

### 高優先

- 持續做中文新聞 / 會議語音實測
- 把 diagnostics / replay 做得更直觀
- 釐清 CUDA 環境與 `effective device`

### 中優先

- 持續清理相容 helper 與 dead code
- 補更多 ASR v2 integration tests

### 低優先

- 更進一步的 speaker diarization
- 更強的 final correction / hotword / domain lexicon

## 結論

重構方向已經定下來：

- `ASR v2` 是唯一執行主線
- 中文用 faster-whisper
- 非中文用 faster-whisper
- UI、diagnostics、config 都以這個模型為準

接下來的工作重點，不再是「要不要重構」，而是「把 v2 實測驗收穩、把觀測與回放工具補齊」。

## faster-whisper 專屬參數化（2026-04-16）

已將 faster-whisper 從共用 ASR 欄位拆出專屬參數：
- 辨識推論：`use_itn`、`batch_size_s_offline`、`batch_size_s_online`
- online 模式：`online_chunk_size`、`online_encoder_chunk_look_back`、`online_decoder_chunk_look_back`
- 後處理抑制：`suppress_low_confidence_short`、`short_text_max_chars`、`min_speech_ratio_for_short_text`、`low_peak_threshold`
- benchmark：`benchmark_window_ms`、`benchmark_overlap_ms`

並新增 `tools/asr_benchmark/run_benchmark.py` 進行中文調參回圈，輸出最佳候選與完整排序結果 JSON。


