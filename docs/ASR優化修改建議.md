# ASR 優化修改建議

本文整理下一階段 ASR 架構優化方向，目標是把目前已經可用的串流 ASR pipeline，推進到更能自我判斷、自我降載、可量化回歸測試的狀態。

內容以「可落地實作」為原則，每一項都包含現況、目標、建議修改點、測試方式、驗收標準與風險。此文件是開發建議，不代表所有項目已經實作。

## 目前基礎

目前 ASR v2 主要路徑如下：

```text
AudioCapture
  -> AudioInputManager
  -> AudioRouter
  -> ASRManagerV2
  -> SourceRuntimeV2
  -> EndpointingRuntime
  -> FasterWhisper partial/final backend
  -> TranscriptPostProcessor
  -> TranscriptBuffer
  -> TranslationDispatcher / UI / TTS
```

關鍵檔案：

- `app/infra/asr/manager_v2.py`：local/remote runtime 建立、語言 profile 解析、診斷 stats。
- `app/infra/asr/worker_v2.py`：queue、VAD segment、partial/final decode、背壓處理。
- `app/infra/asr/streaming_policy.py`：依 backlog、VAD、segment 長度決定 emit partial/final。
- `app/infra/asr/language_profiles.py`：依語言覆蓋 beam、VAD、streaming、frontend 參數。
- `app/infra/asr/backend_v2.py`：建立 faster-whisper partial/final backend。
- `app/infra/asr/faster_whisper_adapter.py`：實際呼叫 faster-whisper。
- `app/application/transcript_postprocessor.py`：文字正規化、重複片段抑制、partial/final 保護。
- `tools/asr_benchmark/run_zh_preset_matrix.py`：中文 meeting/dialogue × belle/turbo benchmark。

目前已具備：

- local/remote 雙 ASR 通道。
- partial/final 分離。
- Silero/RMS endpointing。
- queue overflow 背壓。
- congested/degraded mode。
- final retry 與 partial fallback 保護。
- hallucination filter。
- benchmark 與 runtime smoke test。

下一階段不建議大改主流程，而是補上更精細的自動判斷、信心分數、回歸測試與診斷可視化。

## 優先順序

建議實作順序：

1. Auto 語言偵測後的動態 profile 切換
2. Final 優先的 ASR 負載控制
3. faster-whisper metadata 信心分數
4. 固定 ASR regression corpus
5. UI 顯示實際生效參數
6. 選配中文標點恢復

## 1. Auto 語言偵測後的動態 Profile 切換

### 現況

目前 `auto` 會以空語言送進 faster-whisper：

- `profile_selection.requested_asr_language_for_source()` 遇到 `auto` 回傳空字串。
- `backend_v2._sanitize_initial_prompt_for_language()` 遇到 auto 時會移除中文 prompt。
- `faster_whisper_adapter.FasterWhisperEngine._transcribe()` 只有在 `language` 非空時才傳入 faster-whisper。

這樣做安全，但缺點是：

- auto 偵測到中文後，仍不會自動套中文 profile。
- 中文 prompt、中文 VAD、中文 final history 不會生效。
- 中文標點可能比指定 `zh-TW` 時少。

### 目標

保留 auto 的安全性，但讓 runtime 在觀察到穩定語言後，自動切換到更合適的 language profile。

行為設計：

- auto 啟動時仍使用中性 profile。
- 若連續多個 final 都偵測為中文，切到中文 profile。
- 若連續多個 final 偵測為英文或其他語言，切到對應非中文 profile。
- 中英混雜時不頻繁切換。
- 切換只影響 ASR profile，不改 UI 上使用者選的 `auto`。

### 建議實作

新增檔案：

- `app/infra/asr/auto_language_state.py`

建議 dataclass：

```python
@dataclass(slots=True)
class AutoLanguageState:
    requested_language: str = ""
    effective_language: str = ""
    last_detected_language: str = ""
    stable_family: str = "auto"
    chinese_streak: int = 0
    non_chinese_streak: int = 0
    mixed_count: int = 0
    last_switch_ms: int = 0
```

建議判斷規則：

- 只在使用者 ASR 語言為 `auto` 或空字串時啟用。
- final event 才更新，不用 partial 更新，避免抖動。
- 以 `detected_language` 加上文字 script ratio 判斷：
  - CJK 字元比例大於 0.35，視為中文候選。
  - Latin token 比例高且 CJK 很低，視為非中文候選。
- 中文連續 2 或 3 次 final 後切到 `zh-TW` 或 `zh`。
- 非中文連續 2 或 3 次 final 後切到 `en` 或空語言。
- 切換 cooldown 建議 20-30 秒。
- 若單句太短，例如少於 4 個中文字或少於 3 個英文單詞，不納入切換。

修改點：

- `ASRManagerV2` 增加每 source 一個 `AutoLanguageState`。
- `_asr_language_for_source(source)` 拆成兩層：
  - 使用者 requested language。
  - runtime effective language。
- `SourceRuntimeV2` final event 回到 manager 時，manager 更新 auto language state。
- 當 effective language 改變時，安全重建該 source 的 runtime。

安全重建策略：

1. 收到 final event 後更新 state。
2. 若需要切換，標記 `pending_rebuild[source] = True`。
3. 下一次送入 chunk 前，如果該 source 不在 segment 中或 queue 很低，停止舊 runtime 並重建。
4. 避免在 ASR worker callback 裡直接 stop/join 同一個 worker thread。

建議新增 manager 方法：

```python
def _requested_language_for_source(self, source: str) -> str:
    ...

def _effective_language_for_source(self, source: str) -> str:
    ...

def _observe_detected_language(self, source: str, event: V2RuntimeEvent) -> None:
    ...

def _maybe_rebuild_runtime_for_language_switch(self, source: str) -> None:
    ...
```

### 測試

新增測試：

- `tests/test_asr_auto_language_switch.py`

測試案例：

- auto 初始使用 non_chinese/auto profile。
- 連續中文 final 後 effective language 變成 `zh-TW`。
- 中英交錯時不切換。
- 太短句不觸發切換。
- cooldown 內不重複切換。
- 切換後 stats 顯示 requested=`auto`，effective=`zh-TW`。

Benchmark：

```powershell
uv run python tools/asr_benchmark/run_zh_preset_matrix.py --models turbo --modes meeting,dialogue --speed 8 --queue-maxsize 256
```

另建議新增 auto replay：

```powershell
uv run python tools/asr_benchmark/streaming_sim.py --source remote --language auto --audio <zh_audio.wav> --reference <zh_ref.srt>
```

### 驗收標準

- auto 中文素材在 2-3 句 final 後使用中文 effective profile。
- 中文 auto 的平均準確率接近指定 `zh-TW`。
- 中英混雜素材不頻繁切換。
- runtime rebuild 不造成 ASR thread crash。
- queue dropped chunks 不增加。

### 風險

- 語言切換太積極會造成中英混雜素材錯路由。
- runtime 重建時機不當可能造成短暫漏音。
- 如果 detected language 不穩，需要 script ratio 輔助。

## 2. Final 優先的 ASR 負載控制

### 現況

目前高壓時已有：

- queue 超過 70% 時暫停 partial。
- overflow 時 drop oldest chunk。
- `StreamingPolicy` 在 congested/degraded 時 suppress partial。
- backlog 高時可以 force final。

但仍有問題：

- partial decode 和 final decode 共享 GPU/model lock。
- 壓力高時，即使 partial 頻率降低，仍可能吃掉 final 的時機。
- final 是使用者和翻譯最依賴的結果，應該優先保護。

### 目標

在 queue 或 decode latency 壓力上升時，建立「final priority」模式：

- 停止 partial decode。
- 只保留 endpointing 與 final decode。
- 優先縮短 segment，讓 final 更快釋放。
- 壓力恢復後再開 partial。

### 建議實作

修改檔案：

- `app/infra/asr/streaming_policy.py`
- `app/infra/asr/worker_v2.py`
- `app/infra/config/schema.py`
- `app/ui/pages/diagnostics_page.py`

新增 runtime config：

```yaml
runtime:
  asr_final_priority_enabled: true
  asr_final_priority_queue_ratio: 0.45
  asr_final_priority_latency_ms: 1800
  asr_final_priority_recover_queue_ratio: 0.15
  asr_final_priority_recover_after_ms: 8000
```

新增狀態：

```python
final_priority_active: bool
final_priority_reason: str
final_priority_since_ms: int
```

觸發條件：

- queue ratio >= 0.45。
- 最近 final latency p95 > 1800 ms。
- dropped_chunks 增加。
- GPU/shared model lock 等待時間過高，若之後有量測。

行為：

- `StreamingPolicy.decide()` 若 final priority active：
  - `emit_partial = False`
  - `suppress_partial = True`
  - 降低 `adaptive_length_limit_ms`
  - queue 高時更早 `force_final`
- `SourceRuntimeV2._emit_partial()` 入口也應再次檢查，避免 race condition。
- stats 顯示 `final_priority_active`。

恢復條件：

- queue ratio <= 0.15。
- 最近 8 秒沒有 dropped chunk。
- final latency 回落。

### 測試

新增/擴充：

- `tests/test_asr_streaming_and_profiles.py`

測試案例：

- queue pressure 進入 final priority。
- final priority 時不 emit partial。
- final priority 時仍可 emit final。
- 壓力解除後 partial 恢復。
- dropped chunks 不會使狀態永遠卡住。

壓力 replay：

```powershell
uv run python tools/asr_benchmark/run_zh_preset_matrix.py --models turbo --modes meeting --speed 12 --queue-maxsize 128
```

### 驗收標準

- 高壓時 dropped chunks 下降。
- final latency 不惡化。
- partial 數量下降但 final 數量穩定。
- UI 能看到 final priority 狀態。

### 風險

- partial 變少，使用者即時感降低。
- 如果 threshold 太低，會常駐 final-only。
- 需要透過 benchmark 找到 RTX 3060 合理預設。

## 3. faster-whisper Metadata 信心分數

### 現況

目前 `FasterWhisperEngine._transcribe()` 只回傳：

```python
TranscribeResult(text: str, detected_language: str)
```

但 faster-whisper segment 通常包含：

- `avg_logprob`
- `no_speech_prob`
- `compression_ratio`
- `start`
- `end`
- `words` 或 segment timestamps

這些資訊目前沒有用於判斷低信心、幻覺或是否重跑 final。

### 目標

把 segment metadata 收進結果，讓後處理可以更聰明：

- 低信心 final 可重跑或丟棄。
- 高 no_speech_prob 的短句可抑制。
- compression_ratio 過高可視為重複 loop。
- segment 時間可協助判斷尾音是否被裁切。

### 建議實作

修改檔案：

- `app/infra/asr/faster_whisper_adapter.py`
- `app/infra/asr/backend_v2.py`
- `app/infra/asr/worker_v2.py`
- `app/infra/asr/transcript_validator_v2.py`
- `app/infra/asr/_hallucination_filter.py`

擴充 dataclass：

```python
@dataclass(slots=True)
class SegmentMetadata:
    text: str
    start: float
    end: float
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None

@dataclass(slots=True)
class TranscribeResult:
    text: str
    detected_language: str
    segments: tuple[SegmentMetadata, ...] = ()
    avg_logprob: float | None = None
    max_no_speech_prob: float | None = None
    max_compression_ratio: float | None = None
```

聚合規則：

- `avg_logprob`：依 segment 時長加權平均。
- `max_no_speech_prob`：取最大值。
- `max_compression_ratio`：取最大值。

建議信心策略：

- final 且 `max_no_speech_prob > 0.75` 且文字很短：抑制。
- `max_compression_ratio > 2.6`：標記 loop risk。
- `avg_logprob < -1.2`：低信心，若 partial 比 final 更完整，可偏向 partial。
- 中文短 filler 不因低信心直接丟棄，要搭配音訊能量與文字長度。

### 測試

新增：

- `tests/test_faster_whisper_metadata.py`
- `tests/test_asr_confidence_policy.py`

使用 fake segment 物件測：

- metadata 正確讀取。
- 沒有 metadata 時相容舊版 faster-whisper。
- 高 no_speech_prob 短句被抑制。
- 高 compression_ratio 觸發 loop risk。
- 低 avg_logprob 不會誤殺正常中文長句。

### 驗收標準

- 現有 377+ 測試維持通過。
- fake backend 可覆蓋 metadata。
- benchmark 中 hallucination/loop 減少。
- 不造成正常短句如「好」「謝謝」大量消失。

### 風險

- faster-whisper 不同版本 metadata 欄位可能不同。
- threshold 需要實測調整。
- 太依賴 logprob 可能誤殺口音、雜訊或遠距音訊。

## 4. 固定 ASR Regression Corpus

### 現況

已有 benchmark 工具與 downloads 素材，但尚未形成正式 regression corpus。

問題：

- 每次調參缺少固定門檻。
- belle/turbo、meeting/dialogue 的比較結果不易長期追蹤。
- 不同素材類型的退步可能被平均值掩蓋。

### 目標

建立固定 ASR corpus，讓每次 ASR 修改都能回答：

- 準確率有沒有退步？
- dropped chunks 有沒有增加？
- 重複率有沒有增加？
- 哪種素材退步？

### 建議目錄

```text
downloads/asr_regression/
  manifest.yaml
  audio/
    zh_news_01.wav
    zh_meeting_01.wav
    zh_dialogue_01.wav
    zh_novel_01.wav
    zh_short_video_01.wav
    zh_noise_01.wav
    en_meeting_01.wav
  refs/
    zh_news_01.txt
    zh_meeting_01.txt
    ...
```

`manifest.yaml` 範例：

```yaml
samples:
  - id: zh_news_01
    language: zh-TW
    category: news
    audio: audio/zh_news_01.wav
    reference: refs/zh_news_01.txt
    min_accuracy:
      meeting_turbo: 0.86
      dialogue_turbo: 0.83
    max_dropped_chunks: 0
    max_repetition_ratio: 0.02
```

素材類型建議：

- 新聞：句子清楚、標準語速。
- 會議：多人、停頓、不完整句。
- 對話：短句、輪替快。
- 小說/有聲書：長句、敘事語氣。
- 短影音：背景音樂、快語速。
- 雜訊：風扇、鍵盤、房間回音。
- 男女聲：至少各一組。

### 建議實作

新增工具：

- `tools/asr_benchmark/run_regression_corpus.py`

功能：

- 讀取 `manifest.yaml`。
- 支援 `--models belle,turbo`。
- 支援 `--modes meeting,dialogue`。
- 支援 `--speed` 壓力倍率。
- 輸出 `summary.json`、`summary.md`。
- 若低於門檻，exit code 非 0。

建議指令：

```powershell
uv run python tools/asr_benchmark/run_regression_corpus.py --manifest downloads/asr_regression/manifest.yaml --output-dir downloads/benchmark_results/asr_regression_manual
```

CI/本地快速版：

```powershell
uv run python tools/asr_benchmark/run_regression_corpus.py --manifest downloads/asr_regression/manifest.yaml --quick
```

### 測試

新增：

- `tests/test_asr_regression_manifest.py`
- `tests/test_asr_regression_runner.py`

測試：

- manifest schema 驗證。
- 不存在的 audio/ref 會報錯。
- summary 聚合正確。
- 門檻失敗時 exit code 非 0。

### 驗收標準

- 每次 ASR 調參都能產生可比較 summary。
- 至少 6 類中文素材。
- summary 顯示每組模式/模型準確率、drop、重複率。
- 有 quick mode 可在合理時間內跑完。

### 風險

- 素材授權要注意。
- 長音檔 benchmark 很耗時。
- 參考字幕品質會影響評分可信度。

## 5. UI 顯示實際生效參數

### 現況

設定 UI 顯示的是 config 值，但 runtime 實際使用值可能經過：

- language profile 覆蓋。
- endpoint profile 覆蓋。
- auto 語言解析。
- queue/degradation 動態調整。
- CUDA fallback。

因此使用者看到的設定不一定等於 ASR 正在使用的有效參數。

### 目標

診斷頁直接顯示 effective runtime params。

建議顯示：

- requested ASR language
- effective ASR language
- detected language
- language family
- configured model
- effective model
- beam size
- final beam size
- condition_on_previous_text
- final_condition_on_previous_text
- VAD backend
- neural threshold / RMS threshold
- min silence
- speech pad
- partial interval
- final history seconds
- soft final ms
- endpoint profile
- enhancement enabled
- queue size / max
- dropped chunks
- degradation level
- final priority active
- avg logprob / no speech prob / compression ratio，若 metadata 已實作

### 建議實作

修改：

- `ASRManagerV2.stats()`
- `SourceRuntimeV2.stats()`
- `app/ui/pages/diagnostics_page.py`
- `app/ui/main_window.py`

建議 stats 增加：

```python
"effective": {
    "requested_language": "...",
    "effective_language": "...",
    "detected_language": "...",
    "language_family": "...",
    "model": "...",
    "beam_size": 1,
    "final_beam_size": 4,
    "vad": {...},
    "streaming": {...},
    "endpoint_profile": "...",
    "frontend": {...},
}
```

UI 呈現建議：

- 診斷頁每個 channel 一個區塊。
- 用短行顯示，例如：

```text
remote: lang auto -> zh-TW, model large-v3-turbo, beam 1/4, profile meeting_room
queue 0/256, drop 0, deg normal, final-priority off
vad silero th=0.45 silence=520 pad=320, partial=460ms final_history=20s
```

### 測試

修改/新增：

- `tests/test_ui_behavior.py`
- `tests/test_diagnostics_export.py`
- `tests/test_asr_streaming_and_profiles.py`

測試：

- stats 包含 effective 欄位。
- diagnostics UI 顯示 effective model/language/profile。
- disabled ASR 不崩潰。
- auto dynamic switch 後顯示 requested/effective 差異。

### 驗收標準

- 使用者不看 log 也能知道 ASR 實際跑什麼。
- queue/drop/degradation/final priority 狀態明確。
- packaged runtime 也能輸出相同診斷資訊。

### 風險

- UI 資訊太多會變雜。
- 建議預設顯示摘要，展開才看完整 effective params。

## 6. 選配中文標點恢復

### 現況

ASR 原文目前忠實顯示 faster-whisper 輸出。若 auto 模式或某些模型沒有輸出中文標點，UI 原文就沒有標點。

翻譯結果通常會有標點，因為 LLM 會自然整理句子，但 ASR 原文不會額外補標點。

### 目標

提供「顯示用標點整理」選項：

- 不改 ASR 原始事件。
- 不影響翻譯 pipeline。
- 只影響 UI 顯示或匯出。
- 可關閉。

### 建議實作

新增 config：

```yaml
runtime:
  asr_display_punctuation_enabled: false
  asr_display_punctuation_languages: zh
  asr_display_punctuation_mode: lightweight
```

建議模式：

- `off`：完全不處理。
- `lightweight`：規則型，只補句尾標點與簡單停頓。
- `llm`：未來可選，用本地 LLM 做標點恢復，但不建議第一版做。

第一版建議只做 lightweight。

新增檔案：

- `app/application/display_punctuation.py`

規則建議：

- 只處理 CJK。
- 若文字已含 `。！？；，、`，不強行改。
- final 且長度大於 8 個中文字、結尾無標點，補 `。`。
- 若包含明顯問句詞：
  - `嗎`
  - `呢`
  - `是不是`
  - `為什麼`
  - `怎麼`
  結尾可補 `？`。
- 不對 partial 補標點，避免畫面抖動。
- 不對短 filler 補標點，例如 `好`、`嗯`、`對`。

接入點：

- UI 顯示前套用，而不是 ASR event 進入 translation 前。
- 建議在 `LiveCaptionPage` 或 display adapter 層處理。
- 匯出時可選擇匯出原始文字或顯示文字。

### 測試

新增：

- `tests/test_display_punctuation.py`

測試：

- 中文 final 無標點補 `。`。
- 問句補 `？`。
- 已有標點不重複補。
- partial 不補。
- 英文不補。
- 短 filler 不補。

### 驗收標準

- 預設關閉，保持現狀。
- 開啟後只改善顯示，不影響翻譯結果。
- 不產生明顯錯誤標點。

### 風險

- 規則型標點不可能完全準。
- 新聞/會議長句可能需要逗號，但 lightweight 不應過度猜測。
- 若使用者以 ASR 原文做嚴格逐字稿，應保留原始匯出。

## 分階段開發建議

### Phase 1：可觀測與安全基礎

優先做：

- UI effective params。
- ASR metadata dataclass，但先只收集不影響行為。
- regression corpus manifest 與 runner。

原因：

- 先讓系統能量測，再改自動策略。
- 可降低後續動態切換和 final priority 的調參風險。

### Phase 2：負載與品質策略

接著做：

- final priority。
- metadata-based confidence policy。
- benchmark 門檻。

原因：

- 這些會影響即時行為，需要 corpus 保護。

### Phase 3：Auto 智慧切換與顯示整理

最後做：

- auto dynamic profile switch。
- display punctuation。

原因：

- auto switch 涉及 runtime rebuild，風險較高。
- display punctuation 是體驗層，可獨立開關。

## 建議測試矩陣

每次 ASR 修改至少跑：

```powershell
uv run python -m compileall -q app main.py tools tests
uv run pytest
```

ASR 核心測試：

```powershell
uv run pytest tests/test_asr_streaming_and_profiles.py tests/test_asr_postprocess_v2.py tests/test_transcript_postprocessor.py tests/test_transcript_service.py
```

中文 benchmark：

```powershell
uv run python tools/asr_benchmark/run_zh_preset_matrix.py --models turbo --modes meeting,dialogue --speed 8 --queue-maxsize 256
```

高壓 benchmark：

```powershell
uv run python tools/asr_benchmark/run_zh_preset_matrix.py --models turbo --modes meeting --speed 12 --queue-maxsize 128
```

packaged smoke：

```powershell
uv run python tools/runtime_smoke/run_runtime_smoke.py --config config.yaml --packaged-onedir .\dist\SyncTranslate-onedir
```

## 建議驗收指標

品質：

- 中文 benchmark 平均 accuracy 不低於目前 baseline。
- 重複率 `repetition_ratio <= 0.02`。
- dropped chunks 維持 0 或低於 baseline。

延遲：

- final latency 不因新策略明顯增加。
- 高壓時 final latency 優先於 partial 互動性。

穩定：

- ASR worker 不 crash。
- runtime rebuild 不漏掉大量音訊。
- auto mode 不頻繁切換。

可觀測：

- 診斷頁能看出 requested/effective language。
- 能看出模型、profile、queue、drop、degradation、final priority。
- benchmark summary 可保存並比較。

## 不建議做的事

短期不建議：

- 直接用 LLM 改寫 ASR 原文再送翻譯。
- 預設開啟中文標點恢復。
- auto 模式一偵測到中文就立刻切 profile。
- 大幅重構 `SourceRuntimeV2`。
- 為了單一素材手動調死全域參數。

理由：

- ASR 原文應保持可信。
- auto 誤切會比沒有標點更糟。
- 目前主架構已可用，應用 benchmark 和診斷資訊穩健推進。

## 最終建議

最推薦的落地路線是：

1. 先做 effective params 與 metadata 收集。
2. 建立 regression corpus，固定 baseline。
3. 做 final priority，解決高壓場景。
4. 用 metadata 改善 hallucination 與低信心 final。
5. 做保守版 auto dynamic profile switch。
6. 最後提供預設關閉的顯示用中文標點整理。

這樣可以讓 ASR 架構從「可用且已調參」進一步變成「可量測、可自我調整、可長期維護」。
