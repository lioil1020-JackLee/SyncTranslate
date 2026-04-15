# ASR 重構藍圖

## 目前狀態

ASR 重構已進入第二階段：

- `ASR v2` 已建立並成為主要路線
- backend 已改成依通道語言自動路由
- 中文鏈路已接入 `FunASR + FSMN-VAD`
- 非中文與 `auto` 仍走 `faster-whisper`
- lazy registry / runtime stats / diagnostics 已落地

也就是說，這份文件不再描述「是否要重構」，而是描述：

- 已完成什麼
- 還沒完成什麼
- 舊版何時刪除

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

- 中文 -> `funasr_v2`
- 非中文 / `auto` -> `faster_whisper_v2`
- `none` -> disabled

### 3. 中文鏈路

已完成：

- `FunASR` backend
- `FSMN-VAD`
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

### 5. UI 同步

已完成：

- Local AI 頁與 Live Caption 頁的 backend 提示
- device / effective device 認知同步
- 舊式 engine 選項淡出

## 尚未完成

### 1. legacy 清理

雖然 `ASR v2` 已成為主線，但 `legacy` 仍存在。

下一步要做：

- 清點所有 legacy 專用 config
- 確認沒有任何 runtime 還會回落依賴 legacy
- 刪除不再使用的 legacy ASR 實作與測試

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

## 刪除 legacy 的條件

只有在下面條件都達成後，才建議刪除 legacy：

1. `ASR v2` 成為預設且已穩定使用
2. 中文與非中文測試都能走正確 backend
3. session report 能穩定輸出完整 stats
4. UI 與 config 不再依賴 legacy 概念
5. 真實中文場景下，沒有再出現「幾乎只剩 2-3 句」這類根本性錯誤

## 下一階段建議

### 高優先

- 持續做中文新聞 / 會議語音實測
- 把 diagnostics 再做得更直觀
- 釐清 CUDA 環境與 `effective device`

### 中優先

- 整理並刪除 legacy config / dead code
- 補更多 ASR v2 integration tests

### 低優先

- 更進一步的 speaker diarization
- 更強的 final correction / hotword / domain lexicon

## 結論

重構方向已經定下來：

- `ASR v2` 是主線
- 中文用 FunASR
- 非中文用 faster-whisper
- UI、diagnostics、config 都以這個模型為準

接下來的工作重點，不再是「要不要重構」，而是「把 v2 驗收穩、把 legacy 刪乾淨」。
