# 音訊 16kHz 問題修復記錄

本文整併原先 5 份英文分析文件：

- `README_16KHZ_ANALYSIS.md`
- `DEBUGGING_16KHZ_ISSUE.md`
- `DATA_FLOW_DIAGRAM.md`
- `RECOMMENDED_FIXES.md`
- `ROOT_CAUSE_ANALYSIS_DETAILED.md`

用途改為保留歷史脈絡、說明根因、記錄已落地修復，以及標示仍可持續重構的部分。

---

## 問題摘要

當使用者把 `config.runtime.sample_rate` 設為 `16000` 以最佳化 ASR 時，這個值曾經被一路傳到 remote speaker capture，導致 bridge 直接把原始 `48kHz` 的遠端音訊降採樣成 `16kHz` 再送回 Python 層。結果是：

- 遠端 passthrough 品質下降
- 本地播放端需要再升頻回 `48kHz`
- 日誌中會出現 `sample_rate=48000.0 requested_rate=16000.0` 這類不一致訊號

這不是 `_on_remote_audio_chunk()` 或 `InputManager.add_consumer()` 的邏輯錯誤，而是更上游的取樣率語意混用。

---

## 根因鏈

問題發生前的鏈路如下：

1. UI 讀取 `config.runtime.sample_rate`
2. `SessionService.start()` 把同一個 `sample_rate` 傳入 `AudioRouter`
3. `AudioRouter` 在 `_reconcile_single_source()` 內把同一個 rate 同時套到 local 與 remote source
4. `VirtualSpeakerSource.start()` 把這個 rate 傳給 `VirtualAudioBridgeClient.start_remote_input()`
5. bridge 收到 `16000` 後，直接把原始遠端喇叭音訊重採樣為 `16kHz`
6. Python 層後續收到的 remote audio 已經是降採樣後的結果

核心問題是：`runtime.sample_rate` 原本應該表達 ASR 管線取樣率，卻被當成 remote speaker capture rate 使用。

---

## 已落地修復

目前程式碼已經修正這個 blocker：

- `app/infra/audio/sources.py` 的 `VirtualSpeakerSource.start()` 會忽略傳入的 ASR sample rate，固定向 bridge 請求 `48000 Hz`
- `app/infra/audio/virtual_bridge_client.py` 內的 `_remote_input_sample_rate` 預設值也是 `48000`

這代表原本英文文件中最推薦的 quick fix 已經落地，remote 音訊不再跟著 ASR 設定一起被壓成 `16kHz`。

---

## 目前仍可持續改善的部分

### 1. 常數集中化

修復雖已落地，但 `48000` 目前仍是分散在 audio source / bridge client 的魔術數字。若要降低未來回歸風險，可以把它收斂成共用常數，例如：

- `BRIDGE_SAMPLE_RATE = 48000`
- `BRIDGE_CHANNELS = 2`
- `BRIDGE_BIT_DEPTH = 16`

### 2. 驅動格式文件化與驗證

Python 端目前仍以 `float32` 音訊在多數路徑流動，但 driver/bridge 實際期望的 PCM 格式需要更清楚地固定在文件與驗證腳本中，避免未來出現 mono/stereo 或 sample format 誤配。

### 3. 文件同步

舊英文分析文件描述的是「修復前」狀態，若保留容易讓後續閱讀者誤以為問題仍未處理，因此本次改為集中在此中文文件維護。

---

## 資料流示意

### 修復前

```text
config.runtime.sample_rate = 16000
  -> AudioRouter._sample_rate = 16000
  -> VirtualSpeakerSource.start(sample_rate=16000)
  -> bridge.start_remote_input(sample_rate=16000)
  -> bridge 將 48kHz 遠端音訊降採樣為 16kHz
  -> passthrough / 播放品質劣化
```

### 修復後

```text
config.runtime.sample_rate = 16000      # 僅供 ASR 使用
  -> AudioRouter 仍可用於 local ASR
  -> VirtualSpeakerSource.start(...) 強制 remote_sample_rate = 48000
  -> bridge.start_remote_input(sample_rate=48000)
  -> remote passthrough 維持 48kHz
```

---

## 與其他文件的關係

- 架構層面的固定規則，應回寫到 `docs/架構說明.md`
- 後續待辦與重構建議，應以 `docs/修改計畫_v2_產品化重構版.md` 為主
- 若再發生類似 regressions，應直接在此文件追加，不再拆成多份英文分析檔
