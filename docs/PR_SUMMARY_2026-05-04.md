# SyncTranslate 變更摘要（2026-05-04）

## 範圍
- 完成 docs/修改計畫.md 的 P0 與 P1 實作。
- 整併與清理過時翻譯修復文件，將工具腳本移至 tools/runtime_setup。
- 補齊對應測試並完成回歸驗證。

## 主要變更

### 1. 穩定性與可觀測性
- ASR 與 translation dispatcher 線程加上頂層異常捕獲與日誌。
- translation dispatcher 的靜默例外改為 warning + 堆疊資訊。
- healthcheck 子程序錯誤保留更多摘要上下文，並記錄完整 stderr。

### 2. 記憶體與佇列保護
- ASR segment chunk 加入上限保護，避免長段落無限成長。
- ASR queue overflow 加入節流 warning（每 10 次一次）。
- TTS stop 改為 deadline join，並對逾時未結束線程發出 warning。

### 3. 安全性
- Windows 音量控制加入 flow 白名單與 device name 安全字元驗證。
- 驗證失敗採 warning + 跳過，避免啟動流程中斷。

### 4. 配置與預設值治理
- 新增 app/domain/constants.py 的 ASR/TTS 預設值常數並套用到核心路徑。
- _schema_parser 增加關鍵 runtime 欄位夾限與 display_partial_strategy 合法值回退。
- TranslatorManager 新增 update_config 並清空語言快取。

### 5. 文件與工具
- 刪除過時文件：SETUP_GUIDE.md、TRANSLATION_* 修復報告/快速修復/技術修復。
- 生成並更新 docs/修改計畫.md（含完成勾選）。
- 更新 docs/快速安裝手冊.md。
- 工具腳本移動到 tools/runtime_setup。

## 測試
- 全量測試：407 passed（pytest tests/ -q）。
- 新增/更新測試涵蓋：
  - 無效 runtime 配置夾限
  - healthcheck 摘要行為
  - session 啟動失敗狀態轉移
  - translator update_config 清快取

## 風險與後續
- 核心模組 coverage >= 70% 在目前環境受 coverage + numpy 載入衝突影響。
- 已提供隔離式 coverage 執行腳本（tools/runtime_setup/run_core_coverage.ps1）供乾淨 shell 驗證。
