# SyncTranslate

Windows 本地雙向口譯工具（開發中）。

## 目前進度

- Stage 1-6: 專案骨架、設定檔、裝置列舉、路由檢查、分頁主控台
- Stage 7-14: remote/local pipeline 架構、字幕緩衝、診斷匯出、模式切換
- 目前 ASR/Translate/TTS 為 mock 實作（先驗證音訊與流程）

## 執行

```powershell
uv run python .\main.py
```

## 快速檢查（不開視窗）

```powershell
uv run python .\main.py --check
```

## 設定檔

- 主要設定：`config.yaml`
- 範例設定：`config.example.yaml`
- `model.asr_provider` 支援 `mock`、`openai`
- `model.translate_provider` 支援 `mock`、`openai`
- `model.tts_provider` 支援 `mock`、`openai`
- 使用 `openai` provider 時，請先設定環境變數 `OPENAI_API_KEY`

## OpenAI Provider 快速測試

1. 設定環境變數（PowerShell）：
```powershell
$env:OPENAI_API_KEY="你的金鑰"
```
2. 啟動程式後到 `模型與語言` 分頁
3. 將 `ASR` / `Translate` / `TTS` 切換為 `openai`（可分開切）
4. 確認 OpenAI 相關欄位（base URL / models / voice）
5. 回 `音訊路由` 按 `儲存設定`
6. 在 `模型與語言` 可先按 `測試 ASR / 測試 Translate / 測試 TTS`
7. 到 `音訊診斷` 按 `測試英文送出`
8. 再按 `開始` 跑單向或雙向模式

補充：Provider 測試會在背景執行，不會卡住 UI；一次僅允許一個測試。
可在模型頁按「取消測試」中止等待結果；若超過約 25 秒會標示逾時。
可按「清除測試狀態」重置 `Provider test` 與 `Last success` 顯示。
`Last success` 會寫入 `config.yaml`，重啟程式後仍會保留。
`Last success` 儲存為完整日期時間與結果文字，UI 會自動截短顯示避免過長。
