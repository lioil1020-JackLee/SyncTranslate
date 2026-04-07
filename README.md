# SyncTranslate

SyncTranslate 是一個在 Windows 上運作的雙向即時翻譯桌面工具，整合本地 ASR、LM Studio LLM 與 TTS，讓會議遠端音訊與本地麥克風都能各自完成辨識、翻譯、字幕顯示與語音播報。

## 專案概覽

- `app/application/`：session、audio router、設定套用、匯出與健康檢查流程
- `app/infra/`：音訊擷取、ASR、翻譯、TTS、設定讀寫等基礎設施
- `app/ui/`：PySide6 桌面介面
- `tests/`：UI、設定 migration、路由與整合測試
- `main.py`：CLI 與桌面程式入口
- `SyncTranslate-onedir.spec`：PyInstaller onedir 打包設定

## 依賴管理

此專案現在完全由 `uv` 管理依賴：

- 依賴來源：`pyproject.toml`
- 鎖定檔：`uv.lock`
- 開發群組：`dev`
- 打包群組：`build`
- 可選本地 ASR 依賴：`local`

安裝基本環境：

```powershell
uv sync --locked
```

安裝完整開發與本地 ASR 環境：

```powershell
uv sync --locked --extra local
```

安裝包含 PyInstaller 的打包環境：

```powershell
uv sync --locked --extra local --group build
```

## 常用命令

啟動程式：

```powershell
uv run python .\main.py
```

檢查設定與裝置：

```powershell
uv run python .\main.py --check
```

執行測試：

```powershell
uv run pytest -q
```

建立 onedir：

```powershell
uv run pyinstaller .\SyncTranslate-onedir.spec --noconfirm --clean
```

輸出目錄位於 `dist/SyncTranslate-onedir/`。

## 打包與發行

本機打包建議流程：

```powershell
uv sync --locked --extra local --group build
uv run pytest -q
uv run pyinstaller .\SyncTranslate-onedir.spec --noconfirm --clean
```

GitHub Actions 也使用相同的 `uv` 流程，對應設定在 `.github/workflows/build-release.yml`。

## 相關文件

- `docs/架構說明.md`
- `docs/設定說明.md`
- `docs/測試說明.md`
- `docs/快速安裝手冊.md`
- `docs/更新紀錄.md`
- `docs/音訊裝置建議配置.md`
