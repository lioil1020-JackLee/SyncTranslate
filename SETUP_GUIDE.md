# SyncTranslate 一鍵安裝指南

## 快速安裝 (推薦)

```powershell
cd e:\py\SyncTranslate
.\tools\runtime_setup\setup_complete.ps1
```

該腳本會自動:
1. ✓ 檢查 Visual Studio Build Tools
2. ✓ 準備外部運行時 (runtimes/)
3. ✓ 下載 LLM 模型 (HY-MT1.5-7B GGUF)
4. ✓ 安裝 llama-cpp-python
5. ✓ 驗證所有依賴

---

## 如果 Build Tools 未安裝

如果 `setup_complete.ps1` 報告缺少 Visual Studio Build Tools:

1. **下載**: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. **執行安裝程式**
3. **選擇**: "Desktop development with C++" 工作負載
4. **完成安裝** (~2GB，需要 5-10 分鐘)
5. **重啟電腦** (使 nmake 在 PATH 中生效)
6. **重新運行**:
   ```powershell
   .\tools\runtime_setup\setup_complete.ps1
   ```

---

## 打包發布版本

準備就緒後，打包 onedir:

```powershell
.\tools\runtime_setup\package_onedir.ps1 `
    -Source "dist/SyncTranslate-onedir" `
    -Output "dist/SyncTranslate-onedir-windows.zip"
```

---

## 故障排除

### 問題: "nmake not found" 或編譯錯誤

**解決**: Visual Studio Build Tools 未完整安裝

```powershell
# 驗證 nmake 是否可用
where nmake
```

如果返回路徑 → Build Tools 已正確安裝
如果無返回 → 重新安裝 Build Tools (選擇 C++ 工作負載)

### 問題: "llama-cpp-python installation failed"

1. 確保 Build Tools 已完整安裝
2. 重啟電腦
3. 重新運行 setup_complete.ps1

### 問題: "模型文件下載失敗"

通常是臨時網路問題:
```powershell
# 手動下載
.\tools\runtime_setup\setup_complete.ps1
# 或重新執行 prepare_external_runtimes.ps1
```

---

## 文件結構說明

安裝完成後，應該看到:

```
runtimes/
  ├── shared/                          ← CUDA + in-process LLM runtime
  │   ├── Lib/site-packages/
  │   │   ├── torch/
  │   │   ├── llama_cpp/               ← 關鍵
  │   │   └── onnxruntime/
  │   └── Scripts/python.exe
  ├── faster_whisper/                  ← ASR runtime
  │   └── Lib/site-packages/
  │       └── faster_whisper/
  └── models/
      ├── belle-zh-ct2/                ← 中文 ASR 模型
      └── llm/
          └── hy-mt1.5-7b.gguf         ← 翻譯模型 (4.3GB)
```

---

## 驗證安裝

手動驗證所有組件:

```python
# 測試 llama-cpp-python
python -c "from app.infra.translation.provider import create_translation_provider; from app.infra.config.schema import LlmConfig; p = create_translation_provider(LlmConfig()); ok, msg = p.health_check(); print('Translation ready!' if ok else f'Not ready: {msg}')"
```

---

## 何時需要重新運行

通常只需運行一次。但如果:
- ✓ 更新了模型
- ✓ 清除了 runtimes/ 目錄
- ✓ 升級了 Python

需要重新執行:
```powershell
.\tools\runtime_setup\setup_complete.ps1
```

---

## 技術背景

**為什麼需要 Visual Studio Build Tools?**

llama-cpp-python 是 C++ 擴展，在 Windows 上需要編譯。Build Tools 提供了必要的編譯器 (MSVC cl.exe, nmake 等)。

**架構概述**

```
SyncTranslate 應用
    ↓
[LocalLlamaTranslationProvider]
    ↓
[InProcessLlamaClient]
    ↓
[llama-cpp-python]
    ↓
[runtimes/shared/Lib/site-packages/llama_cpp/]
    ↓
[HY-MT1.5-7B GGUF Model] ← runtimes/models/llm/
```

無需外部 HTTP 服務器，直接在進程內運行。
