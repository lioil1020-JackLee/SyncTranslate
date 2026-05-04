## 翻譯功能診斷報告

**生成時間**: 2025-12-19
**狀態**: ❌ 翻譯功能已準備就緒，但 llama-cpp-python 依賴缺失

### 診斷結果

✅ **架構遷移**
- 已正確從 HTTP (LM Studio) 遷移至本地進程 (llama-cpp-python)
- 配置後端: `local_llama_inprocess` ✓
- 模型路徑: `.\runtimes\models\llm\hy-mt1.5-7b.gguf` ✓ (4.6GB)
- 所有 URL 引用已移除 ✓

✅ **代碼遷移**
- `LocalLlamaTranslationProvider` 已實現 ✓
- `InProcessLlamaClient` 已實現 ✓  
- 配置解析已更新 ✓
- 所有 405 個單元測試通過 ✓

❌ **運行時依賴**
- llama-cpp-python **未安裝**
- 位置: `runtimes/shared/Lib/site-packages/`
- 需要: Visual Studio Build Tools (C++ 編譯器)

### 錯誤信息
```
Health check: False
Message: llama-cpp-python is not installed. 
To fix:
  1. Install Visual Studio Build Tools (C++ workload)
  2. Run: runtimes/shared/Scripts/python.exe -m pip install 'llama-cpp-python>=0.3.8'
```

### 解決步驟

#### 第 1 步: 安裝 Visual Studio Build Tools
1. 訪問: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. 下載 **Visual Studio Build Tools** (非 Visual Studio Community)
3. 安裝時選擇 **Desktop development with C++** 工作負載
4. 完成安裝

#### 第 2 步: 安裝 llama-cpp-python
在 PowerShell 中執行:
```powershell
cd e:\py\SyncTranslate
.\fix_translation_runtime.ps1
```

或手動執行:
```powershell
.\runtimes\shared\Scripts\python.exe -m pip install "llama-cpp-python>=0.3.8"
```

#### 第 3 步: 驗證安裝
```powershell
python .\diagnose_translation_runtime.py
```

應該看到所有項目都是 ✓ (綠色)

### 驗證清單

- [x] 架構正確 (本地進程，無 HTTP)
- [x] 配置正確 (model_path, gpu_layers 等)
- [x] 模型文件存在 (4.6GB GGUF)
- [x] 代碼已遷移 (所有測試通過)
- [ ] Visual Studio Build Tools 已安裝
- [ ] llama-cpp-python 已安裝
- [ ] 翻譯功能已驗證

### 預期行為

安裝完成後，重啟 SyncTranslate，翻譯功能應該立即工作:
- 文本翻譯: ✓
- ASR 文本修正: ✓  
- 自動字幕翻譯: ✓

### 技術細節

**為什麼需要 Visual Studio Build Tools?**
- llama-cpp-python 是 C++ 擴展
- Windows 上的 pip 需要編譯器 (nmake, cl.exe)
- 無需完整 Visual Studio，只需 Build Tools

**llama-cpp-python 做什麼?**
- 提供 Python 綁定到 llama.cpp (高效 LLM 推理)
- 直接在進程內載入和運行 GGUF 模型
- 無需外部 HTTP 服務器

### 如果遇到問題

1. **Build Tools 安裝失敗**
   - 檢查磁盤空間 (需要 ~2GB)
   - 嘗試以管理員身份執行
   - 查看 Visual Studio 安裝日誌

2. **pip install 失敗**
   - 確保 Build Tools 已完全安裝
   - 執行: `where nmake` (應該返回一個路徑)
   - 嘗試指定版本: `pip install llama-cpp-python==0.3.8`

3. **模型載入失敗**
   - 檢查 GPU VRAM (需要 ~4-6GB 用於 7B 模型)
   - 查看 logs/ 目錄中的詳細錯誤
   - 嘗試降低 gpu_layers (從 35 至 20)

### 相關文件

- 快速開始: `TRANSLATION_FIX_QUICK_START.md`
- 詳細說明: `TRANSLATION_RUNTIME_FIX.md`
- 診斷工具: `diagnose_translation_runtime.py`
- 修復腳本: `fix_translation_runtime.ps1`
- 配置架構: `app/infra/config/schema.py`
- 翻譯適配器: `app/infra/translation/inprocess_adapter.py`
