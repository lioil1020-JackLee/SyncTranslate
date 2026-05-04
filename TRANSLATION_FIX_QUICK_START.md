# 翻譯功能快速修復指南

## 根本原因
翻譯系統已完全遷移至 **本地進程模式 (llama-cpp-python)**，但該依賴 **未安裝**。

```
架構: ✅ 已遷移
代碼: ✅ 已完成
配置: ✅ 已更新  
模型: ✅ 已存在
依賴: ❌ llama-cpp-python 未安裝 ← 唯一問題
```

## 3 步快速修復 (20-30 分鐘)

### 1️⃣ 安裝 Visual Studio Build Tools (5-10 分鐘)

1. 訪問: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. 下載並執行安裝程式
3. 選擇 **Desktop development with C++** 工作負載
4. 等待安裝完成（~2GB）
5. **重啟電腦** (使 nmake 在 PATH 中生效)

### 2️⃣ 安裝 llama-cpp-python (2-5 分鐘)

在 PowerShell 中執行:
```powershell
cd e:\py\SyncTranslate
.\fix_translation_runtime.ps1
```

或手動安裝:
```powershell
.\runtimes\shared\Scripts\python.exe -m pip install "llama-cpp-python>=0.3.8"
```

### 3️⃣ 驗證修復 (1 分鐘)

運行診斷工具確認：
```powershell
python .\diagnose_translation_runtime.py
```

應該看到綠色的 ✅ 符號和「所有檢查都通過」信息。

---

## 或者：手動修復

如果自動腳本不工作，手動執行：

```powershell
$sharedPy = "runtimes/shared/Scripts/python.exe"
& $sharedPy -m pip install "llama-cpp-python>=0.3.8"
```

---

## 故障排除

| 錯誤 | 原因 | 解決 |
|------|------|------|
| `CMake Error: CMAKE_C_COMPILER not set` | 缺少 MSVC | 安裝 Visual Studio Build Tools |
| `no such file or directory (nmake)` | 同上 | 同上 |
| 安裝後仍無法工作 | 模組緩存 | 重啟應用程式 |

---

## 相關文件

- 📄 [TRANSLATION_RUNTIME_FIX.md](TRANSLATION_RUNTIME_FIX.md) — 詳細技術指南
- 🐍 [diagnose_translation_runtime.py](diagnose_translation_runtime.py) — 診斷工具
- 🔧 [fix_translation_runtime.ps1](fix_translation_runtime.ps1) — 自動修復腳本
- 🔗 [app/infra/translation/inprocess_adapter.py](app/infra/translation/inprocess_adapter.py) — 改進的錯誤消息

---

## 技術背景

**為什麼需要這些步驟？**

- `in-process` 翻譯模式（2026-05-01 改動）完全在本程式內運行 LLM，不需要外部 server
- `llama-cpp-python` 是一個 C 擴展，需要編譯
- Windows 上編譯 C 擴展需要 Visual Studio Build Tools 提供的編譯器和工具

**架構變化**

```
舊方式: 應用 ←→ llama-server.exe (HTTP)
新方式: 應用 → llama-cpp-python → Llama 運行時（in-process）
```

---

## 下一步

修復完成後：
1. 重啟 SyncTranslate 應用
2. 在 UI 中測試翻譯功能
3. 檢查日誌確認沒有錯誤

祝修復順利！✨
