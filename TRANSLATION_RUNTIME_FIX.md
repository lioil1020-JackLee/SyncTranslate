# 翻譯功能 In-Process 運行時修復指南

## 問題

翻譯功能無法工作，出現「in-process translation backend silently fails」或「llama-cpp-python is not installed」的錯誤。

## 原因

`llama-cpp-python` 沒有安裝在 `runtimes/shared` 虛擬環境中。該依賴是在運行 `prepare_external_runtimes.ps1` 時安裝的，但在 Windows 上需要 Visual Studio Build Tools（C++ 編譯器）才能構建。

## 快速修復

### 前置要求

安裝 Visual Studio Build Tools 中的 C++ 編譯工具：

1. 前往 [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. 下載並運行安裝程式
3. 勾選「Desktop development with C++」工作負載
4. 完成安裝

### 安裝步驟

在 PowerShell 中運行以下命令：

```powershell
$sharedPy = "e:\py\SyncTranslate\runtimes\shared\Scripts\python.exe"
& $sharedPy -m pip install "llama-cpp-python>=0.3.8"
```

### 驗證安裝

```powershell
$sharedPy = "e:\py\SyncTranslate\runtimes\shared\Scripts\python.exe"
& $sharedPy -c "import llama_cpp; print('llama-cpp-python installed successfully')"
```

如果輸出 `llama-cpp-python installed successfully`，則安裝成功。

## 詳細技術說明

### 運行時架構

```
主程式 (.venv)
  ↓
configure_external_ai_runtime()
  ↓
掃描 runtimes/shared/Lib/site-packages
  ↓
InProcessLlamaClient (lazy load)
  ↓
Llama 運行時 (首次使用時初始化)
```

### 依賴關係

- **主 .venv**：不需要 llama-cpp-python
- **runtimes/shared**：必須包含：
  - torch >= 2.10.0+cu128
  - torchaudio >= 2.10.0+cu128
  - onnxruntime >= 1.22.0
  - **llama-cpp-python >= 0.3.8** ← 缺少此項

### 故障排除

#### 錯誤：「CMake Error: CMAKE_C_COMPILER not set」

**原因**：缺少 Visual Studio Build Tools

**解決**：按上述步驟安裝 Visual Studio Build Tools

#### 錯誤：「no such file or directory (nmake)」

**原因**：同上

**解決**：同上

#### 驗證失敗但安裝成功

嘗試重新啟動應用程式，因為 Python 模組緩存可能已加載。

## 代碼相關位置

- **模型初始化**：[app/infra/translation/inprocess_adapter.py](app/infra/translation/inprocess_adapter.py#L28)
- **提供者工廠**：[app/infra/translation/provider.py](app/infra/translation/provider.py#L88)
- **運行時配置**：[app/bootstrap/external_runtime.py](app/bootstrap/external_runtime.py)
- **安裝腳本**：[tools/runtime_setup/prepare_external_runtimes.ps1](tools/runtime_setup/prepare_external_runtimes.ps1#L32)

## 相關配置

確保 `config.yaml` 中設置正確：

```yaml
llm:
  backend: local_llama_inprocess
  runtime:
    model_path: "runtimes/models/llm/hy-mt1.5-7b.gguf"
    ctx_size: 4096
    gpu_layers: 35
    threads: 8
    batch_size: 512
```

## 完整運行時設置

如果上述修復無效，可以重新運行完整的運行時設置：

```powershell
cd e:\py\SyncTranslate
.\tools\runtime_setup\prepare_external_runtimes.ps1
```

確保系統已安裝：
- Visual Studio Build Tools（C++ 工作負載）
- Python 3.12
- Git（用於模型下載）
