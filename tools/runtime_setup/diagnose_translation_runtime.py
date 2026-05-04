#!/usr/bin/env python
"""
診斷 In-Process 翻譯運行時是否正確安裝
"""
from pathlib import Path
import sys
import subprocess

def check_directory(path_str, desc):
    """檢查目錄是否存在"""
    path = Path(path_str)
    status = "✓" if path.exists() else "✗"
    print(f"  [{status}] {desc}: {path}")
    return path.exists()

def check_module(py_exe, module_name):
    """檢查 Python 模組是否已安裝"""
    try:
        result = subprocess.run(
            [str(py_exe), "-c", f"import {module_name}; print('OK')"],
            capture_output=True,
            text=True,
            timeout=5
        )
        status = "✓" if result.returncode == 0 else "✗"
        print(f"  [{status}] {module_name}: {result.stdout.strip() if result.returncode == 0 else result.stderr[:50]}")
        return result.returncode == 0
    except Exception as e:
        print(f"  [✗] {module_name}: {e}")
        return False

def main():
    print("=" * 70)
    print("In-Process 翻譯運行時診斷工具")
    print("=" * 70)
    
    base_dir = Path(__file__).parent
    print(f"\n基礎目錄: {base_dir}\n")
    
    # 檢查目錄結構
    print("1. 檢查外部運行時目錄結構:")
    shared_exists = check_directory(base_dir / "runtimes" / "shared", "runtimes/shared")
    shared_py_exists = check_directory(base_dir / "runtimes" / "shared" / "Scripts" / "python.exe", "shared Python 可執行文件")
    shared_site_packages_exists = check_directory(base_dir / "runtimes" / "shared" / "Lib" / "site-packages", "shared site-packages")
    
    main_venv_exists = check_directory(base_dir / ".venv" / "Scripts" / "python.exe", "主 .venv Python 可執行文件")
    
    # 檢查模組
    print("\n2. 檢查 Python 模組安裝狀態:")
    
    if shared_py_exists:
        shared_py = base_dir / "runtimes" / "shared" / "Scripts" / "python.exe"
        print("  在 runtimes/shared 中:")
        has_llama_cpp_shared = check_module(shared_py, "llama_cpp")
        has_torch_shared = check_module(shared_py, "torch")
        has_onnxruntime_shared = check_module(shared_py, "onnxruntime")
    else:
        has_llama_cpp_shared = False
        print("  [✗] 無法檢查：runtimes/shared/Scripts/python.exe 不存在")
    
    if main_venv_exists:
        main_py = base_dir / ".venv" / "Scripts" / "python.exe"
        print("  在主 .venv 中:")
        has_llama_cpp_main = check_module(main_py, "llama_cpp")
    else:
        has_llama_cpp_main = False
        print("  [✗] 無法檢查：.venv/Scripts/python.exe 不存在")
    
    # 檢查模型文件
    print("\n3. 檢查 LLM 模型文件:")
    model_file = base_dir / "runtimes" / "models" / "llm" / "hy-mt1.5-7b.gguf"
    model_exists = check_directory(model_file, "LLM GGUF 模型")
    
    # 總結
    print("\n" + "=" * 70)
    print("診斷結果:")
    print("=" * 70)
    
    issues = []
    
    if not shared_exists:
        issues.append("❌ runtimes/shared 目錄不存在")
    
    if not has_llama_cpp_shared:
        issues.append("❌ llama-cpp-python 未安裝在 runtimes/shared")
    
    if not model_exists:
        issues.append("❌ LLM 模型文件不存在")
    
    if not has_torch_shared:
        issues.append("⚠️  torch 未安裝在 runtimes/shared")
    
    if not has_onnxruntime_shared:
        issues.append("⚠️  onnxruntime 未安裝在 runtimes/shared")
    
    if issues:
        print("\n發現以下問題:\n")
        for issue in issues:
            print(f"  {issue}")
        print("\n修復步驟:")
        if not has_llama_cpp_shared:
            print("""
  1. 安裝 Visual Studio Build Tools (C++ 工作負載):
     https://visualstudio.microsoft.com/visual-cpp-build-tools/
  
  2. 在 PowerShell 中運行:
     $sharedPy = "runtimes/shared/Scripts/python.exe"
     & $sharedPy -m pip install "llama-cpp-python>=0.3.8"
""")
        if not model_exists:
            print("""
  模型文件缺失，運行完整的運行時設置:
     .\\tools\\runtime_setup\\prepare_external_runtimes.ps1
""")
    else:
        print("\n✅ 所有檢查都通過！翻譯運行時應該能正常工作。")
    
    print("\n" + "=" * 70)
    
    return 0 if not issues else 1

if __name__ == "__main__":
    sys.exit(main())
