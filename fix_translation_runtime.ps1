#!/usr/bin/env powershell
#
# 自動修復 In-Process 翻譯運行時
# 使用方法: .\fix_translation_runtime.ps1
#

param(
    [switch]$SkipBuildToolsCheck = $false
)

$ErrorActionPreference = "Stop"

Write-Host "=================================================="
Write-Host "翻譯運行時自動修復工具"
Write-Host "=================================================="

$baseDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$sharedPy = Join-Path $baseDir "runtimes\shared\Scripts\python.exe"

# 檢查基本條件
Write-Host "`n1. 檢查基本條件..."

if (-not (Test-Path $sharedPy)) {
    Write-Host "[✗] runtimes/shared/Scripts/python.exe 不存在" -ForegroundColor Red
    Write-Host "提示: 需要先運行 tools/runtime_setup/prepare_external_runtimes.ps1" -ForegroundColor Yellow
    exit 1
}
Write-Host "[✓] 找到 shared venv" -ForegroundColor Green

# 檢查 llama_cpp 是否已安裝
Write-Host "`n2. 檢查 llama-cpp-python 安裝狀態..."
$hasLlamaCpp = & $sharedPy -c "import llama_cpp; print('installed')" 2>&1 | Select-String "installed"

if ($hasLlamaCpp) {
    Write-Host "[✓] llama-cpp-python 已安裝" -ForegroundColor Green
    Write-Host "翻譯運行時應該已經能正常工作"
    exit 0
}

Write-Host "[✗] llama-cpp-python 未安裝" -ForegroundColor Red

# 檢查編譯工具
Write-Host "`n3. 檢查 C++ 編譯工具..."
$hasCl = Get-Command cl 2>&1 | Out-Null; if ($?) { Write-Host "[✓] MSVC 編譯器找到" -ForegroundColor Green } else { Write-Host "[✗] MSVC 編譯器未找到" -ForegroundColor Red }
$hasNmake = Get-Command nmake 2>&1 | Out-Null; if ($?) { Write-Host "[✓] nmake 找到" -ForegroundColor Green } else { Write-Host "[✗] nmake 未找到，需要安裝 Visual Studio Build Tools" -ForegroundColor Red }

if (-not $SkipBuildToolsCheck) {
    $hasNmake = Get-Command nmake 2>&1 | Out-Null
    if (-not $?) {
        Write-Host "`n❌ 缺少必要的編譯工具" -ForegroundColor Red
        Write-Host "`n修復步驟:"
        Write-Host "1. 下載 Visual Studio Build Tools:"
        Write-Host "   https://visualstudio.microsoft.com/visual-cpp-build-tools/"
        Write-Host ""
        Write-Host "2. 運行安裝程式，選擇『Desktop development with C++』工作負載"
        Write-Host ""
        Write-Host "3. 安裝完成後重新運行此腳本:"
        Write-Host "   .\fix_translation_runtime.ps1"
        Write-Host ""
        Write-Host "或使用 -SkipBuildToolsCheck 標誌跳過檢查（不推薦）:"
        Write-Host "   .\fix_translation_runtime.ps1 -SkipBuildToolsCheck"
        exit 1
    }
}

# 嘗試安裝 llama-cpp-python
Write-Host "`n4. 安裝 llama-cpp-python..."
Write-Host "執行: & $sharedPy -m pip install 'llama-cpp-python>=0.3.8'" -ForegroundColor Cyan

$output = & $sharedPy -m pip install "llama-cpp-python>=0.3.8" 2>&1
$lastLine = $output[-1]

if ($lastLine -match "Successfully installed") {
    Write-Host "[✓] 安裝成功" -ForegroundColor Green
} elseif ($lastLine -match "Requirement already satisfied") {
    Write-Host "[✓] 已經安裝" -ForegroundColor Green
} else {
    Write-Host "[✗] 安裝可能失敗" -ForegroundColor Red
    Write-Host "最後一行輸出: $lastLine"
}

# 驗證安裝
Write-Host "`n5. 驗證安裝..."
$verify = & $sharedPy -c "import llama_cpp; print('OK')" 2>&1
if ($verify -match "OK") {
    Write-Host "[✓] llama-cpp-python 驗證成功" -ForegroundColor Green
    Write-Host "`n=================================================="
    Write-Host "✅ 修復完成！翻譯運行時應該能正常工作"
    Write-Host "=================================================="
    exit 0
} else {
    Write-Host "[✗] 驗證失敗" -ForegroundColor Red
    Write-Host "輸出: $verify"
    exit 1
}
