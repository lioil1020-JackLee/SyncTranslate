#!/usr/bin/env powershell
<#
.SYNOPSIS
完整一鍵設置 SyncTranslate 運行環境
自動檢查和安裝所有必要的依賴，包括 Visual Studio Build Tools

.PARAMETER SkipVsInstall
跳過 Visual Studio Build Tools 檢查 (不推薦)

.EXAMPLE
.\setup_complete.ps1
#>

param(
    [switch]$SkipVsInstall = $false
)

$ErrorActionPreference = "Stop"

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host "=" * 70 -ForegroundColor Cyan
    Write-Host $Title -ForegroundColor Cyan
    Write-Host "=" * 70 -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Write-Warning-Custom {
    param([string]$Message)
    Write-Host "[!] $Message" -ForegroundColor Yellow
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Test-VsToolsInstalled {
    # 檢查 Visual Studio Build Tools (nmake, cl.exe)
    $nmake = Get-Command "nmake" -ErrorAction SilentlyContinue
    $cl = Get-Command "cl" -ErrorAction SilentlyContinue
    
    return ($nmake -ne $null) -and ($cl -ne $null)
}

function Invoke-PrepareRuntimes {
    Write-Section "Step 2: 準備外部運行時 (外部模型+依賴)"
    
    $scriptPath = Join-Path (Split-Path -Parent $PSScriptRoot) "prepare_external_runtimes.ps1"
    
    if (-not (Test-Path $scriptPath)) {
        Write-Error-Custom "找不到 prepare_external_runtimes.ps1"
        throw "Setup failed: prepare_external_runtimes.ps1 not found"
    }
    
    & $scriptPath -RuntimeRoot "runtimes"
}

function Test-LlamaCppInstalled {
    param([string]$SharedPy)
    
    $output = & $SharedPy -c "import llama_cpp; print('OK')" 2>&1
    return $LASTEXITCODE -eq 0
}

function Install-LlamaCppPython {
    param([string]$SharedPy)
    
    Write-Host "安裝 llama-cpp-python..."
    & $SharedPy -m pip install "llama-cpp-python>=0.3.8" --no-cache-dir
    
    if ($LASTEXITCODE -ne 0) {
        return $false
    }
    
    # 驗證安裝
    return Test-LlamaCppInstalled $SharedPy
}

# ============================================================
# 主流程
# ============================================================

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║      SyncTranslate 完整運行環境設置 (一鍵安裝)              ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan

# Step 1: 檢查 Visual Studio Build Tools
Write-Section "Step 1: 檢查 Visual Studio Build Tools (C++ 編譯器)"

if (Test-VsToolsInstalled) {
    Write-Success "Visual Studio Build Tools 已安裝"
} else {
    if ($SkipVsInstall) {
        Write-Warning-Custom "跳過 Build Tools 檢查 (可能導致 llama-cpp-python 安裝失敗)"
    } else {
        Write-Error-Custom "未找到 Visual Studio Build Tools"
        Write-Host ""
        Write-Host "請安裝 Visual Studio Build Tools:" -ForegroundColor Yellow
        Write-Host "  1. 訪問: https://visualstudio.microsoft.com/visual-cpp-build-tools/" -ForegroundColor Yellow
        Write-Host "  2. 下載 Visual Studio Build Tools" -ForegroundColor Yellow
        Write-Host "  3. 安裝時選擇 'Desktop development with C++' 工作負載" -ForegroundColor Yellow
        Write-Host "  4. 完成安裝後，重啟此腳本" -ForegroundColor Yellow
        Write-Host ""
        throw "Setup failed: Visual Studio Build Tools not installed"
    }
}

# Step 2: 準備運行時
Invoke-PrepareRuntimes

# Step 3: 驗證 llama-cpp-python
Write-Section "Step 3: 驗證 llama-cpp-python"

$sharedPy = Join-Path (Get-Location) "runtimes\shared\Scripts\python.exe"

if (-not (Test-Path $sharedPy)) {
    Write-Error-Custom "找不到 runtimes/shared/Scripts/python.exe"
    throw "Setup failed"
}

if (Test-LlamaCppInstalled $sharedPy) {
    Write-Success "llama-cpp-python 已安裝"
} else {
    Write-Warning-Custom "llama-cpp-python 未安裝，嘗試安裝..."
    
    if (Install-LlamaCppPython $sharedPy) {
        Write-Success "llama-cpp-python 安裝成功"
    } else {
        Write-Error-Custom "llama-cpp-python 安裝失敗"
        Write-Host ""
        Write-Host "故障排除:" -ForegroundColor Yellow
        Write-Host "  1. 確保 Visual Studio Build Tools 已完整安裝 (包括 nmake)" -ForegroundColor Yellow
        Write-Host "  2. 執行: where nmake (應返回一個路徑)" -ForegroundColor Yellow
        Write-Host "  3. 重啟電腦後重試" -ForegroundColor Yellow
        throw "llama-cpp-python installation failed"
    }
}

# Step 4: 最終驗證
Write-Section "Step 4: 最終驗證"

# 檢查所有必要的組件
$checks = @{
    "runtimes/shared 存在" = (Test-Path "runtimes\shared\Lib\site-packages")
    "torch 已安裝" = (& $sharedPy -c "import torch; print('OK')" 2>&1) -ne $null
    "onnxruntime 已安裝" = (& $sharedPy -c "import onnxruntime; print('OK')" 2>&1) -ne $null
    "llama-cpp-python 已安裝" = (Test-LlamaCppInstalled $sharedPy)
    "模型文件存在" = (Test-Path "runtimes\models\llm\hy-mt1.5-7b.gguf")
}

$allPassed = $true
foreach ($check in $checks.GetEnumerator()) {
    if ($check.Value) {
        Write-Success $check.Key
    } else {
        Write-Error-Custom $check.Key
        $allPassed = $false
    }
}

Write-Host ""
if ($allPassed) {
    Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Green
    Write-Host "║            所有檢查通過! 翻譯功能已準備就緒!               ║" -ForegroundColor Green
    Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Green
    Write-Host ""
    Write-Host "下一步:" -ForegroundColor Green
    Write-Host "  1. 重啟 SyncTranslate 應用" -ForegroundColor Green
    Write-Host "  2. 測試翻譯功能" -ForegroundColor Green
} else {
    Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Red
    Write-Host "║            某些檢查失敗，請查看上面的錯誤信息             ║" -ForegroundColor Red
    Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Red
    throw "Setup incomplete"
}
