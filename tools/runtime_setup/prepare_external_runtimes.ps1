param(
    [string]$RuntimeRoot = "runtimes",
    [switch]$CpuOnly,
    [string]$BelleModelRepo = ""
)

$ErrorActionPreference = "Stop"

function New-RuntimeVenv {
    param(
        [string]$Name,
        [string[]]$Packages,
        [switch]$InstallCudaTorch
    )

    $runtimePath = Join-Path $RuntimeRoot $Name
    if (Test-Path $runtimePath) {
        Remove-Item $runtimePath -Recurse -Force
    }

    Write-Host "[runtime:$Name] create venv"
    uv run python -m venv $runtimePath

    $py = Join-Path $runtimePath "Scripts\python.exe"
    & $py -m pip install --upgrade pip

    if ($InstallCudaTorch -and -not $CpuOnly) {
        Write-Host "[runtime:$Name] install CUDA torch stack"
        & $py -m pip install torch==2.10.0+cu128 torchaudio==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128
    }

    if ($Packages.Count -gt 0) {
        Write-Host "[runtime:$Name] install packages: $($Packages -join ', ')"
        & $py -m pip install @Packages
    }
}

New-Item -ItemType Directory -Path $RuntimeRoot -Force | Out-Null

# Shared runtime for CUDA / common dependencies.
New-RuntimeVenv -Name "shared" -InstallCudaTorch -Packages @(
    "onnxruntime>=1.22.0"
)

# Non-Chinese ASR runtime.
New-RuntimeVenv -Name "faster_whisper" -Packages @(
    "faster-whisper>=1.2.1",
    "ctranslate2>=4.6.0",
    "tiktoken>=0.11.0"
)

$modelsRoot = Join-Path $RuntimeRoot "models"
New-Item -ItemType Directory -Path $modelsRoot -Force | Out-Null

$downloadArgs = @(
    ".\tools\runtime_setup\download_belle_model.py",
    "--local-dir",
    (Join-Path $modelsRoot "belle-zh-ct2")
)

if ($BelleModelRepo) {
    $downloadArgs += @("--repo-id", $BelleModelRepo)
}
elseif ($env:SYNC_TRANSLATE_BELLE_MODEL_REPO) {
    $downloadArgs += @("--repo-id", $env:SYNC_TRANSLATE_BELLE_MODEL_REPO)
}

Write-Host "[runtime:belle_model] download model snapshot"
uv run python @downloadArgs

Write-Host "External runtimes prepared under: $RuntimeRoot"
Write-Host "Expected structure:"
Write-Host "  runtimes/shared/Lib/site-packages"
Write-Host "  runtimes/faster_whisper/Lib/site-packages"
Write-Host "  runtimes/models/belle-zh-ct2"
