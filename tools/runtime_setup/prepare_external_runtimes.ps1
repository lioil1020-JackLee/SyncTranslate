param(
    [string]$RuntimeRoot = "runtimes",
    [switch]$CpuOnly
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

# Shared runtime for CUDA / model loading dependencies used by both ASR backends.
New-RuntimeVenv -Name "shared" -InstallCudaTorch -Packages @(
    "onnxruntime>=1.22.0",
    "modelscope>=1.18.0",
    "addict>=2.4.0"
)

# Non-Chinese ASR runtime.
New-RuntimeVenv -Name "faster_whisper" -Packages @(
    "faster-whisper>=1.2.1",
    "ctranslate2>=4.6.0",
    "tiktoken>=0.11.0"
)

# Chinese ASR runtime.
New-RuntimeVenv -Name "funasr" -Packages @(
    "funasr>=1.3.0",
    "silero-vad>=6.2.1"
)

Write-Host "External runtimes prepared under: $RuntimeRoot"
Write-Host "Expected structure:"
Write-Host "  runtimes/shared/Lib/site-packages"
Write-Host "  runtimes/faster_whisper/Lib/site-packages"
Write-Host "  runtimes/funasr/Lib/site-packages"
