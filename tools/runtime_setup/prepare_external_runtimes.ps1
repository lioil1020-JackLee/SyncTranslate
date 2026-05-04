param(
    [string]$RuntimeRoot = "runtimes",
    [switch]$CpuOnly,
    [string]$BelleModelRepo = "",
    [string]$LlmModelPath = "",
    [string]$LlmModelRepo = "tencent/HY-MT1.5-7B-GGUF",
    [string]$LlmModelFile = "HY-MT1.5-7B-Q4_K_M.gguf",
    [switch]$SkipLlmDownload,
    [string]$LlamaCudaFlavor = "cu125",
    # Force CPU-only llama-cpp-python wheel (for CI without GPU, or CPU-only machines)
    [switch]$LlamaCpuOnly
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

# Shared runtime for CUDA / common dependencies + in-process LLM runtime.
New-RuntimeVenv -Name "shared" -InstallCudaTorch -Packages @(
    "onnxruntime>=1.22.0"
)

# Install llama-cpp-python. Auto-detects CUDA+MSVC for GPU build, falls back to CPU wheel.
$sharedPy = Join-Path $RuntimeRoot "shared\Scripts\python.exe"

function Find-VcVars64 {
    $candidates = @(
        "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files (x86)\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files (x86)\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { return $c }
    }
    return ""
}

function Install-LlamaCppPython {
    param([string]$Py, [switch]$CpuOnly)

    $hasNvcc = (Get-Command "nvcc" -ErrorAction SilentlyContinue) -ne $null
    $vcvars  = Find-VcVars64

    if (-not $CpuOnly -and $hasNvcc -and $vcvars) {
        Write-Host "[runtime:shared] CUDA + MSVC detected - building llama-cpp-python with CUDA support..."
        # Must run inside vcvars64 environment; use NMake generator to avoid VS CUDA toolset requirement
        # /utf-8 is required on Traditional Chinese Windows (CP950) to prevent C2001 errors in llama.cpp jinja headers
        $cmdFile = Join-Path $env:TEMP "synctranslate_llama_cpp_build.cmd"
        $cmdScript = @"
call "$vcvars"
set CMAKE_ARGS=-DGGML_CUDA=on -G "NMake Makefiles" -DCMAKE_C_FLAGS=/utf-8 -DCMAKE_CXX_FLAGS=/utf-8
set FORCE_CMAKE=1
"$Py" -m pip install "llama-cpp-python>=0.3.8" --no-binary llama-cpp-python --upgrade
"@
        Set-Content -LiteralPath $cmdFile -Value $cmdScript -Encoding ASCII
        & $cmdFile
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[runtime:shared] llama-cpp-python installed with CUDA support"
            return $true
        }
        Write-Warning "[runtime:shared] CUDA build failed (exit $LASTEXITCODE), falling back to prebuilt CUDA wheel"
    } else {
        if (-not $hasNvcc) { Write-Host "[runtime:shared] nvcc not found - skipping CUDA build" }
        if (-not $vcvars)  { Write-Host "[runtime:shared] MSVC (vcvars64.bat) not found - skipping CUDA build" }
    }

    # Fallback: prebuilt wheel from GitHub releases. GitHub-hosted runners do
    # not reliably provide nvcc, but release builds should still carry CUDA
    # llama.cpp so they behave like local GPU builds.
    $wheelKind = "CUDA"
    if ($CpuOnly) {
        $wheelKind = "CPU"
    }
    Write-Host "[runtime:shared] installing llama-cpp-python (prebuilt $wheelKind wheel)..."
    $releases = Invoke-RestMethod "https://api.github.com/repos/abetlen/llama-cpp-python/releases?per_page=30"
    $asset = $null
    if ($CpuOnly) {
        foreach ($release in $releases) {
            $tag = [string]$release.tag_name
            if ($tag -match "cu[0-9]+") {
                continue
            }
            $asset = $release.assets |
                Where-Object { $_.name -like "llama_cpp_python-*-py3-none-win_amd64.whl" } |
                Select-Object -First 1
            if ($asset) {
                break
            }
        }
    } else {
        $preferredCudaTags = @()
        if ($LlamaCudaFlavor) {
            $preferredCudaTags += $LlamaCudaFlavor
        }
        $preferredCudaTags += @("cu125", "cu124", "cu123")
        foreach ($flavor in ($preferredCudaTags | Select-Object -Unique)) {
            foreach ($release in $releases) {
                $tag = [string]$release.tag_name
                if ($tag -notmatch [regex]::Escape($flavor)) {
                    continue
                }
                $asset = $release.assets |
                    Where-Object { $_.name -like "llama_cpp_python-*-py3-none-win_amd64.whl" } |
                    Select-Object -First 1
                if ($asset) {
                    break
                }
            }
            if ($asset) {
                break
            }
        }
    }
    if ((-not $asset) -and (-not $CpuOnly)) {
        foreach ($release in $releases) {
            $tag = [string]$release.tag_name
            if ($tag -notmatch "cu[0-9]+") {
                continue
            }
            $asset = $release.assets |
                Where-Object { $_.name -like "llama_cpp_python-*-win_amd64.whl" } |
                Select-Object -First 1
            if ($asset) {
                break
            }
        }
    } elseif ((-not $asset) -and $CpuOnly) {
        foreach ($release in $releases) {
            $tag = [string]$release.tag_name
            if ($tag -match "cu[0-9]+") {
                continue
            }
            $asset = $release.assets |
                Where-Object { $_.name -like "llama_cpp_python-*-win_amd64.whl" } |
                Select-Object -First 1
            if ($asset) {
                break
            }
        }
    }
    if ($asset) {
        Write-Host "[runtime:shared] installing wheel asset: $($asset.name)"
        & $Py -m pip install $asset.browser_download_url
    } else {
        throw "[runtime:shared] no prebuilt llama-cpp-python Windows $wheelKind wheel found"
    }
    if ($LASTEXITCODE -ne 0) {
        throw "[runtime:shared] llama-cpp-python installation failed"
    }
    return $true
}

Install-LlamaCppPython -Py $sharedPy -CpuOnly:($CpuOnly -or $LlamaCpuOnly)

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

$llmModelsRoot = Join-Path $modelsRoot "llm"
New-Item -ItemType Directory -Path $llmModelsRoot -Force | Out-Null
$targetLlmModel = Join-Path $llmModelsRoot "hy-mt1.5-7b.gguf"
if ($LlmModelPath) {
    if (-not (Test-Path $LlmModelPath)) {
        throw "LLM model file not found: $LlmModelPath"
    }
    Copy-Item -LiteralPath $LlmModelPath -Destination $targetLlmModel -Force
}
elseif ((-not $SkipLlmDownload) -and (-not (Test-Path $targetLlmModel))) {
    uv run python ".\tools\runtime_setup\download_llm_model.py" `
        --repo-id $LlmModelRepo `
        --filename $LlmModelFile `
        --local-file $targetLlmModel
}
elseif (-not (Test-Path $targetLlmModel)) {
    Write-Warning "LLM GGUF not found at $targetLlmModel. Run without -SkipLlmDownload or pass -LlmModelPath."
}

Write-Host "External runtimes prepared under: $RuntimeRoot"
Write-Host "Expected structure:"
Write-Host "  runtimes/shared/Lib/site-packages"
Write-Host "  runtimes/faster_whisper/Lib/site-packages"
Write-Host "  runtimes/models/belle-zh-ct2"
Write-Host "  runtimes/models/llm/hy-mt1.5-7b.gguf"
Write-Host "  runtimes/shared/Lib/site-packages/llama_cpp"

$verifyLlamaCpp = @'
from llama_cpp import Llama
import llama_cpp
from llama_cpp.llama_cpp import llama_supports_gpu_offload
print("llama_cpp ready: " + str(getattr(llama_cpp, "__version__", "unknown")))
print("llama_cpp gpu: " + str(llama_supports_gpu_offload()))
if not llama_supports_gpu_offload():
    raise SystemExit("llama-cpp-python does not support GPU offload")
'@
$verifyLlamaCppPath = Join-Path $env:TEMP "synctranslate_verify_llama_cpp.py"
Set-Content -LiteralPath $verifyLlamaCppPath -Value $verifyLlamaCpp -Encoding ASCII
& $sharedPy $verifyLlamaCppPath
if ($LASTEXITCODE -ne 0) {
    throw "llama-cpp-python verification failed in runtimes/shared"
}

if (-not (Test-Path (Join-Path $RuntimeRoot "shared\Lib\site-packages\llama_cpp\lib\llama.dll"))) {
    throw "llama.cpp DLL is missing from runtimes/shared"
}

if (-not (Test-Path $targetLlmModel)) {
    throw "LLM GGUF model is missing: $targetLlmModel"
}
