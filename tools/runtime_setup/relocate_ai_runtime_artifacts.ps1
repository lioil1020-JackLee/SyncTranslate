param(
    [string]$DistRoot = "dist/SyncTranslate-onedir",
    [string]$DevRuntimesPath = "runtimes",
    [switch]$SkipValidation
)

$ErrorActionPreference = "Stop"

# Copy external runtimes from development environment to dist root
# This ensures packaged app has same runtimes structure as development
if (!(Test-Path $DevRuntimesPath)) {
    throw "Development runtimes not found at $DevRuntimesPath. Run tools\runtime_setup\prepare_external_runtimes.ps1 first."
}

function Assert-RuntimeFile {
    param(
        [string]$Path,
        [string]$Label
    )
    if (!(Test-Path $Path)) {
        throw "$Label is missing: $Path"
    }
    $item = Get-Item $Path
    if ($item.PSIsContainer) {
        return
    }
    if ($item.Length -le 0) {
        throw "$Label is empty: $Path"
    }
}

if (-not $SkipValidation) {
    Assert-RuntimeFile -Path (Join-Path $DevRuntimesPath "shared") -Label "Shared runtime"
    Assert-RuntimeFile -Path (Join-Path $DevRuntimesPath "faster_whisper") -Label "faster-whisper runtime"
    Assert-RuntimeFile -Path (Join-Path $DevRuntimesPath "models\belle-zh-ct2\config.json") -Label "Belle ASR model"
    Assert-RuntimeFile -Path (Join-Path $DevRuntimesPath "models\llm\hy-mt1.5-7b.gguf") -Label "HY-MT1.5 GGUF model"
}

$runtimesDestRoot = Join-Path $DistRoot "runtimes"
if (Test-Path $runtimesDestRoot) {
    Write-Host "Removing existing runtimes in $DistRoot"
    Remove-Item $runtimesDestRoot -Recurse -Force
}

Write-Host "Copying runtimes from development environment..."
New-Item -ItemType Directory -Path $runtimesDestRoot -Force | Out-Null

foreach ($runtimeName in @("shared", "faster_whisper")) {
    $src = Join-Path $DevRuntimesPath $runtimeName
    $dst = Join-Path $runtimesDestRoot $runtimeName
    Write-Host "Copying runtime: $runtimeName"
    Copy-Item -LiteralPath $src -Destination $dst -Recurse -Force
}

$modelsDest = Join-Path $runtimesDestRoot "models"
New-Item -ItemType Directory -Path $modelsDest -Force | Out-Null

foreach ($modelName in @("belle-zh-ct2", "llm")) {
    $src = Join-Path (Join-Path $DevRuntimesPath "models") $modelName
    $dst = Join-Path $modelsDest $modelName
    Write-Host "Copying model assets: models/$modelName"
    Copy-Item -LiteralPath $src -Destination $dst -Recurse -Force
}

$runtimesInDist = Get-ChildItem $runtimesDestRoot -Directory | Select-Object -ExpandProperty Name
Write-Host "Copied runtimes: $($runtimesInDist -join ', ')"

if (-not $SkipValidation) {
    Assert-RuntimeFile -Path (Join-Path $runtimesDestRoot "models\llm\hy-mt1.5-7b.gguf") -Label "Packaged HY-MT1.5 GGUF model"
}

# Clean up any legacy _internal/runtimes/ directory if it exists
$internalRoot = Join-Path $DistRoot "_internal"
$oldRuntimesDir = Join-Path $internalRoot "runtimes"
if (Test-Path $oldRuntimesDir) {
    Remove-Item $oldRuntimesDir -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "Cleaned up legacy _internal/runtimes/"
}

Write-Host "Relocation completed: runtimes/ now at root of $DistRoot"
