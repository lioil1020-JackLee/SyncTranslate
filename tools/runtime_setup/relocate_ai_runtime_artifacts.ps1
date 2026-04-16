param(
    [string]$DistRoot = "dist/SyncTranslate-onedir",
    [string]$DevRuntimesPath = "runtimes"
)

$ErrorActionPreference = "Stop"

# Copy external runtimes from development environment to dist root
# This ensures packaged app has same runtimes structure as development
if (!(Test-Path $DevRuntimesPath)) {
    Write-Host "Warning: Development runtimes not found at $DevRuntimesPath, skipping copy"
    exit 0
}

$runtimesDestRoot = Join-Path $DistRoot "runtimes"
if (Test-Path $runtimesDestRoot) {
    Write-Host "Removing existing runtimes in $DistRoot"
    Remove-Item $runtimesDestRoot -Recurse -Force
}

Write-Host "Copying runtimes from development environment..."
Copy-Item $DevRuntimesPath $runtimesDestRoot -Recurse -Force

$runtimesInDist = Get-ChildItem $runtimesDestRoot -Directory | Select-Object -ExpandProperty Name
Write-Host "Copied runtimes: $($runtimesInDist -join ', ')"

# Clean up any legacy _internal/runtimes/ directory if it exists
$internalRoot = Join-Path $DistRoot "_internal"
$oldRuntimesDir = Join-Path $internalRoot "runtimes"
if (Test-Path $oldRuntimesDir) {
    Remove-Item $oldRuntimesDir -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "Cleaned up legacy _internal/runtimes/"
}

Write-Host "Relocation completed: runtimes/ now at root of $DistRoot"
