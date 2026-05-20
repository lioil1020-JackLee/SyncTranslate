param(
    [string]$Source = "dist/SyncTranslate-onedir",
    [string]$Output = "dist/SyncTranslate-onedir-windows.zip",
    [string]$ChecksumOutput = ""
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $Source)) {
    throw "Missing onedir source: $Source"
}

$internalBridge = Join-Path $Source "_internal/runtimes/audio/sync_audio_bridge.exe"
$releaseBridge = Join-Path $Source "runtimes/audio/sync_audio_bridge.exe"
if ((Test-Path $internalBridge) -and !(Test-Path $releaseBridge)) {
    New-Item -ItemType Directory -Path (Split-Path -Parent $releaseBridge) -Force | Out-Null
    Copy-Item -LiteralPath $internalBridge -Destination $releaseBridge -Force
    Write-Host "[package] copied bridge executable to $releaseBridge"
}

$outputParent = Split-Path -Parent $Output
if ($outputParent) {
    New-Item -ItemType Directory -Path $outputParent -Force | Out-Null
}
if (Test-Path $Output) {
    Remove-Item $Output -Force
}

$sevenZip = $null
$pathCommand = Get-Command "7z.exe" -ErrorAction SilentlyContinue
if ($pathCommand) {
    $sevenZip = $pathCommand.Source
}
elseif (Test-Path "C:\Program Files\7-Zip\7z.exe") {
    $sevenZip = "C:\Program Files\7-Zip\7z.exe"
}
elseif (Test-Path "C:\Program Files (x86)\7-Zip\7z.exe") {
    $sevenZip = "C:\Program Files (x86)\7-Zip\7z.exe"
}

if (-not $sevenZip) {
    throw "7-Zip not found. Install 7-Zip or add 7z.exe to PATH."
}

Write-Host "[package] using 7-Zip: $sevenZip"
$sourceFull = (Resolve-Path $Source).Path
$outputFull = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)
Push-Location $sourceFull
try {
    & $sevenZip a -tzip $outputFull ".\*" -mx=1
    if ($LASTEXITCODE -ne 0) {
        throw "7-Zip failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

Get-Item $Output | Select-Object FullName,Length | Format-Table -AutoSize

$hash = Get-FileHash -Path $Output -Algorithm SHA256
if (!$ChecksumOutput) {
    $ChecksumOutput = "$Output.sha256"
}
$checksumLine = "$($hash.Hash)  $(Split-Path -Leaf $Output)"
Set-Content -Path $ChecksumOutput -Value $checksumLine -Encoding ASCII
Get-Item $ChecksumOutput | Select-Object FullName,Length | Format-Table -AutoSize
