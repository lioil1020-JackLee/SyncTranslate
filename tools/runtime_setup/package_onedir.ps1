param(
    [string]$Source = "dist/SyncTranslate-onedir",
    [string]$Output = "dist/SyncTranslate-onedir-windows.zip"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $Source)) {
    throw "Missing onedir source: $Source"
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
