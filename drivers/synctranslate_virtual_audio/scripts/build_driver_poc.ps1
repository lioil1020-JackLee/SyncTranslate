param(
    [string]$SamplesRoot = "downloads/driver_samples/Windows-driver-samples",
    [string]$Configuration = "Debug",
    [string]$Platform = "x64",
    [string]$ArtifactsDir = "artifacts/driver/synctranslate_virtual_audio",
    [switch]$EnableSpectreMitigation
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$prereq = & (Join-Path $scriptRoot "check_prereqs.ps1") -Json | ConvertFrom-Json
if (!$prereq.build_ready) {
    throw "WDK/Visual Studio build tools are not ready. Run drivers/synctranslate_virtual_audio/scripts/check_prereqs.ps1 for details."
}

$sysvadDir = Join-Path $SamplesRoot "audio/sysvad"
$solution = Join-Path $sysvadDir "sysvad.sln"
if (!(Test-Path $solution)) {
    throw "Missing SysVAD solution: $solution. Run drivers/synctranslate_virtual_audio/scripts/fetch_sysvad.ps1 first."
}

New-Item -ItemType Directory -Path $ArtifactsDir -Force | Out-Null

& (Join-Path $scriptRoot "apply_sysvad_overlay.ps1") -SysvadDir $sysvadDir

Write-Host "[driver-build] building SysVAD-based SyncTranslate driver"
$spectreMitigation = if ($EnableSpectreMitigation) { "Spectre" } else { "false" }
& $prereq.msbuild $solution /m /p:Configuration=$Configuration /p:Platform=$Platform /p:SpectreMitigation=$spectreMitigation /p:Inf2CatUseLocalTime=true
if ($LASTEXITCODE -ne 0) {
    throw "MSBuild failed with exit code $LASTEXITCODE"
}

$candidatePackages = @(
    (Join-Path $sysvadDir "$Configuration/package"),
    (Join-Path $sysvadDir "$Platform/$Configuration/package"),
    (Join-Path $sysvadDir "$Configuration/$Platform/package")
) | Where-Object { Test-Path $_ }

if (!$candidatePackages -or $candidatePackages.Count -eq 0) {
    $candidatePackages = Get-ChildItem -Path $sysvadDir -Recurse -Directory -Filter "package" -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty FullName
}

if (!$candidatePackages -or $candidatePackages.Count -eq 0) {
    throw "Build completed, but no SysVAD package folder was found under $sysvadDir."
}

$sourcePackage = $candidatePackages | Select-Object -First 1
$targetPackage = Join-Path $ArtifactsDir "package"
if (Test-Path $targetPackage) {
    Remove-Item -LiteralPath $targetPackage -Recurse -Force
}
Copy-Item -Path $sourcePackage -Destination $targetPackage -Recurse

Write-Host "[driver-build] copied package to $targetPackage"
Get-ChildItem -Path $targetPackage | Select-Object Name,Length | Format-Table -AutoSize
