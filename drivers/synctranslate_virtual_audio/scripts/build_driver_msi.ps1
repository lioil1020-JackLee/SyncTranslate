param(
    [string]$SamplesRoot = "downloads/driver_samples/Windows-driver-samples",
    [string]$Configuration = "Debug",
    [string]$Platform = "x64",
    [string]$ArtifactsDir = "artifacts/driver/synctranslate_virtual_audio",
    [string]$ProductVersion = "2.1.0",
    [switch]$SkipFetch,
    [switch]$SkipSign
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

$prereq = & (Join-Path $scriptRoot "check_prereqs.ps1") -Json | ConvertFrom-Json
if (!$prereq.build_ready) {
    throw "Driver build prerequisites are incomplete. Install Visual Studio C++ tools, Windows SDK, and Windows Driver Kit, then rerun check_prereqs.ps1."
}
if (!$prereq.msi_ready) {
    throw "MSI packaging prerequisites are incomplete. Install wix.exe or install the .NET SDK so WiX can be installed as a dotnet tool."
}

if (!$SkipFetch) {
    & (Join-Path $scriptRoot "fetch_sysvad.ps1") -Destination $SamplesRoot
}

& (Join-Path $scriptRoot "build_driver_poc.ps1") `
    -SamplesRoot $SamplesRoot `
    -Configuration $Configuration `
    -Platform $Platform `
    -ArtifactsDir $ArtifactsDir

$packageDir = Join-Path $ArtifactsDir "package"
$certificatePath = Join-Path $ArtifactsDir "SyncTranslateVirtualAudioTest.cer"

if (!$SkipSign) {
    & (Join-Path $scriptRoot "sign_driver_package.ps1") `
        -PackageDir $packageDir `
        -CertificatePath $certificatePath
}

& (Join-Path $scriptRoot "validate_driver_package.ps1") `
    -PackageDir $packageDir `
    -CertificatePath $certificatePath `
    -SkipSignatureVerify:$SkipSign

& (Join-Path $scriptRoot "package_driver_msi.ps1") `
    -PackageDir $packageDir `
    -CertificatePath $certificatePath `
    -OutputMsi (Join-Path $ArtifactsDir "SyncTranslateVirtualAudioDriver.msi") `
    -ProductVersion $ProductVersion

Write-Host "[driver-msi] full Phase 3 MSI pipeline complete"
