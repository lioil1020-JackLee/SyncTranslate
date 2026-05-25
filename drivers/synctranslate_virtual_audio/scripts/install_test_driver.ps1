param(
    [string]$PackageDir = "artifacts/driver/synctranslate_virtual_audio/package",
    [string]$CertificatePath = "artifacts/driver/synctranslate_virtual_audio/SyncTranslateVirtualAudioTest.cer",
    [switch]$AllowHostInstall
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

function Test-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Test-TestMode {
    $bcdedit = Join-Path $env:windir "System32\bcdedit.exe"
    if (!(Test-Path $bcdedit)) { return $false }
    $output = & $bcdedit /enum 2>&1
    return [bool]($output | Where-Object { $_ -match "testsigning\s+Yes|testsigning\s+on" })
}

if (!(Test-Administrator)) {
    Write-Host "FAIL Administrator PowerShell is required to install the test-signed driver."
    exit 1
}
if (!(Test-TestMode)) {
    Write-Host "WARN Windows Test Mode is not enabled."
    Write-Host "Run as Administrator, then reboot: bcdedit /set testsigning on"
    Write-Host "This script will not force a reboot."
    exit 1
}
if (!(Test-Path $PackageDir)) {
    Write-Host "FAIL Driver package directory not found: $PackageDir"
    exit 1
}

Write-Host "PASS installing SyncTranslate test driver package with pnputil/devcon path"
& (Join-Path $scriptRoot "install_driver_package.ps1") -PackageDir $PackageDir -CertificatePath $CertificatePath -RequireTestSigning -AllowHostInstall:$AllowHostInstall
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAIL install_driver_package.ps1 failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

& (Join-Path $scriptRoot "verify_driver_format.ps1")
exit $LASTEXITCODE
