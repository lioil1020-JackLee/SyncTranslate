param(
    [switch]$Disable
)

$ErrorActionPreference = "Stop"

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (!$principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Run this script from an elevated PowerShell."
}

function Invoke-TestSigningBcdEdit {
    param([string]$State)
    $output = & bcdedit /set TESTSIGNING $State 2>&1
    $exitCode = $LASTEXITCODE
    $output | ForEach-Object { Write-Host $_ }
    if ($exitCode -ne 0) {
        $joined = ($output | Out-String)
        if ($joined -match "安全開機|Secure Boot|protected by Secure Boot policy") {
            throw "Windows Secure Boot is blocking TESTSIGNING. Disable Secure Boot in UEFI/BIOS first, then run this script again and reboot. If BitLocker is enabled, suspend or save the recovery key before changing Secure Boot."
        }
        throw "bcdedit failed with exit code $exitCode."
    }
}

function Get-WindowsSecureBootEnabled {
    try {
        $state = Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\SecureBoot\State" -ErrorAction Stop
        if ($null -ne $state.UEFISecureBootEnabled) {
            return [bool]$state.UEFISecureBootEnabled
        }
    }
    catch {
    }
    try {
        return [bool](Confirm-SecureBootUEFI)
    }
    catch {
        return $null
    }
}

if ($Disable) {
    Write-Host "[driver-testmode] disabling Windows test signing"
    Invoke-TestSigningBcdEdit -State "OFF"
}
else {
    $secureBootEnabled = Get-WindowsSecureBootEnabled
    if ($secureBootEnabled -eq $true) {
        throw "Windows reports Secure Boot is enabled, so TESTSIGNING cannot be enabled. Run diagnose_test_signing.ps1 to confirm the Windows-side state, then disable Secure Boot in UEFI/BIOS and run this script again."
    }
    Write-Host "[driver-testmode] enabling Windows test signing"
    Invoke-TestSigningBcdEdit -State "ON"
}

Write-Host "[driver-testmode] reboot is required for this change to take effect."
