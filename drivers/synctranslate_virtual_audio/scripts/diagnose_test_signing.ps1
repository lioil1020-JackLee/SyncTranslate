param()

$ErrorActionPreference = "Continue"

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host "== $Title =="
}

Write-Section "Administrator"
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
Write-Host "Identity: $($identity.Name)"
Write-Host "IsAdmin: $($principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator))"

Write-Section "Firmware / Secure Boot"
try {
    $computerInfo = Get-ComputerInfo -Property BiosFirmwareType,OsName,OsVersion,WindowsProductName,WindowsVersion
    $computerInfo | Format-List
}
catch {
    Write-Warning $_.Exception.Message
}

try {
    $secureBootState = Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\SecureBoot\State" -ErrorAction Stop
    Write-Host "UEFISecureBootEnabled registry value: $($secureBootState.UEFISecureBootEnabled)"
    Write-Host "PolicyPublisher: $($secureBootState.PolicyPublisher -join ',')"
    Write-Host "PolicyVersion: $($secureBootState.PolicyVersion)"
}
catch {
    Write-Warning "Unable to read Secure Boot registry state: $($_.Exception.Message)"
}

try {
    $secureBoot = Confirm-SecureBootUEFI
    Write-Host "Confirm-SecureBootUEFI: $secureBoot"
}
catch {
    Write-Warning "Confirm-SecureBootUEFI failed: $($_.Exception.Message)"
}

Write-Section "BCD"
try {
    & "$env:windir\System32\bcdedit.exe" /enum
}
catch {
    Write-Warning "bcdedit failed: $($_.Exception.Message)"
}

Write-Section "Device Guard / VBS"
try {
    Get-CimInstance -ClassName Win32_DeviceGuard -Namespace root\Microsoft\Windows\DeviceGuard |
        Select-Object `
            VirtualizationBasedSecurityStatus,
            SecurityServicesConfigured,
            SecurityServicesRunning,
            CodeIntegrityPolicyEnforcementStatus,
            UsermodeCodeIntegrityPolicyEnforcementStatus |
        Format-List
}
catch {
    Write-Warning "Device Guard query failed: $($_.Exception.Message)"
}

Write-Section "BitLocker"
try {
    & manage-bde -status C:
}
catch {
    Write-Warning "manage-bde failed: $($_.Exception.Message)"
}

Write-Host ""
Write-Host "Hint: if UEFISecureBootEnabled is 1 or Confirm-SecureBootUEFI is True, Windows will block TESTSIGNING."
