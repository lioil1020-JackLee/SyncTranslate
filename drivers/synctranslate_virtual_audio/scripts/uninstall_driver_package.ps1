param(
    [string]$ProviderName = "SyncTranslate",
    [switch]$WhatIfOnly
)

$ErrorActionPreference = "Stop"

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (!$WhatIfOnly -and !$principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Run this script from an elevated PowerShell."
}

$sysnativePnputil = Join-Path $env:windir "Sysnative\pnputil.exe"
$system32Pnputil = Join-Path $env:windir "System32\pnputil.exe"
if (Test-Path $sysnativePnputil) {
    $pnputil = (Resolve-Path $sysnativePnputil).Path
}
elseif (Test-Path $system32Pnputil) {
    $pnputil = (Resolve-Path $system32Pnputil).Path
}
else {
    $cmd = Get-Command pnputil.exe -ErrorAction SilentlyContinue
    if (!$cmd) {
        throw "pnputil.exe was not found."
    }
    $pnputil = $cmd.Source
}

$drivers = & $pnputil /enum-drivers
$published = @()
$current = @{}
foreach ($line in $drivers) {
    if ($line -match "Published Name\s*:\s*(.+)$") {
        if ($current.Count -gt 0) {
            $published += [pscustomobject]$current
        }
        $current = @{ PublishedName = $Matches[1].Trim() }
        continue
    }
    if ($line -match "Provider Name\s*:\s*(.+)$") {
        $current.ProviderName = $Matches[1].Trim()
    }
    if ($line -match "Original Name\s*:\s*(.+)$") {
        $current.OriginalName = $Matches[1].Trim()
    }
}
if ($current.Count -gt 0) {
    $published += [pscustomobject]$current
}

$matches = $published | Where-Object {
    ($_.ProviderName -eq $ProviderName) -and
    ($_.OriginalName -match "ComponentizedAudioSample|SyncTranslate")
}

if (!$matches) {
    Write-Host "[driver-uninstall] no matching driver package was found"
    exit 0
}

$matches | Select-Object PublishedName,ProviderName,OriginalName | Format-Table -AutoSize

if ($WhatIfOnly) {
    Write-Host "[driver-uninstall] WhatIfOnly set; no driver packages were removed."
    exit 0
}

foreach ($driver in $matches) {
    Write-Host "[driver-uninstall] removing $($driver.PublishedName)"
    & $pnputil /delete-driver $driver.PublishedName /uninstall /force
}
