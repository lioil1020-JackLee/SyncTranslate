param(
    [string]$LogPath = "$env:ProgramData\SyncTranslate\Logs\driver-cleanup.log",
    [switch]$WhatIfOnly
)

$ErrorActionPreference = "Stop"

function Write-CleanupLog {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host $Message
    Add-Content -Path $script:LogPath -Value "[$timestamp] $Message" -Encoding UTF8
}

function Invoke-LoggedCommand {
    param(
        [string]$FilePath,
        [string[]]$Arguments
    )
    $commandLine = "$FilePath $($Arguments -join ' ')"
    Write-CleanupLog "[driver-cleanup] exec: $commandLine"
    if ($WhatIfOnly) {
        return 0
    }
    $output = & $FilePath @Arguments 2>&1
    $exitCode = $LASTEXITCODE
    foreach ($line in $output) {
        Write-CleanupLog $line
    }
    Write-CleanupLog "[driver-cleanup] exit: $exitCode"
    return $exitCode
}

function Split-PnPBlocks {
    param(
        [string[]]$Lines,
        [string]$StartPattern
    )
    $blocks = @()
    $current = New-Object System.Collections.Generic.List[string]
    foreach ($line in $Lines) {
        if ($line -match $StartPattern) {
            if ($current.Count -gt 0) {
                $blocks += ,(@($current.ToArray()))
                $current.Clear()
            }
        }
        $current.Add($line) | Out-Null
    }
    if ($current.Count -gt 0) {
        $blocks += ,(@($current.ToArray()))
    }
    return $blocks
}

function Get-FieldValue {
    param(
        [string[]]$Block,
        [string]$Field
    )
    foreach ($line in $Block) {
        if ($line -match "^\s*$([regex]::Escape($Field))\s*:\s*(.+)$") {
            return $Matches[1].Trim()
        }
    }
    return ""
}

$script:LogPath = $LogPath
$logParent = Split-Path -Parent $script:LogPath
if ($logParent) {
    New-Item -ItemType Directory -Path $logParent -Force | Out-Null
}
Set-Content -Path $script:LogPath -Value "[driver-cleanup] started $(Get-Date -Format o)" -Encoding UTF8

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (!$WhatIfOnly -and !$principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Run this script from an elevated PowerShell."
}

$pnputil = Join-Path $env:windir "System32\pnputil.exe"
if (!(Test-Path $pnputil)) {
    $cmd = Get-Command pnputil.exe -ErrorAction SilentlyContinue
    if (!$cmd) {
        throw "pnputil.exe was not found."
    }
    $pnputil = $cmd.Source
}

Write-CleanupLog "[driver-cleanup] pnputil: $pnputil"

$deviceLines = & $pnputil /enum-devices /class MEDIA
$deviceBlocks = Split-PnPBlocks -Lines $deviceLines -StartPattern "^\s*Instance ID:"
$syncDevices = @()
foreach ($block in $deviceBlocks) {
    $text = $block -join "`n"
    if ($text -match "SyncTranslate") {
        $instanceId = Get-FieldValue -Block $block -Field "Instance ID"
        $driverName = Get-FieldValue -Block $block -Field "Driver Name"
        $status = Get-FieldValue -Block $block -Field "Status"
        $syncDevices += [pscustomobject]@{
            InstanceId = $instanceId
            DriverName = $driverName
            Status = $status
            Raw = $text
        }
    }
}

Write-CleanupLog "[driver-cleanup] SyncTranslate media devices found: $($syncDevices.Count)"
foreach ($device in $syncDevices) {
    Write-CleanupLog "[driver-cleanup] device: $($device.InstanceId) driver=$($device.DriverName) status=$($device.Status)"
    if ($device.InstanceId) {
        [void](Invoke-LoggedCommand -FilePath $pnputil -Arguments @("/remove-device", $device.InstanceId))
    }
}

$driverLines = & $pnputil /enum-drivers
$driverBlocks = Split-PnPBlocks -Lines $driverLines -StartPattern "^\s*Published Name:"
$syncDrivers = @()
foreach ($block in $driverBlocks) {
    $provider = Get-FieldValue -Block $block -Field "Provider Name"
    $original = Get-FieldValue -Block $block -Field "Original Name"
    if ($provider -eq "SyncTranslate" -and $original -match "componentizedaudiosample\.inf") {
        $published = Get-FieldValue -Block $block -Field "Published Name"
        $syncDrivers += [pscustomobject]@{
            PublishedName = $published
            ProviderName = $provider
            OriginalName = $original
            Raw = ($block -join "`n")
        }
    }
}

Write-CleanupLog "[driver-cleanup] SyncTranslate driver packages found: $($syncDrivers.Count)"
foreach ($driver in $syncDrivers) {
    Write-CleanupLog "[driver-cleanup] driver package: $($driver.PublishedName) provider=$($driver.ProviderName) original=$($driver.OriginalName)"
    if ($driver.PublishedName) {
        [void](Invoke-LoggedCommand -FilePath $pnputil -Arguments @("/delete-driver", $driver.PublishedName, "/uninstall", "/force"))
    }
}

if (!$WhatIfOnly) {
    $remainingDevices = (& $pnputil /enum-devices /class MEDIA) | Select-String -Pattern "SyncTranslate" -SimpleMatch
    $remainingDrivers = (& $pnputil /enum-drivers) | Select-String -Pattern "Provider Name:\s+SyncTranslate"
    Write-CleanupLog "[driver-cleanup] remaining SyncTranslate media markers: $(@($remainingDevices).Count)"
    Write-CleanupLog "[driver-cleanup] remaining SyncTranslate driver provider markers: $(@($remainingDrivers).Count)"

    if (@($remainingDevices).Count -gt 0 -or @($remainingDrivers).Count -gt 0) {
        throw "SyncTranslate driver cleanup did not fully clear all markers. See $script:LogPath"
    }
}

Write-CleanupLog "[driver-cleanup] complete"
