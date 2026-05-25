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

function Clear-StaleMmDeviceCache {
    param([switch]$AsSystem)

    $cleanupBlock = {
        foreach ($root in @("HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\MMDevices\Audio\Render", "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\MMDevices\Audio\Capture")) {
            if (!(Test-Path $root)) {
                continue
            }
            Get-ChildItem $root -ErrorAction SilentlyContinue | ForEach-Object {
                $propsPath = Join-Path $_.PSPath "Properties"
                $props = Get-ItemProperty $propsPath -ErrorAction SilentlyContinue
                $values = @($props.PSObject.Properties | Where-Object { $_.Value -is [string] } | ForEach-Object { [string]$_.Value })
                $joined = $values -join "`n"
                if ($joined -match "SyncTranslate|Root\\SyncTranslateVirtualAudio|Root\\Sysvad_ComponentizedAudioSample|SYSVAD") {
                    Write-Host "[driver-uninstall] removing stale MMDevice endpoint cache: $($_.PSChildName)"
                    Remove-Item -LiteralPath $_.PSPath -Recurse -Force -ErrorAction Stop
                }
            }
        }
    }

    if (!$AsSystem) {
        try {
            & $cleanupBlock
            return
        }
        catch {
            Write-Host "[driver-uninstall] MMDevice cache cleanup requires SYSTEM rights: $($_.Exception.Message)"
        }
    }

    if ($WhatIfOnly) {
        Write-Host "[driver-uninstall] WhatIfOnly set; SYSTEM MMDevice cache cleanup was skipped."
        return
    }

    $taskName = "SyncTranslateClearStaleMmDeviceCache"
    $tempScript = Join-Path $env:ProgramData "SyncTranslate\clear_stale_mmdevice_cache.ps1"
    New-Item -ItemType Directory -Path (Split-Path -Parent $tempScript) -Force | Out-Null
    @'
$ErrorActionPreference = "Continue"
foreach ($root in @("HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\MMDevices\Audio\Render", "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\MMDevices\Audio\Capture")) {
    if (!(Test-Path $root)) { continue }
    Get-ChildItem $root -ErrorAction SilentlyContinue | ForEach-Object {
        $propsPath = Join-Path $_.PSPath "Properties"
        $props = Get-ItemProperty $propsPath -ErrorAction SilentlyContinue
        $values = @($props.PSObject.Properties | Where-Object { $_.Value -is [string] } | ForEach-Object { [string]$_.Value })
        $joined = $values -join "`n"
        if ($joined -match "SyncTranslate|Root\\SyncTranslateVirtualAudio|Root\\Sysvad_ComponentizedAudioSample|SYSVAD") {
            Remove-Item -LiteralPath $_.PSPath -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}
'@ | Set-Content -Path $tempScript -Encoding ASCII
    $startTime = (Get-Date).AddMinutes(1).ToString("HH:mm")
    schtasks.exe /Create /TN $taskName /SC ONCE /ST $startTime /RU SYSTEM /RL HIGHEST /TR "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$tempScript`"" /F | Out-Null
    schtasks.exe /Run /TN $taskName | Out-Null
    Start-Sleep -Seconds 3
    schtasks.exe /Delete /TN $taskName /F | Out-Null
    Write-Host "[driver-uninstall] SYSTEM MMDevice cache cleanup requested."
}

$deviceInstances = @(Get-CimInstance Win32_PnPEntity -ErrorAction SilentlyContinue | Where-Object {
    ($_.PNPDeviceID -like "ROOT\MEDIA\*") -and
    (($_.Service -like "sysvad_componentizedaudiosample") -or ($_.Name -match "SyncTranslate|SYSVAD"))
})
foreach ($device in $deviceInstances) {
    Write-Host "[driver-uninstall] removing stale media device instance: $($device.PNPDeviceID)"
    if (!$WhatIfOnly) {
        & $pnputil /remove-device $device.PNPDeviceID 2>&1 | ForEach-Object { Write-Host $_ }
    }
}

$drivers = & $pnputil /enum-drivers
$published = @()
$currentLines = New-Object System.Collections.Generic.List[string]
foreach ($line in $drivers) {
    if ($line -match ":\s*(oem\d+\.inf)\s*$") {
        if ($currentLines.Count -gt 0) {
            $block = ($currentLines -join "`n")
            if ($block -match ":\s*(oem\d+\.inf)") {
                $published += [pscustomobject]@{
                    PublishedName = $Matches[1]
                    Block = $block
                }
            }
            $currentLines.Clear()
        }
    }
    $currentLines.Add($line) | Out-Null
}
if ($currentLines.Count -gt 0) {
    $block = ($currentLines -join "`n")
    if ($block -match ":\s*(oem\d+\.inf)") {
        $published += [pscustomobject]@{
            PublishedName = $Matches[1]
            Block = $block
        }
    }
}

$matches = $published | Where-Object {
    ($_.Block -match [Regex]::Escape($ProviderName)) -or
    ($_.Block -match "componentizedaudiosample|componentizedaposample|synctranslate")
}

if (!$matches) {
    Write-Host "[driver-uninstall] no matching driver package was found"
}
else {
    $matches | Select-Object PublishedName,Block | Format-Table -AutoSize

    if ($WhatIfOnly) {
        Write-Host "[driver-uninstall] WhatIfOnly set; no driver packages were removed."
    }
    else {
        foreach ($driver in $matches) {
            Write-Host "[driver-uninstall] removing $($driver.PublishedName)"
            & $pnputil /delete-driver $driver.PublishedName /uninstall /force
        }
    }
}

Clear-StaleMmDeviceCache

Write-Host "[driver-uninstall] done. Reboot Windows or restart AudioEndpointBuilder before reinstalling."
