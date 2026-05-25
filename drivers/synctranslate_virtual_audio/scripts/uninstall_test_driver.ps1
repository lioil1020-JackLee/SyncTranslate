param(
    [switch]$WhatIfOnly,
    [switch]$AllowHostInstall
)

$ErrorActionPreference = "Stop"

function Find-PnPUtil {
    $candidates = @(
        (Join-Path $env:windir "Sysnative\pnputil.exe"),
        (Join-Path $env:windir "System32\pnputil.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { return $candidate }
    }
    $cmd = Get-Command pnputil.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return ""
}

$pnputil = Find-PnPUtil
if (!$pnputil) {
    Write-Host "FAIL pnputil.exe was not found."
    exit 1
}

Write-Host "PASS pnputil: $pnputil"
if (!$AllowHostInstall) {
    Write-Host "FAIL Host driver uninstall is blocked by default. Run this only in a disposable Windows VM, or pass -AllowHostInstall."
    exit 1
}

$devconCandidates = @(
    (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "devcon.exe"),
    "${env:ProgramFiles(x86)}\Windows Kits\10\Tools\10.0.26100.0\x64\devcon.exe"
)
$devcon = ""
foreach ($candidate in $devconCandidates) {
    if ($candidate -and (Test-Path $candidate)) {
        $devcon = $candidate
        break
    }
}
if (!$devcon) {
    $cmd = Get-Command devcon.exe -ErrorAction SilentlyContinue
    if ($cmd) { $devcon = $cmd.Source }
}
if ($devcon) {
    Write-Host "PASS removing root devices with devcon: $devcon"
    if ($WhatIfOnly) {
        Write-Host "WARN WhatIfOnly: devcon remove Root\SyncTranslateVirtualAudio"
    }
    else {
        & $devcon remove "Root\SyncTranslateVirtualAudio"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "WARN devcon remove returned exit code $LASTEXITCODE"
        }
        $syncDevices = @(Get-CimInstance Win32_PnPEntity -ErrorAction SilentlyContinue | Where-Object {
            ($_.Name -like "*SyncTranslate*Virtual*") -or
            (($_.PNPDeviceID -like "ROOT\MEDIA\*") -and ($_.Service -like "sysvad_componentizedaudiosample"))
        })
        foreach ($device in $syncDevices) {
            if (!$device.PNPDeviceID) { continue }
            Write-Host "PASS removing stale SyncTranslate device instance: $($device.PNPDeviceID)"
            & $devcon remove "@$($device.PNPDeviceID)"
            if ($LASTEXITCODE -ne 0) {
                Write-Host "WARN devcon remove failed for $($device.PNPDeviceID) with exit code $LASTEXITCODE"
            }
        }
    }
}
else {
    Write-Host "WARN devcon.exe was not found; pnputil will remove packages but may not remove existing root devices."
}

Write-Host "PASS enumerating SyncTranslate driver packages only"
$drivers = & $pnputil /enum-drivers 2>&1
$blocks = @()
$current = @()
foreach ($line in $drivers) {
    if ($line -match "\boem\d+\.inf\b" -and $current.Count -gt 0) {
        $blocks += ,($current -join "`n")
        $current = @()
    }
    $current += $line
}
if ($current.Count -gt 0) { $blocks += ,($current -join "`n") }

$syncDrivers = @($blocks | Where-Object {
    $lower = $_.ToLowerInvariant()
    (
        $lower -match "synctranslate" -or
        $lower -match "componentizedaudiosample\.inf" -or
        $lower -match "componentizedaudiosampleextension\.inf" -or
        $lower -match "componentizedaposample\.inf"
    ) -and (
        $lower -match "provider name:\s*synctranslate" -or
        $lower -match "synctranslate" -or
        $lower -match "componentizedaudiosample\.inf" -or
        $lower -match "componentizedaudiosampleextension\.inf" -or
        $lower -match "componentizedaposample\.inf"
    )
})
if ($syncDrivers.Count -eq 0) {
    Write-Host "WARN No SyncTranslate driver package was found."
}
else {
    foreach ($block in $syncDrivers) {
        $published = ""
        if ($block -match "(?i)\boem\d+\.inf\b") { $published = $Matches[0] }
        if (!$published) { continue }
        Write-Host "PASS candidate SyncTranslate package: $published"
        if ($WhatIfOnly) {
            Write-Host "WARN WhatIfOnly: pnputil /delete-driver $published /uninstall /force"
        }
        else {
            & $pnputil /delete-driver $published /uninstall /force
            if ($LASTEXITCODE -ne 0) {
                Write-Host "WARN pnputil failed for $published with exit code $LASTEXITCODE"
            }
        }
    }
}

foreach ($root in @("HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\MMDevices\Audio\Render", "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\MMDevices\Audio\Capture")) {
    if (!(Test-Path $root)) {
        continue
    }
    Get-ChildItem $root -ErrorAction SilentlyContinue | ForEach-Object {
        $props = Get-ItemProperty (Join-Path $_.PSPath "Properties") -ErrorAction SilentlyContinue
        $values = @($props.PSObject.Properties | Where-Object { $_.Value -is [string] } | ForEach-Object { [string]$_.Value })
        $joined = $values -join "`n"
        if ($joined -match "SyncTranslate|Root\\SyncTranslateVirtualAudio|Root\\Sysvad_ComponentizedAudioSample|ROOT\\MEDIA\\0000") {
            if ($WhatIfOnly) {
                Write-Host "WARN WhatIfOnly: remove stale MMDevice endpoint cache $($_.PSChildName)"
            }
            else {
                Write-Host "PASS removing stale MMDevice endpoint cache: $($_.PSChildName)"
                Remove-Item -LiteralPath $_.PSPath -Recurse -Force -ErrorAction SilentlyContinue
            }
        }
    }
}
Write-Host "PASS uninstall complete. Reboot Windows or restart AudioEndpointBuilder before reinstalling."
exit 0
