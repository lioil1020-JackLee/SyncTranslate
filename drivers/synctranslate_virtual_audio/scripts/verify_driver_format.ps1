param(
    [string]$JsonOutput = ""
)

$ErrorActionPreference = "Continue"
# Usage: verify_driver_format.ps1 -JsonOutput downloads/validation/driver_format.json

function New-ItemResult {
    param([string]$Name, [string]$Status, [string]$Message, [hashtable]$Details = @{})
    [ordered]@{ name = $Name; status = $Status; message = $Message; details = $Details }
}

function Write-Report {
    param([array]$Items)
    $status = if ($Items.status -contains "FAIL") { "FAIL" } elseif ($Items.status -contains "WARN") { "WARN" } else { "PASS" }
    $report = [ordered]@{
        name = "SyncTranslate driver format verification"
        status = $status
        driver_format_expected = "48000Hz PCM16 2ch"
        items = $Items
    }
    if ($JsonOutput) {
        $parent = Split-Path -Parent $JsonOutput
        if ($parent) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
        $report | ConvertTo-Json -Depth 6 | Set-Content -Path $JsonOutput -Encoding UTF8
        Write-Host "JSON written: $JsonOutput"
    }
    Write-Host "$status SyncTranslate driver format verification"
    foreach ($item in $Items) {
        Write-Host "[$($item.status)] $($item.name): $($item.message)"
    }
    if ($status -eq "FAIL") { exit 1 }
    exit 0
}

$items = @()
$mediaDevices = @()
$interfaceNames = @()
$presentEndpointDevices = @()
$cachedEndpointNames = @()

try {
    $mediaDevices = @(Get-CimInstance Win32_PnPEntity -ErrorAction Stop | Where-Object {
        ($_.PNPClass -eq "MEDIA") -and
        ($_.PNPDeviceID -like "ROOT\MEDIA\*") -and
        (
            $_.Name -like "*SyncTranslate*Virtual*" -or
            $_.Service -like "sysvad_componentizedaudiosample" -or
            (($_.HardwareID -join "`n") -like "*Root\SyncTranslateVirtualAudio*")
        )
    })
}
catch {
    $items += New-ItemResult "pnp_query" "WARN" "Unable to query MEDIA PnP devices: $($_.Exception.Message)"
}

try {
    $interfaceNames = @(Get-ChildItem "HKLM:\SYSTEM\CurrentControlSet\Control\DeviceClasses" -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "*ROOT#MEDIA*" -and ($_.Name -like "*#WaveSpeaker" -or $_.Name -like "*#WaveMicArray1") } |
        ForEach-Object { $_.Name })
}
catch {
    $items += New-ItemResult "device_interface_query" "WARN" "Unable to query KS device interfaces: $($_.Exception.Message)"
}

$speakerInterfaces = @($interfaceNames | Where-Object { $_ -like "*#WaveSpeaker" })
$microphoneInterfaces = @($interfaceNames | Where-Object { $_ -like "*#WaveMicArray1" })
$items += New-ItemResult "ks_wave_speaker_interface" ($(if ($speakerInterfaces.Count -gt 0) { "PASS" } else { "FAIL" })) ($(if ($speakerInterfaces.Count -gt 0) { "SyncTranslate WaveSpeaker KS interface exists." } else { "SyncTranslate WaveSpeaker KS interface is missing." })) @{ interface_count = $speakerInterfaces.Count }
$items += New-ItemResult "ks_wave_microphone_interface" ($(if ($microphoneInterfaces.Count -gt 0) { "PASS" } else { "FAIL" })) ($(if ($microphoneInterfaces.Count -gt 0) { "SyncTranslate WaveMicArray1 KS interface exists." } else { "SyncTranslate WaveMicArray1 KS interface is missing." })) @{ interface_count = $microphoneInterfaces.Count }

try {
    $presentEndpointDevices = @(Get-PnpDevice -PresentOnly -Class AudioEndpoint -ErrorAction Stop |
        Where-Object { $_.FriendlyName -like "*SyncTranslate*" })
}
catch {
    $items += New-ItemResult "audio_endpoint_query" "WARN" "Unable to query present AudioEndpoint PnP devices: $($_.Exception.Message)"
}

$renderEndpointNames = @($presentEndpointDevices |
    Where-Object {
        $_.InstanceId -like "SWD\MMDEVAPI\{0.0.0.*" -or
        $_.FriendlyName -like "*Virtual Speaker*" -or
        $_.FriendlyName -like "*Speaker*" -or
        $_.FriendlyName -like "*Speakers*" -or
        $_.FriendlyName -like "*Headphone*" -or
        $_.FriendlyName -like "*喇叭*" -or
        $_.FriendlyName -like "*耳機*"
    } |
    ForEach-Object { $_.FriendlyName } |
    Select-Object -Unique)

$captureEndpointNames = @($presentEndpointDevices |
    Where-Object {
        $_.InstanceId -like "SWD\MMDEVAPI\{0.0.1.*" -or
        $_.FriendlyName -like "*Virtual Microphone*" -or
        $_.FriendlyName -like "*Microphone*" -or
        $_.FriendlyName -like "*Mic*" -or
        $_.FriendlyName -like "*麥克風*"
    } |
    ForEach-Object { $_.FriendlyName } |
    Select-Object -Unique)

try {
    foreach ($root in @("HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\MMDevices\Audio\Render", "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\MMDevices\Audio\Capture")) {
        if (Test-Path $root) {
            Get-ChildItem $root -ErrorAction SilentlyContinue | ForEach-Object {
                $props = Get-ItemProperty (Join-Path $_.PSPath "Properties") -ErrorAction SilentlyContinue
                $matches = @($props.PSObject.Properties | Where-Object { $_.Value -like "*SyncTranslate*" } | ForEach-Object { [string]$_.Value })
                $cachedEndpointNames += $matches
            }
        }
    }
}
catch {
    $items += New-ItemResult "mmdevice_registry_query" "WARN" "Unable to query MMDevice registry endpoint cache: $($_.Exception.Message)"
}

$hasSpeakerEndpoint = ($renderEndpointNames.Count -gt 0)
$hasMicrophoneEndpoint = ($captureEndpointNames.Count -gt 0)
$items += New-ItemResult "virtual_speaker_endpoint" ($(if ($hasSpeakerEndpoint) { "PASS" } else { "FAIL" })) ($(if ($hasSpeakerEndpoint) { "Present SyncTranslate render AudioEndpoint exists." } else { "Present SyncTranslate render AudioEndpoint is missing; it will not appear under Windows audio output devices." })) @{
    endpoint_names = $renderEndpointNames
    present_endpoint_count = $presentEndpointDevices.Count
    cached_endpoint_names = @($cachedEndpointNames | Select-Object -Unique)
}
$items += New-ItemResult "virtual_microphone_endpoint" ($(if ($hasMicrophoneEndpoint) { "PASS" } else { "FAIL" })) ($(if ($hasMicrophoneEndpoint) { "Present SyncTranslate capture AudioEndpoint exists." } else { "Present SyncTranslate capture AudioEndpoint is missing; it will not appear under Windows audio input devices." })) @{
    endpoint_names = $captureEndpointNames
    present_endpoint_count = $presentEndpointDevices.Count
    cached_endpoint_names = @($cachedEndpointNames | Select-Object -Unique)
}

try {
    $audioDevices = @(Get-CimInstance Win32_SoundDevice -ErrorAction Stop | Where-Object { $_.Name -like "*SyncTranslate*" })
    $items += New-ItemResult "sound_devices" ($(if ($audioDevices.Count -gt 0) { "PASS" } else { "WARN" })) "Found $($audioDevices.Count) SyncTranslate sound device record(s)." @{ names = @($audioDevices | ForEach-Object { $_.Name }) }
    $problemDevices = @($mediaDevices | Where-Object { [int]($_.ConfigManagerErrorCode) -ne 0 -or $_.Status -notin @("OK", $null, "") })
    $items += New-ItemResult "single_sync_translate_device" ($(if ($mediaDevices.Count -le 1) { "PASS" } else { "FAIL" })) "SyncTranslate MEDIA device instances: $($mediaDevices.Count)." @{ ids = @($mediaDevices | ForEach-Object { $_.PNPDeviceID }) }
    $items += New-ItemResult "no_problem_sync_translate_devices" ($(if ($problemDevices.Count -eq 0) { "PASS" } else { "FAIL" })) "SyncTranslate MEDIA problem device instances: $($problemDevices.Count)." @{ ids = @($problemDevices | ForEach-Object { "$($_.PNPDeviceID):$($_.ConfigManagerErrorCode):$($_.Status)" }) }
}
catch {
$items += New-ItemResult "sound_devices" "WARN" "Unable to query Win32_SoundDevice: $($_.Exception.Message)"
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..") -ErrorAction SilentlyContinue
$formatProbe = if ($repoRoot) { Join-Path $repoRoot.Path "tools\validation\wasapi_endpoint_format.py" } else { "" }
if ($formatProbe -and (Test-Path $formatProbe)) {
    $formatJson = Join-Path ([System.IO.Path]::GetTempPath()) ("synctranslate_wasapi_format_{0}.json" -f ([System.Guid]::NewGuid().ToString("N")))
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    $python = Get-Command python -ErrorAction SilentlyContinue
    try {
        if ($uv) {
            Push-Location $repoRoot.Path
            try {
                & $uv.Source run python $formatProbe --json $formatJson | Out-Host
            }
            finally {
                Pop-Location
            }
        }
        elseif ($python) {
            & $python.Source $formatProbe --json $formatJson | Out-Host
        }
        else {
            throw "Neither uv nor python was found on PATH."
        }
        if (!(Test-Path $formatJson)) {
            throw "WASAPI format probe did not write JSON output."
        }
        $formatReport = Get-Content -Raw -Path $formatJson -Encoding UTF8 | ConvertFrom-Json
        $endpointDetails = @()
        if ($formatReport.endpoints) {
            $endpointDetails = @($formatReport.endpoints)
        }
        $items += New-ItemResult "endpoint_device_format" ([string]$formatReport.status) ([string]$formatReport.message) @{
            expected_sample_rate = 48000
            expected_bit_depth = 16
            expected_channels = 2
            expected_dtype = "PCM16"
            status = [string]$formatReport.status
            endpoints = $endpointDetails
            shared_mix_formats = @($formatReport.shared_mix_formats)
            source = "tools/validation/wasapi_endpoint_format.py"
        }
    }
    catch {
        $items += New-ItemResult "endpoint_device_format" "WARN" "WASAPI endpoint device format probe could not complete: $($_.Exception.Message)" @{
            expected_sample_rate = 48000
            expected_bit_depth = 16
            expected_channels = 2
            expected_dtype = "PCM16"
            status = "UNKNOWN"
            source = "tools/validation/wasapi_endpoint_format.py"
        }
    }
    finally {
        Remove-Item -LiteralPath $formatJson -Force -ErrorAction SilentlyContinue
    }
}
else {
    $items += New-ItemResult "endpoint_device_format" "WARN" "WASAPI endpoint device format probe is unavailable. Expected 48000Hz PCM16 2ch." @{
        expected_sample_rate = 48000
        expected_bit_depth = 16
        expected_channels = 2
        expected_dtype = "PCM16"
        status = "UNKNOWN"
    }
}

Write-Report $items
