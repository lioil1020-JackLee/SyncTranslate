param(
    [switch]$Json
)

$ErrorActionPreference = "Stop"

$pnpDevices = @(
    Get-PnpDevice -ErrorAction SilentlyContinue |
        Where-Object {
            $_.FriendlyName -like "*SyncTranslate*" -or
            $_.InstanceId -like "*SyncTranslate*" -or
            $_.InstanceId -like "*ROOT\MEDIA*"
        } |
        Select-Object Status, Class, FriendlyName, InstanceId
)

$audioEndpoints = @(
    $pnpDevices |
        Where-Object {
            $_.Class -eq "AudioEndpoint" -and
            $_.FriendlyName -like "*SyncTranslate*"
        }
)

$mediaDevices = @(
    $pnpDevices |
        Where-Object {
            $_.Class -eq "MEDIA" -and
            $_.FriendlyName -like "*SyncTranslate*"
        }
)

$hasVirtualSpeaker = [bool](
    $audioEndpoints |
        Where-Object {
            $_.FriendlyName -like "*SyncTranslate Virtual Speaker*" -or
            $_.FriendlyName -like "*SyncTranslate Virtual Audio Device*"
        } |
        Where-Object { $_.InstanceId -like "SWD\MMDEVAPI\{0.0.0.*" }
)

$hasVirtualMicrophone = [bool](
    $audioEndpoints |
        Where-Object {
            $_.FriendlyName -like "*SyncTranslate Virtual Microphone*" -or
            $_.FriendlyName -like "*SyncTranslate Virtual Audio Device*"
        } |
        Where-Object { $_.InstanceId -like "SWD\MMDEVAPI\{0.0.1.*" }
)

$result = [pscustomobject]@{
    installed = [bool]($mediaDevices.Count -gt 0 -and $audioEndpoints.Count -gt 0)
    media_device_count = $mediaDevices.Count
    audio_endpoint_count = $audioEndpoints.Count
    has_virtual_speaker_endpoint = $hasVirtualSpeaker
    has_virtual_microphone_endpoint = $hasVirtualMicrophone
    devices = $pnpDevices
}

if ($Json) {
    $result | ConvertTo-Json -Depth 5
    exit 0
}

Write-Host "[driver-verify] installed: $($result.installed)"
Write-Host "[driver-verify] media devices: $($result.media_device_count)"
Write-Host "[driver-verify] audio endpoints: $($result.audio_endpoint_count)"
Write-Host "[driver-verify] virtual speaker endpoint: $($result.has_virtual_speaker_endpoint)"
Write-Host "[driver-verify] virtual microphone endpoint: $($result.has_virtual_microphone_endpoint)"
Write-Host ""
$pnpDevices | Format-Table -AutoSize

if (!$result.installed -or !$hasVirtualSpeaker -or !$hasVirtualMicrophone) {
    exit 1
}
