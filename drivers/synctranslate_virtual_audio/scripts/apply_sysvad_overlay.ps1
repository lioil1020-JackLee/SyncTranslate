param(
    [string]$SysvadDir = "downloads/driver_samples/Windows-driver-samples/audio/sysvad",
    [string]$HardwareId = "Root\SyncTranslateVirtualAudio"
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

if (!(Test-Path $SysvadDir)) {
    throw "Missing SysVAD directory: $SysvadDir. Run fetch_sysvad.ps1 first."
}

$componentInf = Join-Path $SysvadDir "TabletAudioSample/ComponentizedAudioSample.inx"
$extensionInf = Join-Path $SysvadDir "TabletAudioSample/ComponentizedAudioSampleExtension.inx"
if (!(Test-Path $componentInf)) {
    throw "Missing ComponentizedAudioSample.inx: $componentInf"
}

function Update-TextFile {
    param(
        [string]$Path,
        [object[]]$Replacements
    )
    $text = Get-Content -Raw -Encoding UTF8 $Path
    foreach ($pair in $Replacements) {
        if ($pair -is [System.Array] -and $pair.Count -ge 2) {
            $oldValue = [string]$pair[0]
            $newValue = [string]$pair[1]
        }
        else {
            Write-Warning "Invalid replacement pair in $Path. Skip one item."
            continue
        }
        $text = $text.Replace($oldValue, $newValue)
    }
    Set-Content -Path $Path -Value $text -Encoding ASCII
}

function Update-TextFileRegex {
    param(
        [string]$Path,
        [object[]]$Replacements
    )
    $text = Get-Content -Raw -Encoding UTF8 $Path
    $options = [System.Text.RegularExpressions.RegexOptions]::Singleline -bor [System.Text.RegularExpressions.RegexOptions]::Multiline
    foreach ($pair in $Replacements) {
        if ($pair -is [System.Array] -and $pair.Count -ge 2) {
            $pattern = [string]$pair[0]
            $replacement = [string]$pair[1]
        }
        else {
            Write-Warning "Invalid regex replacement pair in $Path. Skip one item."
            continue
        }
        $text = [regex]::Replace($text, $pattern, $replacement, $options)
    }
    Set-Content -Path $Path -Value $text -Encoding ASCII
}

function Set-InfStringDefinition {
    param(
        [string]$Path,
        [string]$Name,
        [string]$Value
    )

    $lines = [System.Collections.Generic.List[string]]::new()
    [string[]](Get-Content -Path $Path -Encoding UTF8) | ForEach-Object { [void]$lines.Add($_) }

    $stringsIndex = -1
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match '^\[Strings\]\s*$') {
            $stringsIndex = $i
            break
        }
    }
    if ($stringsIndex -lt 0) {
        throw "Missing [Strings] section in INF template: $Path"
    }

    $definition = "$Name = `"$Value`""
    $existingIndex = -1
    for ($i = $stringsIndex + 1; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match '^\[') {
            break
        }
        if ($lines[$i] -match ("^\s*" + [regex]::Escape($Name) + "\s*=")) {
            $existingIndex = $i
            break
        }
    }

    if ($existingIndex -ge 0) {
        $lines[$existingIndex] = $definition
    }
    else {
        $insertIndex = $stringsIndex + 1
        while ($insertIndex -lt $lines.Count -and $lines[$insertIndex] -match '^\s*(;.*)?$') {
            $insertIndex++
        }
        $lines.Insert($insertIndex, $definition)
    }

    Set-Content -Path $Path -Value $lines -Encoding ASCII
}

function Update-InfSectionText {
    param(
        [string]$Path,
        [string]$Section,
        [string]$OldValue,
        [string]$NewValue
    )

    $text = Get-Content -Raw -Encoding UTF8 $Path
    $escapedSection = [regex]::Escape($Section)
    $pattern = "(?ms)^(\[$escapedSection\]\r?\n)(.*?)(?=^\[|\z)"
    $match = [regex]::Match($text, $pattern)
    if (!$match.Success) {
        throw "Missing INF section [$Section] in $Path"
    }

    $body = $match.Groups[2].Value
    if ($body.IndexOf($OldValue, [System.StringComparison]::Ordinal) -lt 0) {
        if ($body.IndexOf($NewValue, [System.StringComparison]::Ordinal) -ge 0) {
            return
        }
        throw "Missing expected line in INF section [$Section]: $OldValue"
    }

    $updatedBody = $body.Replace($OldValue, $NewValue)
    $updatedText = $text.Substring(0, $match.Groups[2].Index) + $updatedBody + $text.Substring($match.Groups[2].Index + $match.Groups[2].Length)
    Set-Content -Path $Path -Value $updatedText -Encoding ASCII
}

function Add-InfSectionLine {
    param(
        [string]$Path,
        [string]$Section,
        [string]$Line
    )

    $text = Get-Content -Raw -Encoding UTF8 $Path
    $escapedSection = [regex]::Escape($Section)
    $pattern = "(?ms)^(\[$escapedSection\]\r?\n)(.*?)(?=^\[|\z)"
    $match = [regex]::Match($text, $pattern)
    if (!$match.Success) {
        throw "Missing INF section [$Section] in $Path"
    }

    $body = $match.Groups[2].Value
    if ($body.IndexOf($Line, [System.StringComparison]::Ordinal) -ge 0) {
        return
    }

    $lineEnding = if ($body.Contains("`r`n")) { "`r`n" } else { "`n" }
    $updatedBody = $body.TrimEnd("`r", "`n") + $lineEnding + $Line + $lineEnding
    $updatedText = $text.Substring(0, $match.Groups[2].Index) + $updatedBody + $text.Substring($match.Groups[2].Index + $match.Groups[2].Length)
    Set-Content -Path $Path -Value $updatedText -Encoding ASCII
}

function Set-InfSectionLineRegex {
    param(
        [string]$Path,
        [string]$Section,
        [string]$Pattern,
        [string]$Replacement
    )

    $text = Get-Content -Raw -Encoding UTF8 $Path
    $escapedSection = [regex]::Escape($Section)
    $sectionPattern = "(?ms)^(\[$escapedSection\]\r?\n)(.*?)(?=^\[|\z)"
    $match = [regex]::Match($text, $sectionPattern)
    if (!$match.Success) {
        throw "Missing INF section [$Section] in $Path"
    }

    $body = $match.Groups[2].Value
    $updatedBody = [regex]::Replace($body, $Pattern, $Replacement, [System.Text.RegularExpressions.RegexOptions]::Multiline)
    if ($updatedBody -eq $body) {
        throw "Pattern did not match INF section [$Section]: $Pattern"
    }

    $updatedText = $text.Substring(0, $match.Groups[2].Index) + $updatedBody + $text.Substring($match.Groups[2].Index + $match.Groups[2].Length)
    Set-Content -Path $Path -Value $updatedText -Encoding ASCII
}

function Remove-InfLinesRegex {
    param(
        [string]$Path,
        [string]$Pattern
    )

    $lines = [System.Collections.Generic.List[string]]::new()
    [string[]](Get-Content -Path $Path -Encoding UTF8) | ForEach-Object {
        if ($_ -notmatch $Pattern) {
            [void]$lines.Add($_)
        }
    }
    Set-Content -Path $Path -Value $lines -Encoding ASCII
}

function Set-SyncTranslateV2FormatContract {
    param(
        [string]$SysvadDir
    )

    $speakerWaveTable = Join-Path $SysvadDir "EndpointsCommon/speakerwavtable.h"
    if (Test-Path $speakerWaveTable) {
        Update-TextFile -Path $speakerWaveTable -Replacements @(
            ,@('#define SPEAKER_HOST_MIN_SAMPLE_RATE                24000', '#define SPEAKER_HOST_MIN_SAMPLE_RATE                48000')
            ,@('#define SPEAKER_HOST_MAX_SAMPLE_RATE                96000', '#define SPEAKER_HOST_MAX_SAMPLE_RATE                48000')
            ,@('#define SPEAKER_OFFLOAD_MIN_SAMPLE_RATE             44100', '#define SPEAKER_OFFLOAD_MIN_SAMPLE_RATE             48000')
            ,@('#define SPEAKER_LOOPBACK_MIN_SAMPLE_RATE            SPEAKER_HOST_MIN_SAMPLE_RATE', '#define SPEAKER_LOOPBACK_MIN_SAMPLE_RATE            48000')
            ,@('#define SPEAKER_LOOPBACK_MAX_SAMPLE_RATE            SPEAKER_HOST_MAX_SAMPLE_RATE', '#define SPEAKER_LOOPBACK_MAX_SAMPLE_RATE            48000')
        )
        Set-SyncTranslateSpeaker48kStereoOnly -Path $speakerWaveTable
    }

    $micWaveTable = Join-Path $SysvadDir "EndpointsCommon/micarraywavtable.h"
    if (Test-Path $micWaveTable) {
        Update-TextFile -Path $micWaveTable -Replacements @(
            ,@('#define MICARRAY_RAW_CHANNELS                   2', '#define MICARRAY_RAW_CHANNELS                   2')
            ,@('#define MICARRAY_PROCESSED_CHANNELS             1', '#define MICARRAY_PROCESSED_CHANNELS             2')
            ,@('#define MICARRAY_DEVICE_MAX_CHANNELS            2', '#define MICARRAY_DEVICE_MAX_CHANNELS            2')
            ,@('#define MICARRAY_32_BITS_PER_SAMPLE_PCM         32', '#define MICARRAY_32_BITS_PER_SAMPLE_PCM         16')
            ,@('#define MICARRAY_PROCESSED_MIN_SAMPLE_RATE      8000', '#define MICARRAY_PROCESSED_MIN_SAMPLE_RATE      48000')
            ,@('#define MICARRAY_PROCESSED_MAX_SAMPLE_RATE      48000', '#define MICARRAY_PROCESSED_MAX_SAMPLE_RATE      48000')
        )
        Set-SyncTranslateMicArray48kStereoOnly -Path $micWaveTable
    }

    Write-Host "[sysvad-overlay] enforced SyncTranslate v2 endpoint format macros: 48000Hz PCM16 2ch"
}

function New-SyncTranslateSpeakerFormatTable {
    param(
        [string]$Name,
        [string]$Comment
    )

@"
static
KSDATAFORMAT_WAVEFORMATEXTENSIBLE $Name[] =
{
    // SyncTranslate v2 virtual speaker boundary: 48 KHz 16-bit stereo PCM.
    {
        {
            sizeof(KSDATAFORMAT_WAVEFORMATEXTENSIBLE),
            0,
            0,
            0,
            STATICGUIDOF(KSDATAFORMAT_TYPE_AUDIO),
            STATICGUIDOF(KSDATAFORMAT_SUBTYPE_PCM),
            STATICGUIDOF(KSDATAFORMAT_SPECIFIER_WAVEFORMATEX)
        },
        {
            {
                WAVE_FORMAT_EXTENSIBLE,
                2,
                48000,
                192000,
                4,
                16,
                sizeof(WAVEFORMATEXTENSIBLE) - sizeof(WAVEFORMATEX)
            },
            16,
            KSAUDIO_SPEAKER_STEREO,
            STATICGUIDOF(KSDATAFORMAT_SUBTYPE_PCM)
        }
    },
};
"@
}

function Set-SyncTranslateSpeaker48kStereoOnly {
    param(
        [string]$Path
    )

    $text = Get-Content -Raw -Encoding UTF8 $Path
    $options = [System.Text.RegularExpressions.RegexOptions]::Singleline -bor [System.Text.RegularExpressions.RegexOptions]::Multiline
    $audioEngineFormats = New-SyncTranslateSpeakerFormatTable -Name "SpeakerAudioEngineSupportedDeviceFormats" -Comment "audio engine"
    $hostFormats = New-SyncTranslateSpeakerFormatTable -Name "SpeakerHostPinSupportedDeviceFormats" -Comment "host"
    $offloadFormats = New-SyncTranslateSpeakerFormatTable -Name "SpeakerOffloadPinSupportedDeviceFormats" -Comment "offload"
    $hostModes = @'
static
MODE_AND_DEFAULT_FORMAT SpeakerHostPinSupportedDeviceModes[] =
{
    {
        STATIC_AUDIO_SIGNALPROCESSINGMODE_RAW,
        &SpeakerHostPinSupportedDeviceFormats[0].DataFormat
    },
    {
        STATIC_AUDIO_SIGNALPROCESSINGMODE_DEFAULT,
        &SpeakerHostPinSupportedDeviceFormats[0].DataFormat
    },
    {
        STATIC_AUDIO_SIGNALPROCESSINGMODE_MEDIA,
        &SpeakerHostPinSupportedDeviceFormats[0].DataFormat
    },
    {
        STATIC_AUDIO_SIGNALPROCESSINGMODE_MOVIE,
        &SpeakerHostPinSupportedDeviceFormats[0].DataFormat
    },
    {
        STATIC_AUDIO_SIGNALPROCESSINGMODE_COMMUNICATIONS,
        &SpeakerHostPinSupportedDeviceFormats[0].DataFormat
    },
    {
        STATIC_AUDIO_SIGNALPROCESSINGMODE_NOTIFICATION,
        &SpeakerHostPinSupportedDeviceFormats[0].DataFormat
    }
};
'@
    $offloadModes = @'
static
MODE_AND_DEFAULT_FORMAT SpeakerOffloadPinSupportedDeviceModes[] =
{
    {
        STATIC_AUDIO_SIGNALPROCESSINGMODE_DEFAULT,
        &SpeakerOffloadPinSupportedDeviceFormats[0].DataFormat
    },
    {
        STATIC_AUDIO_SIGNALPROCESSINGMODE_MEDIA,
        &SpeakerOffloadPinSupportedDeviceFormats[0].DataFormat
    }
};
'@

    $updated = [regex]::Replace($text, 'static\s+KSDATAFORMAT_WAVEFORMATEXTENSIBLE\s+SpeakerAudioEngineSupportedDeviceFormats\[\]\s*=\s*\{.*?\n\};', $audioEngineFormats, $options)
    $updated = [regex]::Replace($updated, 'static\s+KSDATAFORMAT_WAVEFORMATEXTENSIBLE\s+SpeakerHostPinSupportedDeviceFormats\[\]\s*=\s*\{.*?\n\};', $hostFormats, $options)
    $updated = [regex]::Replace($updated, 'static\s+KSDATAFORMAT_WAVEFORMATEXTENSIBLE\s+SpeakerOffloadPinSupportedDeviceFormats\[\]\s*=\s*\{.*?\n\};', $offloadFormats, $options)
    $updated = [regex]::Replace($updated, 'static\s+MODE_AND_DEFAULT_FORMAT\s+SpeakerHostPinSupportedDeviceModes\[\]\s*=\s*\{.*?\n\};', $hostModes, $options)
    $updated = [regex]::Replace($updated, 'static\s+MODE_AND_DEFAULT_FORMAT\s+SpeakerOffloadPinSupportedDeviceModes\[\]\s*=\s*\{.*?\n\};', $offloadModes, $options)

    if ($updated -eq $text) {
        Write-Host "[sysvad-overlay] speaker supported/default formats already limited to 48000Hz PCM16 2ch"
        return
    }
    Set-Content -Path $Path -Value $updated -Encoding ASCII
    Write-Host "[sysvad-overlay] limited speaker supported/default formats to 48000Hz PCM16 2ch"
}

function Set-SyncTranslateMicArray48kStereoOnly {
    param(
        [string]$Path
    )

    $text = Get-Content -Raw -Encoding UTF8 $Path
    $options = [System.Text.RegularExpressions.RegexOptions]::Singleline -bor [System.Text.RegularExpressions.RegexOptions]::Multiline
    $deviceFormats = @'
static
KSDATAFORMAT_WAVEFORMATEXTENSIBLE MicArrayPinSupportedDeviceFormats[] =
{
    // SyncTranslate v2 virtual microphone boundary: 48 KHz 16-bit stereo PCM.
    {
        {
            sizeof(KSDATAFORMAT_WAVEFORMATEXTENSIBLE),
            0,
            0,
            0,
            STATICGUIDOF(KSDATAFORMAT_TYPE_AUDIO),
            STATICGUIDOF(KSDATAFORMAT_SUBTYPE_PCM),
            STATICGUIDOF(KSDATAFORMAT_SPECIFIER_WAVEFORMATEX)
        },
        {
            {
                WAVE_FORMAT_EXTENSIBLE,
                2,
                48000,
                192000,
                4,
                16,
                sizeof(WAVEFORMATEXTENSIBLE) - sizeof(WAVEFORMATEX)
            },
            16,
            KSAUDIO_SPEAKER_STEREO,
            STATICGUIDOF(KSDATAFORMAT_SUBTYPE_PCM)
        }
    },
};
'@
    $deviceModes = @'
static
MODE_AND_DEFAULT_FORMAT MicArrayPinSupportedDeviceModes[] =
{
    {
        STATIC_AUDIO_SIGNALPROCESSINGMODE_RAW,
        &MicArrayPinSupportedDeviceFormats[0].DataFormat
    },
    {
        STATIC_AUDIO_SIGNALPROCESSINGMODE_DEFAULT,
        &MicArrayPinSupportedDeviceFormats[0].DataFormat
    },
    {
        STATIC_AUDIO_SIGNALPROCESSINGMODE_SPEECH,
        &MicArrayPinSupportedDeviceFormats[0].DataFormat
    },
    {
        STATIC_AUDIO_SIGNALPROCESSINGMODE_COMMUNICATIONS,
        &MicArrayPinSupportedDeviceFormats[0].DataFormat
    }
};
'@
    $keywordFormats = @'
static
KSDATAFORMAT_WAVEFORMATEXTENSIBLE KeywordPinSupportedDeviceFormats[] =
{
    // SyncTranslate v2 keeps keyword/capture pins on the same 48 KHz 16-bit stereo PCM boundary.
    {
        {
            sizeof(KSDATAFORMAT_WAVEFORMATEXTENSIBLE),
            0,
            0,
            0,
            STATICGUIDOF(KSDATAFORMAT_TYPE_AUDIO),
            STATICGUIDOF(KSDATAFORMAT_SUBTYPE_PCM),
            STATICGUIDOF(KSDATAFORMAT_SPECIFIER_WAVEFORMATEX)
        },
        {
            {
                WAVE_FORMAT_EXTENSIBLE,
                2,
                48000,
                192000,
                4,
                16,
                sizeof(WAVEFORMATEXTENSIBLE) - sizeof(WAVEFORMATEX)
            },
            16,
            KSAUDIO_SPEAKER_STEREO,
            STATICGUIDOF(KSDATAFORMAT_SUBTYPE_PCM)
        }
    },
};
'@
    $processedRanges = @'
static
KSDATARANGE_AUDIO MicArrayPinDataRangesProcessedStream[] =
{
    {
        {
            sizeof(KSDATARANGE_AUDIO),
            KSDATARANGE_ATTRIBUTES,
            0,
            0,
            STATICGUIDOF(KSDATAFORMAT_TYPE_AUDIO),
            STATICGUIDOF(KSDATAFORMAT_SUBTYPE_PCM),
            STATICGUIDOF(KSDATAFORMAT_SPECIFIER_WAVEFORMATEX)
        },
        MICARRAY_PROCESSED_CHANNELS,
        MICARRAY_16_BITS_PER_SAMPLE_PCM,
        MICARRAY_16_BITS_PER_SAMPLE_PCM,
        MICARRAY_PROCESSED_MIN_SAMPLE_RATE,
        MICARRAY_PROCESSED_MAX_SAMPLE_RATE
    },
};
'@
    $processedPointers = @'
static
PKSDATARANGE MicArrayPinDataRangePointersStream[] =
{
    PKSDATARANGE(&MicArrayPinDataRangesProcessedStream[0]),
    PKSDATARANGE(&PinDataRangeAttributeList),
    PKSDATARANGE(&MicArrayPinDataRangesRawStream[0]),
    PKSDATARANGE(&PinDataRangeAttributeList),
};
'@
    $keywordRanges = @'
static
KSDATARANGE_AUDIO KeywordPinDataRangesStream[] =
{
    {
        {
            sizeof(KSDATARANGE_AUDIO),
            KSDATARANGE_ATTRIBUTES,
            0,
            0,
            STATICGUIDOF(KSDATAFORMAT_TYPE_AUDIO),
            STATICGUIDOF(KSDATAFORMAT_SUBTYPE_PCM),
            STATICGUIDOF(KSDATAFORMAT_SPECIFIER_WAVEFORMATEX)
        },
        2,
        16,
        16,
        48000,
        48000
    },
};
'@

    $updated = [regex]::Replace($text, 'static\s+KSDATAFORMAT_WAVEFORMATEXTENSIBLE\s+MicArrayPinSupportedDeviceFormats\[\]\s*=\s*\{.*?\n\};', $deviceFormats, $options)
    $updated = [regex]::Replace($updated, 'static\s+MODE_AND_DEFAULT_FORMAT\s+MicArrayPinSupportedDeviceModes\[\]\s*=\s*\{.*?\n\};', $deviceModes, $options)
    $updated = [regex]::Replace($updated, 'static\s+KSDATAFORMAT_WAVEFORMATEXTENSIBLE\s+KeywordPinSupportedDeviceFormats\[\]\s*=\s*\{.*?\n\};', $keywordFormats, $options)
    $updated = [regex]::Replace($updated, 'static\s+KSDATARANGE_AUDIO\s+MicArrayPinDataRangesProcessedStream\[\]\s*=\s*\{.*?\n\};', $processedRanges, $options)
    $updated = [regex]::Replace($updated, 'C_ASSERT\(SIZEOF_ARRAY\(MicArrayPinDataRangesProcessedStream\)\s*==\s*8\);', 'C_ASSERT(SIZEOF_ARRAY(MicArrayPinDataRangesProcessedStream) == 1);')
    $updated = [regex]::Replace($updated, 'static\s+PKSDATARANGE\s+MicArrayPinDataRangePointersStream\[\]\s*=\s*\{.*?\n\};', $processedPointers, $options)
    $updated = [regex]::Replace($updated, 'static\s+KSDATARANGE_AUDIO\s+KeywordPinDataRangesStream\[\]\s*=\s*\{.*?\n\};', $keywordRanges, $options)
    if ($updated -eq $text) {
        Write-Host "[sysvad-overlay] mic array supported/default formats already limited to 48000Hz PCM16 2ch"
        return
    }
    Set-Content -Path $Path -Value $updated -Encoding ASCII
    Write-Host "[sysvad-overlay] limited mic array supported/default formats to 48000Hz PCM16 2ch"
}

function Set-SyncTranslateEndpointTables {
    param(
        [string]$SysvadDir
    )

    $minipairsHeader = Join-Path $SysvadDir "TabletAudioSample/minipairs.h"
    if (!(Test-Path $minipairsHeader)) {
        return
    }

    $text = Get-Content -Raw -Encoding UTF8 $minipairsHeader
    $options = [System.Text.RegularExpressions.RegexOptions]::Singleline -bor [System.Text.RegularExpressions.RegexOptions]::Multiline
    $renderEndpoints = @'
static
PENDPOINT_MINIPAIR  g_RenderEndpoints[] =
{
    &SpeakerMiniports,
};
'@
    $captureEndpoints = @'
static
PENDPOINT_MINIPAIR  g_CaptureEndpoints[] =
{
    &MicArray1Miniports,
};
'@
    $updated = [regex]::Replace(
        $text,
        'static\s+PENDPOINT_MINIPAIR\s+g_RenderEndpoints\[\]\s*=\s*\{.*?\};',
        $renderEndpoints,
        $options
    )
    $updated = [regex]::Replace(
        $updated,
        'static\s+PENDPOINT_MINIPAIR\s+g_CaptureEndpoints\[\]\s*=\s*\{.*?\};',
        $captureEndpoints,
        $options
    )
    if ($updated -ne $text) {
        Set-Content -Path $minipairsHeader -Value $updated -Encoding ASCII
    }

    Write-Host "[sysvad-overlay] limited endpoint tables to SyncTranslate virtual speaker + virtual microphone"
}

$commonReplacements = @(
    ,@("Root\sysvad_ComponentizedAudioSample", $HardwareId)
    ,@("Root\Sysvad_ComponentizedAudioSample", $HardwareId)
    ,@('ProviderName = "TODO-Set-Provider"', 'ProviderName = "SyncTranslate"')
    ,@('MfgName      = "TODO-Set-Manufacturer"', 'MfgName      = "SyncTranslate"')
    ,@('MsCopyRight  = "TODO-Set-Copyright"', 'MsCopyRight  = "Copyright (c) SyncTranslate"')
    ,@('SYSVAD_SA.DeviceDesc="Virtual Audio Device (WDM) - Tablet Sample"', 'SYSVAD_SA.DeviceDesc="SyncTranslate Virtual Audio Device"')
    ,@('SYSVAD_ComponentizedAudioSample.SvcDesc="Virtual Audio Device (WDM) - Tablet Sample Driver"', 'SYSVAD_ComponentizedAudioSample.SvcDesc="SyncTranslate Virtual Audio Driver"')
    ,@('SYSVAD.WaveSpeaker.szPname="SYSVAD Wave Speaker"', 'SYSVAD.WaveSpeaker.szPname="SyncTranslate Virtual Speaker"')
    ,@('SYSVAD.TopologySpeaker.szPname="SYSVAD Topology Speaker"', 'SYSVAD.TopologySpeaker.szPname="SyncTranslate Virtual Speaker Topology"')
    ,@('SYSVAD.WaveMicArray1.szPname="SYSVAD Wave Microphone Array - Front"', 'SYSVAD.WaveMicArray1.szPname="SyncTranslate Virtual Microphone"')
    ,@('SYSVAD.TopologyMicArray1.szPname="SYSVAD Topology Microphone Array - Front"', 'SYSVAD.TopologyMicArray1.szPname="SyncTranslate Virtual Microphone Topology"')
    ,@('MicArray1CustomName= "Internal Microphone Array - Front"', 'MicArray1CustomName= "SyncTranslate Virtual Microphone"')
)

# Win11 LTSC 2024 reports build 26100. Keep INF decorations aligned so PnP can match on that OS.
$osDecorationReplacements = @(
    ,@('10.0...22621', '10.0...26100')
)

Update-TextFile -Path $componentInf -Replacements $commonReplacements
Update-TextFile -Path $componentInf -Replacements $osDecorationReplacements
Update-TextFile -Path $componentInf -Replacements @(
    ,@('HKCR,CLSID\', 'HKR,Classes\CLSID\')
)
Set-InfSectionLineRegex -Path $componentInf -Section "SYSVAD_SA.NT" -Pattern '^Needs\s*=.*$' -Replacement 'Needs=KS.Registration, WDMAUDIO.Registration'
Set-InfStringDefinition -Path $componentInf -Name "KSNODETYPE_ANY" -Value "{00000000-0000-0000-0000-000000000000}"
Set-InfStringDefinition -Path $componentInf -Name "KSNODETYPE_SPEAKER" -Value "{DFF21CE1-F70F-11D0-B917-00A0C9223196}"
Set-InfStringDefinition -Path $componentInf -Name "KSNODETYPE_MICROPHONE_ARRAY" -Value "{DFF21BE5-F70F-11D0-B917-00A0C9223196}"
Set-InfStringDefinition -Path $componentInf -Name "PKEY_AudioEngine_DeviceFormat" -Value "{F19F064D-082C-4E27-BC73-6882A1BB8E4C},0"
$syncTranslateEndpointFormat48kPcm16Stereo = "HKR,EP\0,%PKEY_AudioEngine_DeviceFormat%,0x00000001,fe,ff,02,00,80,bb,00,00,00,ee,02,00,04,00,10,00,16,00,10,00,03,00,00,00,01,00,00,00,00,00,10,00,80,00,00,aa,00,38,9b,71"
Add-InfSectionLine -Path $componentInf -Section "SYSVAD.I.WaveSpeaker.AddReg" -Line $syncTranslateEndpointFormat48kPcm16Stereo
Add-InfSectionLine -Path $componentInf -Section "SYSVAD.I.WaveMicArray1.AddReg" -Line $syncTranslateEndpointFormat48kPcm16Stereo
# Keep the stock SysVAD [SYSVAD_SA.NT.Interfaces] AddInterface section intact.
# Hyper-V validation showed these component interfaces are required for Windows
# to create MMDevice AudioEndpoint entries after component INF install + reboot.
if (Test-Path $extensionInf) {
    Update-TextFile -Path $extensionInf -Replacements @(
        ,@("Root\sysvad_ComponentizedAudioSample", $HardwareId)
        ,@("Root\Sysvad_ComponentizedAudioSample", $HardwareId)
    )
    Set-InfStringDefinition -Path $extensionInf -Name "ProviderName" -Value "SyncTranslate"
    Set-InfStringDefinition -Path $extensionInf -Name "MfgName" -Value "SyncTranslate"
    Set-InfStringDefinition -Path $extensionInf -Name "Device.ExtensionDesc" -Value "SyncTranslate Virtual Audio Extension"
    Set-InfStringDefinition -Path $extensionInf -Name "ExtendedFriendlyName" -Value "SyncTranslate Virtual Audio Device"
    Update-TextFile -Path $extensionInf -Replacements $osDecorationReplacements
}

$apoInf = Join-Path $SysvadDir "TabletAudioSample/ComponentizedApoSample.inx"
if (Test-Path $apoInf) {
    Set-InfStringDefinition -Path $apoInf -Name "ProviderName" -Value "SyncTranslate"
    Set-InfStringDefinition -Path $apoInf -Name "MfgName" -Value "SyncTranslate"
    Set-InfStringDefinition -Path $apoInf -Name "Apo.ComponentDesc" -Value "SyncTranslate Virtual Audio APO Component"
    Update-TextFile -Path $apoInf -Replacements $osDecorationReplacements
}

Set-SyncTranslateV2FormatContract -SysvadDir $SysvadDir
Set-SyncTranslateEndpointTables -SysvadDir $SysvadDir

$overlayRoot = Join-Path (Split-Path -Parent $scriptRoot) "overlay"
if (Test-Path $overlayRoot) {
    Get-ChildItem -Path $overlayRoot -Recurse -File | ForEach-Object {
        $relativePath = $_.FullName.Substring($overlayRoot.Length).TrimStart('\', '/')
        $targetPath = Join-Path $SysvadDir $relativePath
        $targetDir = Split-Path -Parent $targetPath
        if (!(Test-Path $targetDir)) {
            New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
        }
        Copy-Item -Path $_.FullName -Destination $targetPath -Force
        Write-Host "[sysvad-overlay] applied override: $relativePath"
    }
}

$endpointsProject = Join-Path $SysvadDir "EndpointsCommon/EndpointsCommon.vcxproj"
if (Test-Path $endpointsProject) {
    $projectText = Get-Content -Raw -Encoding UTF8 $endpointsProject
    $overlayEndpoints = Join-Path $overlayRoot "EndpointsCommon"
    if (Test-Path $overlayEndpoints) {
        Get-ChildItem -Path $overlayEndpoints -Filter "*.cpp" -File | ForEach-Object {
            $compileItem = '<ClCompile Include="' + $_.Name + '" />'
            if ($projectText -notlike "*$compileItem*") {
                $projectText = $projectText.Replace(
                    '    <ClCompile Include="NewDelete.cpp" />',
                    "    $compileItem`r`n    <ClCompile Include=`"NewDelete.cpp`" />"
                )
            }
        }
        Set-Content -Path $endpointsProject -Value $projectText -Encoding ASCII
    }
}

$tabletProject = Join-Path $SysvadDir "TabletAudioSample/TabletAudioSample.vcxproj"
if (Test-Path $tabletProject) {
    $projectText = Get-Content -Raw -Encoding UTF8 $tabletProject
    $overlayCppFiles = Get-ChildItem -Path $overlayRoot -Filter "*.cpp" -File -ErrorAction SilentlyContinue
    foreach ($overlayCpp in $overlayCppFiles) {
        $compileItem = '<ClCompile Include="..\' + $overlayCpp.Name + '" />'
        if ($projectText -notlike "*$compileItem*") {
            $projectText = $projectText.Replace(
                '    <ClCompile Include="..\adapter.cpp" />',
                "    <ClCompile Include=`"..\adapter.cpp`" />`r`n    $compileItem"
            )
        }
    }
    Set-Content -Path $tabletProject -Value $projectText -Encoding ASCII
}

$adapterSource = Join-Path $SysvadDir "adapter.cpp"
if (Test-Path $adapterSource) {
    $adapterText = Get-Content -Raw -Encoding UTF8 $adapterSource
    if ($adapterText -notlike '*#include "synctranslate_control.h"*') {
        $adapterText = $adapterText.Replace(
            '#include "IHVPrivatePropertySet.h"',
            "#include `"IHVPrivatePropertySet.h`"`r`n#include `"synctranslate_control.h`""
        )
    }
    if ($adapterText -notlike '*SyncTranslateControlShutdown(DriverObject);*') {
        $adapterText = $adapterText.Replace(
            '    ReleaseRegistryStringBuffer();',
            "    SyncTranslateControlShutdown(DriverObject);`r`n`r`n    ReleaseRegistryStringBuffer();"
        )
    }
    if ($adapterText -notlike '*SyncTranslateControlInitialize(DriverObject);*') {
        $adapterText = $adapterText.Replace(
            '    DriverObject->MajorFunction[IRP_MJ_PNP] = PnpHandler;',
            "    DriverObject->MajorFunction[IRP_MJ_PNP] = PnpHandler;`r`n`r`n    ntStatus = SyncTranslateControlInitialize(DriverObject);`r`n    if (!NT_SUCCESS(ntStatus))`r`n    {`r`n        DPF(D_ERROR, (`"SyncTranslateControlInitialize failed, 0x%x`", ntStatus));`r`n    }"
        )
    }
    $adapterText = $adapterText.Replace(
        "        SyncTranslateControlShutdown(DriverObject);`r`n`r`n    ReleaseRegistryStringBuffer();",
        "        ReleaseRegistryStringBuffer();"
    )
    Set-Content -Path $adapterSource -Value $adapterText -Encoding ASCII
}

Write-Host "[sysvad-overlay] applied SyncTranslate names and hardware id: $HardwareId"
