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
Update-TextFile -Path $componentInf -Replacements @(,@('; AddInterface=', 'AddInterface='))
Update-TextFile -Path $componentInf -Replacements $osDecorationReplacements
Update-TextFile -Path $componentInf -Replacements @(
    ,@('HKCR,CLSID\', 'HKR,Classes\CLSID\')
    ,@('Needs=KS.Registration, WDMAUDIO.Registration', 'Needs=KS.Registration, WDMAUDIO.Registration, MsApoFxProxy.Registration')
)
if (Test-Path $extensionInf) {
    Update-TextFile -Path $extensionInf -Replacements @(
        ,@("Root\sysvad_ComponentizedAudioSample", $HardwareId)
        ,@("Root\Sysvad_ComponentizedAudioSample", $HardwareId)
    )
    Update-TextFile -Path $extensionInf -Replacements $osDecorationReplacements
}

$apoInf = Join-Path $SysvadDir "TabletAudioSample/ComponentizedApoSample.inf"
if (Test-Path $apoInf) {
    Update-TextFile -Path $apoInf -Replacements $osDecorationReplacements
}

$minipairsHeader = Join-Path $SysvadDir "TabletAudioSample/minipairs.h"
if (Test-Path $minipairsHeader) {
    Update-TextFileRegex -Path $minipairsHeader -Replacements @(
        ,@(
            '(?s)static\s+PENDPOINT_MINIPAIR\s+g_RenderEndpoints\[\]\s*=\s*\{.*?\};',
            @'
static
PENDPOINT_MINIPAIR  g_RenderEndpoints[] =
{
    &SpeakerMiniports,
};
'@
        ),
    ,@(
            '(?s)static\s+PENDPOINT_MINIPAIR\s+g_CaptureEndpoints\[\]\s*=\s*\{.*?\};',
            @'
static
PENDPOINT_MINIPAIR  g_CaptureEndpoints[] =
{
    &MicArray1Miniports,
};
'@
        )
    )
}

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
