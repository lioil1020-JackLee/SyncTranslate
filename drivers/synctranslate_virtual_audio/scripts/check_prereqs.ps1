param(
    [switch]$Json
)

$ErrorActionPreference = "Stop"

function Find-CommandPath {
    param([string]$Name)
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }
    return ""
}

function Find-FirstFile {
    param(
        [string[]]$Roots,
        [string]$Filter
    )
    foreach ($root in $Roots) {
        if (!(Test-Path $root)) {
            continue
        }
        $match = Get-ChildItem -Path $root -Filter $Filter -Recurse -File -ErrorAction SilentlyContinue |
            Sort-Object FullName |
            Select-Object -First 1
        if ($match) {
            return $match.FullName
        }
    }
    return ""
}

function Find-PlatformToolset {
    param(
        [string[]]$Roots,
        [string]$ToolsetName
    )
    foreach ($root in $Roots) {
        if (!(Test-Path $root)) {
            continue
        }
        $match = Get-ChildItem -Path $root -Directory -Filter $ToolsetName -Recurse -ErrorAction SilentlyContinue |
            Where-Object { Test-Path (Join-Path $_.FullName "Toolset.props") } |
            Sort-Object @{ Expression = { if ($_.FullName -match "\\Platforms\\x64\\") { 0 } else { 1 } } }, FullName |
            Select-Object -First 1
        if ($match) {
            return (Join-Path $match.FullName "Toolset.props")
        }
    }
    return ""
}

$windowsKitsRoots = @(
    "${env:ProgramFiles(x86)}\Windows Kits\10",
    "${env:ProgramFiles}\Windows Kits\10"
)

$visualStudioRoots = @(
    "${env:ProgramFiles}\Microsoft Visual Studio",
    "${env:ProgramFiles(x86)}\Microsoft Visual Studio"
)

$msbuild = Find-CommandPath "msbuild.exe"
if (!$msbuild) {
    $msbuild = Find-FirstFile -Roots $visualStudioRoots -Filter "MSBuild.exe"
}

$inf2cat = Find-CommandPath "inf2cat.exe"
if (!$inf2cat) {
    $inf2cat = Find-FirstFile -Roots $windowsKitsRoots -Filter "Inf2Cat.exe"
}

$signtool = Find-CommandPath "signtool.exe"
if (!$signtool) {
    $signtool = Find-FirstFile -Roots $windowsKitsRoots -Filter "signtool.exe"
}

$kernelModeDriverToolset = Find-PlatformToolset -Roots $visualStudioRoots -ToolsetName "WindowsKernelModeDriver10.0"
$applicationForDriversToolset = Find-PlatformToolset -Roots $visualStudioRoots -ToolsetName "WindowsApplicationForDrivers10.0"
if (!$kernelModeDriverToolset) {
    $kernelModeDriverToolset = Find-FirstFile -Roots $windowsKitsRoots -Filter "WDK.*.WindowsKernelModeDriver.props"
}
if (!$applicationForDriversToolset) {
    $applicationForDriversToolset = Find-FirstFile -Roots $windowsKitsRoots -Filter "WDK.*.WindowsApplicationForDrivers.props"
}

$pnputil = Find-CommandPath "pnputil.exe"
$git = Find-CommandPath "git.exe"
$wix = ""
if (Test-Path "artifacts/tools/wix5/wix.exe") {
    $wix = (Resolve-Path "artifacts/tools/wix5/wix.exe").Path
}
elseif (Test-Path "artifacts/tools/wix/wix.exe") {
    $wix = (Resolve-Path "artifacts/tools/wix/wix.exe").Path
}
else {
    $wix = Find-CommandPath "wix.exe"
}
$dotnet = Find-CommandPath "dotnet.exe"
if (Test-Path "artifacts/tools/dotnet/dotnet.exe") {
    $dotnet = (Resolve-Path "artifacts/tools/dotnet/dotnet.exe").Path
}
$dotnetSdks = @()
if ($dotnet) {
    try {
        $dotnetSdks = & $dotnet --list-sdks 2>$null
    }
    catch {
        $dotnetSdks = @()
    }
}

$testSigning = ""
try {
    $bcd = & bcdedit /enum "{current}" 2>$null
    $line = $bcd | Where-Object { $_ -match "testsigning" } | Select-Object -First 1
    if ($line) {
        $testSigning = ($line -replace "\s+", " ").Trim()
    }
}
catch {
    $testSigning = "unknown"
}

$result = [ordered]@{
    git = $git
    msbuild = $msbuild
    inf2cat = $inf2cat
    signtool = $signtool
    kernel_mode_driver_toolset = $kernelModeDriverToolset
    application_for_drivers_toolset = $applicationForDriversToolset
    pnputil = $pnputil
    wix = $wix
    dotnet = $dotnet
    dotnet_sdks = @($dotnetSdks)
    test_signing = $testSigning
    wdk_ready = [bool]($inf2cat -and $signtool -and $kernelModeDriverToolset -and $applicationForDriversToolset)
    build_ready = [bool]($msbuild -and $inf2cat -and $signtool -and $kernelModeDriverToolset -and $applicationForDriversToolset)
    msi_ready = [bool]($wix -or ($dotnet -and $dotnetSdks.Count -gt 0))
}

if ($Json) {
    $result | ConvertTo-Json -Depth 4
    exit 0
}

Write-Host "[driver-prereq] git: $($result.git)"
Write-Host "[driver-prereq] msbuild: $($result.msbuild)"
Write-Host "[driver-prereq] inf2cat: $($result.inf2cat)"
Write-Host "[driver-prereq] signtool: $($result.signtool)"
Write-Host "[driver-prereq] kernel driver toolset: $($result.kernel_mode_driver_toolset)"
Write-Host "[driver-prereq] app driver toolset: $($result.application_for_drivers_toolset)"
Write-Host "[driver-prereq] pnputil: $($result.pnputil)"
Write-Host "[driver-prereq] wix: $($result.wix)"
Write-Host "[driver-prereq] dotnet: $($result.dotnet)"
Write-Host "[driver-prereq] dotnet sdks: $($result.dotnet_sdks -join '; ')"
Write-Host "[driver-prereq] test signing: $($result.test_signing)"

if (!$result.build_ready) {
    Write-Warning "WDK/Visual Studio build tools are incomplete. Install Visual Studio C++ tools, Windows SDK, and Windows Driver Kit before building the SyncTranslate virtual audio driver."
}
if (!$result.msi_ready) {
    Write-Warning "MSI packaging tools are incomplete. Install the .NET SDK so the script can install WiX, or install wix.exe and add it to PATH."
}

exit 0
