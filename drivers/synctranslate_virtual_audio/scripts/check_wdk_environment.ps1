param(
    [switch]$Json
)

$ErrorActionPreference = "Continue"
# Status vocabulary: PASS / WARN / FAIL. Missing build tools are WARN here so
# meeting-mode validation can continue; build scripts promote missing msbuild to FAIL.

function Find-Tool {
    param([string]$Name, [string[]]$Candidates = @())
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($candidate in $Candidates) {
        if ($candidate -and (Test-Path $candidate)) {
            return (Resolve-Path $candidate).Path
        }
    }
    return ""
}

function Find-FirstGlob {
    param([string[]]$Patterns)
    foreach ($pattern in $Patterns) {
        $match = Get-ChildItem -Path $pattern -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($match) { return $match.FullName }
    }
    return ""
}

$kitsRoot = "${env:ProgramFiles(x86)}\Windows Kits\10"
$vsRoots = @("${env:ProgramFiles}\Microsoft Visual Studio\2022", "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022")
$kitBinDirs = @(Get-ChildItem -Path (Join-Path $kitsRoot "bin") -Directory -ErrorAction SilentlyContinue | ForEach-Object { Join-Path $_.FullName "x64" })
$kitToolDirs = @(Get-ChildItem -Path (Join-Path $kitsRoot "Tools") -Directory -ErrorAction SilentlyContinue | ForEach-Object { Join-Path $_.FullName "x64" })
$kitSearchDirs = @($kitBinDirs + $kitToolDirs)

$msbuildCandidate = Find-FirstGlob @(
    "${env:ProgramFiles}\Microsoft Visual Studio\2022\*\MSBuild\Current\Bin\MSBuild.exe",
    "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\*\MSBuild\Current\Bin\MSBuild.exe"
)

$tools = [ordered]@{
    msbuild = Find-Tool "msbuild.exe" @($msbuildCandidate)
    infverif = Find-Tool "infverif.exe" @($kitSearchDirs | ForEach-Object { Join-Path $_ "infverif.exe" })
    stampinf = Find-Tool "stampinf.exe" @($kitSearchDirs | ForEach-Object { Join-Path $_ "stampinf.exe" })
    signtool = Find-Tool "signtool.exe" @($kitSearchDirs | ForEach-Object { Join-Path $_ "signtool.exe" })
}

$missing = @($tools.Keys | Where-Object { -not $tools[$_] })
$status = if ($missing.Count -eq 0) { "PASS" } else { "WARN" }
$result = [ordered]@{
    status = $status
    tools = $tools
    missing_tools = $missing
    windows_sdk_wdk_root = $kitsRoot
    windows_sdk_wdk_root_exists = Test-Path $kitsRoot
    visual_studio_roots = $vsRoots
    suggested_fix = "Install Visual Studio 2022 Build Tools, Windows SDK, and Windows Driver Kit; then reopen Developer PowerShell."
}

if ($Json) {
    $result | ConvertTo-Json -Depth 5
    exit 0
}

Write-Host "$status WDK environment"
foreach ($name in $tools.Keys) {
    $value = if ($tools[$name]) { $tools[$name] } else { "missing" }
    $lineStatus = if ($tools[$name]) { "PASS" } else { "WARN" }
    Write-Host "[${lineStatus}] ${name}: ${value}"
}
if ($missing.Count -gt 0) {
    Write-Host "[WARN] suggested_fix: $($result.suggested_fix)"
}
exit 0
