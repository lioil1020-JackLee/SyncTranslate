param(
    [string]$SamplesRoot = "downloads/driver_samples/Windows-driver-samples",
    [string]$Configuration = "Release",
    [string]$Platform = "x64",
    [string]$ArtifactsDir = "artifacts/driver/synctranslate_virtual_audio",
    [switch]$EnableSpectreMitigation
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

$wdk = & (Join-Path $scriptRoot "check_wdk_environment.ps1") -Json | ConvertFrom-Json
if (-not $wdk.tools.msbuild) {
    Write-Host "FAIL msbuild.exe was not found. Install Visual Studio 2022 Build Tools with WDK components."
    Write-Host "Suggested: drivers/synctranslate_virtual_audio/scripts/install_build_prereqs.ps1 -PrintWdkInstructions"
    exit 1
}
if (-not $wdk.tools.infverif -or -not $wdk.tools.stampinf -or -not $wdk.tools.signtool) {
    Write-Host "WARN WDK tools are incomplete. Build may work, but package validation/signing will be incomplete."
    Write-Host "Missing: $($wdk.missing_tools -join ', ')"
}

function Find-WdkTool {
    param([string]$Name)
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $kitsRoot = "${env:ProgramFiles(x86)}\Windows Kits\10"
    $match = Get-ChildItem -Path $kitsRoot -Filter $Name -Recurse -File -ErrorAction SilentlyContinue |
        Sort-Object @{ Expression = { if ($_.FullName -match "\\x64\\") { 0 } elseif ($_.FullName -match "\\x86\\") { 1 } else { 2 } } }, FullName |
        Select-Object -First 1
    if ($match) { return $match.FullName }
    return ""
}

function Test-InfHasTextContent {
    param([string]$Path)
    if (!(Test-Path $Path)) { return $false }
    $bytes = [System.IO.File]::ReadAllBytes($Path)
    if ($bytes.Length -eq 0) { return $false }
    $nonZero = @($bytes | Where-Object { $_ -ne 0 }).Count
    if ($nonZero -eq 0) { return $false }
    $text = [System.Text.Encoding]::ASCII.GetString($bytes)
    return ($text -match "\[Version\]" -and $text -match "\[Manufacturer\]")
}

function Copy-InfTemplateToPackage {
    param(
        [string]$SourcePath,
        [string]$TargetPath,
        [string]$Platform
    )
    if (!(Test-Path $SourcePath)) {
        throw "Missing source INF template for package repair: $SourcePath"
    }
    $bytes = [System.IO.File]::ReadAllBytes($SourcePath)
    if ($bytes.Length -ge 2 -and $bytes[0] -eq 0xFF -and $bytes[1] -eq 0xFE) {
        $text = [System.Text.Encoding]::Unicode.GetString($bytes)
    }
    elseif ($bytes.Length -ge 2 -and $bytes[0] -eq 0xFE -and $bytes[1] -eq 0xFF) {
        $text = [System.Text.Encoding]::BigEndianUnicode.GetString($bytes)
    }
    else {
        $probeLength = [Math]::Min($bytes.Length, 4096)
        $oddNulls = 0
        $evenNulls = 0
        for ($i = 0; $i -lt $probeLength; $i++) {
            if ($bytes[$i] -eq 0) {
                if (($i % 2) -eq 0) { $evenNulls++ } else { $oddNulls++ }
            }
        }
        if ($oddNulls -gt [Math]::Max(16, [int]($probeLength / 4)) -and $evenNulls -lt [Math]::Max(4, [int]($oddNulls / 8))) {
            $text = [System.Text.Encoding]::Unicode.GetString($bytes)
        }
        elseif ($evenNulls -gt [Math]::Max(16, [int]($probeLength / 4)) -and $oddNulls -lt [Math]::Max(4, [int]($evenNulls / 8))) {
            $text = [System.Text.Encoding]::BigEndianUnicode.GetString($bytes)
        }
        else {
            $text = [System.Text.Encoding]::UTF8.GetString($bytes)
        }
    }
    $text = $text.TrimStart([char]0xFEFF).Replace([string][char]0, "")
    if ($text -notmatch "\[Version\]" -or $text -notmatch "\[Manufacturer\]") {
        $text = (Get-Content -Raw -Path $SourcePath).TrimStart([char]0xFEFF).Replace([string][char]0, "")
    }
    $arch = if ($Platform -eq "x64") { "amd64" } else { $Platform }
    $text = $text.Replace('$ARCH$', $arch)
    $text = $text.Replace('$KMDFVERSION$', '1.15')
    $resolvedTargetPath = [System.IO.Path]::GetFullPath($TargetPath)
    [System.IO.File]::WriteAllText($resolvedTargetPath, $text, [System.Text.Encoding]::ASCII)
    if (!(Test-InfHasTextContent -Path $TargetPath)) {
        throw "Repaired INF still does not contain valid text: $TargetPath"
    }
    Write-Host "WARN repaired package INF from source template: $TargetPath"
}

function Test-CertificateFile {
    param([string]$Path)
    if (!(Test-Path $Path)) { return $false }
    try {
        [void][System.Security.Cryptography.X509Certificates.X509Certificate2]::new($Path)
        return $true
    }
    catch {
        return $false
    }
}

$sysvadDir = Join-Path $SamplesRoot "audio/sysvad"
if (!(Test-Path $sysvadDir)) {
    Write-Host "FAIL Missing SysVAD source: $sysvadDir"
    Write-Host "Run: drivers/synctranslate_virtual_audio/scripts/fetch_sysvad.ps1"
    exit 1
}

Write-Host "PASS applying SyncTranslate SysVAD overlay"
& (Join-Path $scriptRoot "apply_sysvad_overlay.ps1") -SysvadDir $sysvadDir

Write-Host "PASS invoking existing build_driver_poc.ps1"
if ($EnableSpectreMitigation) {
    & (Join-Path $scriptRoot "build_driver_poc.ps1") `
        -SamplesRoot $SamplesRoot `
        -Configuration $Configuration `
        -Platform $Platform `
        -ArtifactsDir $ArtifactsDir `
        -EnableSpectreMitigation
}
else {
    & (Join-Path $scriptRoot "build_driver_poc.ps1") `
        -SamplesRoot $SamplesRoot `
        -Configuration $Configuration `
        -Platform $Platform `
        -ArtifactsDir $ArtifactsDir
}
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAIL driver package build failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

$packageDir = Join-Path $ArtifactsDir "package"

$infRepairMap = @(
    @{
        source = Join-Path $sysvadDir "TabletAudioSample/ComponentizedAudioSample.inx"
        target = Join-Path $packageDir "ComponentizedAudioSample.inf"
    },
    @{
        source = Join-Path $sysvadDir "TabletAudioSample/ComponentizedAudioSampleExtension.inx"
        target = Join-Path $packageDir "ComponentizedAudioSampleExtension.inf"
    },
    @{
        source = Join-Path $sysvadDir "TabletAudioSample/ComponentizedApoSample.inx"
        target = Join-Path $packageDir "ComponentizedApoSample.inf"
    }
)
foreach ($item in $infRepairMap) {
    if (!(Test-InfHasTextContent -Path $item.target)) {
        throw "WDK-generated package INF is missing or invalid: $($item.target). Re-run the build in a clean WDK environment; do not ship template-repaired INFs because Windows AudioEndpoint enumeration depends on stamped package metadata."
    }
    Write-Host "PASS using WDK-generated package INF: $($item.target)"
}
foreach ($requiredMarker in @("Root\SyncTranslateVirtualAudio", "SyncTranslate Virtual Audio", "SyncTranslate Virtual Speaker", "SyncTranslate Virtual Microphone")) {
    $joinedInfText = (Get-ChildItem -Path $packageDir -Filter "*.inf" -File | ForEach-Object { Get-Content -Raw -Path $_.FullName }) -join "`n"
    if ($joinedInfText -notlike "*$requiredMarker*") {
        throw "Package INF marker missing after build: $requiredMarker"
    }
}

$sourceCert = Join-Path (Join-Path $sysvadDir (Join-Path $Platform $Configuration)) "package.cer"
$targetCert = Join-Path $ArtifactsDir "SyncTranslateVirtualAudioTest.cer"
if (Test-CertificateFile -Path $sourceCert) {
    Copy-Item -Path $sourceCert -Destination $targetCert -Force
    if (Test-CertificateFile -Path $targetCert) {
        Write-Host "PASS copied test certificate to $targetCert"
    }
    else {
        Write-Host "WARN copied package certificate is invalid after copy; exporting WDK test certificate from CurrentUser\\My instead."
    }
}
else {
    Write-Host "WARN WDK package certificate was not found or was invalid at $sourceCert."
}

if (!(Test-CertificateFile -Path $targetCert)) {
    $wdkCert = Get-ChildItem Cert:\CurrentUser\My -ErrorAction SilentlyContinue |
        Where-Object { $_.HasPrivateKey -and $_.Subject -like "*WDKTestCert*" } |
        Sort-Object NotBefore -Descending |
        Select-Object -First 1
    if ($wdkCert) {
        Export-Certificate -Cert $wdkCert -FilePath $targetCert -Force | Out-Null
        if (!(Test-CertificateFile -Path $targetCert)) {
            throw "Exported WDK test certificate is still invalid: $targetCert"
        }
        Write-Host "WARN exported WDK test certificate from CurrentUser\\My to $targetCert because package.cer was missing or invalid."
    }
    else {
        Write-Host "WARN WDK test certificate was not found or was invalid at $sourceCert. Test install may require manual certificate import."
    }
}
Write-Host "PASS driver package output: $packageDir"
Write-Host "WARN No certificate or private key is created by this script. Use test signing only in a disposable VM/lab."
exit 0
