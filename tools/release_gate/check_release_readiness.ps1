param(
    [string]$AppExe = "dist/SyncTranslate-onedir/SyncTranslate.exe",
    [string]$BridgeExe = "dist/SyncTranslate-onedir/runtimes/audio/sync_audio_bridge.exe",
    [string]$DriverMsi = "artifacts/driver/synctranslate_virtual_audio/SyncTranslateVirtualAudioDriver.msi",
    [string]$AppZip = "dist/SyncTranslate-onedir-windows.zip",
    [string]$ReportDir = "logs/release_gate"
)

$ErrorActionPreference = "Stop"

function Find-Tool {
    param([string]$Name)

    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    $roots = @(
        "${env:ProgramFiles(x86)}\Windows Kits\10",
        "${env:ProgramFiles}\Windows Kits\10"
    )
    foreach ($root in $roots) {
        if (!(Test-Path $root)) {
            continue
        }
        $match = Get-ChildItem -Path $root -Filter $Name -Recurse -File -ErrorAction SilentlyContinue |
            Sort-Object FullName -Descending |
            Select-Object -First 1
        if ($match) {
            return $match.FullName
        }
    }
    return ""
}

function Test-Signature {
    param(
        [string]$Signtool,
        [string]$Path
    )

    if (!(Test-Path $Path)) {
        return [ordered]@{
            path = $Path
            exists = $false
            signed = $false
            status = "missing"
            signer = ""
        }
    }

    $auth = Get-AuthenticodeSignature -FilePath $Path
    $verified = $false
    if ($Signtool) {
        & $Signtool verify /pa /q $Path | Out-Null
        $verified = ($LASTEXITCODE -eq 0)
    }

    return [ordered]@{
        path = $Path
        exists = $true
        signed = ($auth.Status -eq "Valid" -or $verified)
        status = [string]$auth.Status
        signer = if ($auth.SignerCertificate) { $auth.SignerCertificate.Subject } else { "" }
        thumbprint = if ($auth.SignerCertificate) { $auth.SignerCertificate.Thumbprint } else { "" }
        signtool_verified = $verified
    }
}

function Get-ArtifactInfo {
    param([string]$Path)

    if (!(Test-Path $Path)) {
        return [ordered]@{
            path = $Path
            exists = $false
        }
    }

    $item = Get-Item $Path
    $hash = Get-FileHash -Path $Path -Algorithm SHA256
    return [ordered]@{
        path = $Path
        exists = $true
        bytes = $item.Length
        last_write_time = $item.LastWriteTime.ToString("o")
        sha256 = $hash.Hash
    }
}

New-Item -ItemType Directory -Path $ReportDir -Force | Out-Null

$signtool = Find-Tool "signtool.exe"
$inf2cat = Find-Tool "inf2cat.exe"
$pnputil = Find-Tool "pnputil.exe"

$certs = @(Get-ChildItem Cert:\CurrentUser\My, Cert:\LocalMachine\My -CodeSigningCert -ErrorAction SilentlyContinue |
    Sort-Object NotAfter -Descending |
    ForEach-Object {
        [ordered]@{
            subject = $_.Subject
            thumbprint = $_.Thumbprint
            not_after = $_.NotAfter.ToString("o")
            has_private_key = $_.HasPrivateKey
        }
    })

$defender = $null
try {
    $mp = Get-MpComputerStatus -ErrorAction Stop
    $defender = [ordered]@{
        available = $true
        antivirus_enabled = $mp.AntivirusEnabled
        realtime_protection_enabled = $mp.RealTimeProtectionEnabled
        antispyware_signature_last_updated = $mp.AntispywareSignatureLastUpdated.ToString("o")
        antivirus_signature_last_updated = $mp.AntivirusSignatureLastUpdated.ToString("o")
    }
}
catch {
    $defender = [ordered]@{
        available = $false
        error = $_.Exception.Message
    }
}

$signatures = @(
    Test-Signature -Signtool $signtool -Path $AppExe
    Test-Signature -Signtool $signtool -Path $BridgeExe
    Test-Signature -Signtool $signtool -Path $DriverMsi
)

$artifacts = @(
    Get-ArtifactInfo -Path $AppExe
    Get-ArtifactInfo -Path $BridgeExe
    Get-ArtifactInfo -Path $DriverMsi
    Get-ArtifactInfo -Path $AppZip
)

$selfUseNotes = @(
    [ordered]@{ gate = "Unsigned app binaries"; status = "expected"; reason = "Self-use build does not require paid app code signing; SmartScreen may warn." }
    [ordered]@{ gate = "Test-signed driver"; status = "expected"; reason = "Self-use build requires Windows Test Mode and may require Secure Boot off." }
    [ordered]@{ gate = "Manual call software validation"; status = "manual"; reason = "Verify the one or two call apps you actually use before sharing." }
)

$formalReleaseOptions = @(
    [ordered]@{ gate = "Microsoft attestation or WHQL driver signing"; status = "deferred"; reason = "Requires paid/validated signing identity and Microsoft Partner Center submission." }
    [ordered]@{ gate = "Windows 10/11 clean VM install with Test Mode off"; status = "deferred"; reason = "Only needed for a formal public release." }
    [ordered]@{ gate = "SmartScreen reputation"; status = "deferred"; reason = "Only needed for broad public distribution." }
)

$report = [ordered]@{
    generated_at = (Get-Date).ToString("o")
    tools = [ordered]@{
        signtool = $signtool
        inf2cat = $inf2cat
        pnputil = $pnputil
    }
    code_signing_certificates = $certs
    artifacts = $artifacts
    signatures = $signatures
    defender = $defender
    self_use_notes = $selfUseNotes
    formal_release_options = $formalReleaseOptions
}

$jsonPath = Join-Path $ReportDir "release_readiness.json"
$mdPath = Join-Path $ReportDir "release_readiness.md"

$report | ConvertTo-Json -Depth 8 | Set-Content -Path $jsonPath -Encoding UTF8

$lines = @()
$lines += "# SyncTranslate Self-Use Release Readiness"
$lines += ""
$lines += "- Scope: free self-use / GitHub friends test build"
$lines += "- Expected driver mode: test-signed driver with Windows Test Mode"
$lines += "- Formal signing: deferred"
$lines += "- Generated: $($report.generated_at)"
$lines += "- signtool: $signtool"
$lines += "- inf2cat: $inf2cat"
$lines += "- pnputil: $pnputil"
$lines += ""
$lines += "## Artifacts"
foreach ($artifact in $artifacts) {
    $state = if ($artifact.exists) { "OK" } else { "MISSING" }
    $lines += "- [$state] $($artifact.path)"
    if ($artifact.exists) {
        $lines += "  - SHA256: $($artifact.sha256)"
    }
}
$lines += ""
$lines += "## Signatures"
foreach ($signature in $signatures) {
    $state = if ($signature.signed) { "SIGNED" } elseif ($signature.exists) { "UNSIGNED" } else { "MISSING" }
    $lines += "- [$state] $($signature.path)"
    if ($signature.signer) {
        $lines += "  - Signer: $($signature.signer)"
    }
}
$lines += ""
$lines += "## Self-Use Notes"
foreach ($gate in $selfUseNotes) {
    $lines += "- [$($gate.status.ToUpperInvariant())] $($gate.gate): $($gate.reason)"
}
$lines += ""
$lines += "## Formal Release Options"
foreach ($gate in $formalReleaseOptions) {
    $lines += "- [DEFERRED] $($gate.gate): $($gate.reason)"
}

Set-Content -Path $mdPath -Value $lines -Encoding UTF8

Write-Host "[release-gate] JSON: $jsonPath"
Write-Host "[release-gate] Markdown: $mdPath"
