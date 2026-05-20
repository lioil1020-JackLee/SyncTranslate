# Sign SyncTranslate executable artifacts with an installed code-signing certificate.

param(
    [string[]]$Path = @(
        "dist/SyncTranslate-onedir/SyncTranslate.exe",
        "dist/SyncTranslate-onedir/runtimes/audio/sync_audio_bridge.exe"
    ),
    [string]$CertThumbprint = "",
    [string]$TimestampUrl = "http://timestamp.digicert.com",
    [string]$Description = "SyncTranslate"
)

$ErrorActionPreference = "Stop"

function Resolve-Signtool {
    $cmd = Get-Command signtool.exe -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    $searchRoots = @(
        "${env:ProgramFiles(x86)}\Windows Kits\10",
        "${env:ProgramFiles}\Windows Kits\10"
    )
    foreach ($root in $searchRoots) {
        if (!(Test-Path $root)) {
            continue
        }
        $match = Get-ChildItem -Path $root -Filter "signtool.exe" -Recurse -File -ErrorAction SilentlyContinue |
            Sort-Object FullName -Descending |
            Select-Object -First 1
        if ($match) {
            return $match.FullName
        }
    }
    throw "signtool.exe was not found. Install the Windows SDK."
}

function Get-CodeSigningCertificate {
    param([string]$Thumbprint)

    $normalizedThumbprint = ($Thumbprint -replace "\s", "").ToUpperInvariant()
    if ($normalizedThumbprint) {
        $cert = Get-ChildItem Cert:\CurrentUser\My, Cert:\LocalMachine\My -CodeSigningCert |
            Where-Object { ($_.Thumbprint -replace "\s", "").ToUpperInvariant() -eq $normalizedThumbprint } |
            Select-Object -First 1
        if (!$cert) {
            throw "Code-signing certificate not found for thumbprint $Thumbprint."
        }
        return $cert
    }

    $certs = @(Get-ChildItem Cert:\CurrentUser\My, Cert:\LocalMachine\My -CodeSigningCert |
        Where-Object { $_.HasPrivateKey } |
        Sort-Object NotAfter -Descending)
    if ($certs.Count -eq 0) {
        throw "No code-signing certificate with private key was found in CurrentUser\My or LocalMachine\My."
    }
    if ($certs.Count -eq 1) {
        return $certs[0]
    }

    Write-Host "[cert] Available code-signing certificates:"
    for ($i = 0; $i -lt $certs.Count; $i++) {
        $cert = $certs[$i]
        Write-Host ("  [{0}] {1}  NotAfter={2:u}  Subject={3}" -f $i, $cert.Thumbprint, $cert.NotAfter, $cert.Subject)
    }
    $choice = Read-Host "Select certificate index or thumbprint"
    if ($choice -match "^\d+$") {
        $index = [int]$choice
        if ($index -lt 0 -or $index -ge $certs.Count) {
            throw "Certificate index out of range: $choice"
        }
        return $certs[$index]
    }

    $selected = $certs | Where-Object { ($_.Thumbprint -replace "\s", "").ToUpperInvariant() -eq (($choice -replace "\s", "").ToUpperInvariant()) } | Select-Object -First 1
    if (!$selected) {
        throw "Certificate not selected."
    }
    return $selected
}

function Invoke-Signtool {
    param([string[]]$Arguments)

    & $script:signtool @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "signtool failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

$script:signtool = Resolve-Signtool
$cert = Get-CodeSigningCertificate -Thumbprint $CertThumbprint
$thumbprint = ($cert.Thumbprint -replace "\s", "").ToUpperInvariant()
Write-Host "[cert] Using $thumbprint ($($cert.Subject))"

foreach ($target in $Path) {
    if (!(Test-Path $target)) {
        throw "Executable not found: $target"
    }
    Write-Host "[sign] $target"
    Invoke-Signtool @(
        "sign",
        "/sha1", $thumbprint,
        "/fd", "sha256",
        "/tr", $TimestampUrl,
        "/td", "sha256",
        "/d", $Description,
        $target
    )
    Write-Host "[verify] $target"
    Invoke-Signtool @("verify", "/pa", "/v", $target)
}

Write-Host "[done] Executable signatures verified."
