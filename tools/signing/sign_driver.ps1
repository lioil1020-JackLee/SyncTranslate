# Sign a SyncTranslate virtual audio driver package with an installed code-signing certificate.
#
# This script expects a built driver package directory containing INF/SYS files.
# It copies the package to -OutputDir, regenerates a catalog when inf2cat is
# available, signs SYS/CAT files with signtool /sha1, and verifies signatures.

param(
    [string]$SourceDir = "artifacts/driver/synctranslate_virtual_audio/package",
    [string]$CertThumbprint = "",
    [string]$OutputDir = "artifacts/driver/signed",
    [string]$TimestampUrl = "http://timestamp.digicert.com",
    [string]$Description = "SyncTranslate Virtual Audio Driver",
    [switch]$SkipCatalog,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Resolve-Tool {
    param(
        [string]$Name,
        [string]$InstallHint
    )
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
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
        $match = Get-ChildItem -Path $root -Filter $Name -Recurse -File -ErrorAction SilentlyContinue |
            Sort-Object FullName -Descending |
            Select-Object -First 1
        if ($match) {
            return $match.FullName
        }
    }
    throw "$Name was not found. $InstallHint"
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

function Sign-Target {
    param(
        [string]$Path,
        [string]$Thumbprint
    )
    Write-Host "[sign] $Path"
    Invoke-Signtool @(
        "sign",
        "/sha1", $Thumbprint,
        "/fd", "sha256",
        "/tr", $TimestampUrl,
        "/td", "sha256",
        "/d", $Description,
        $Path
    )
}

function Verify-Target {
    param([string]$Path)

    Write-Host "[verify] $Path"
    Invoke-Signtool @("verify", "/pa", "/v", $Path)
}

if (!(Test-Path $SourceDir)) {
    throw "Source driver package directory not found: $SourceDir"
}

$script:signtool = Resolve-Tool "signtool.exe" "Install the Windows SDK and ensure signtool.exe is on PATH."
try {
    $inf2cat = Resolve-Tool "inf2cat.exe" "Install the Windows Driver Kit."
}
catch {
    $inf2cat = ""
}
$cert = Get-CodeSigningCertificate -Thumbprint $CertThumbprint
$thumbprint = ($cert.Thumbprint -replace "\s", "").ToUpperInvariant()

Write-Host "[cert] Using $thumbprint ($($cert.Subject))"
Write-Host "[tool] signtool=$script:signtool"
if ($inf2cat) {
    Write-Host "[tool] inf2cat=$inf2cat"
}
elseif (!$SkipCatalog) {
    throw "inf2cat.exe was not found. Install the Windows Driver Kit or rerun with -SkipCatalog to sign an existing catalog."
}

if ((Test-Path $OutputDir) -and !$Force) {
    throw "OutputDir already exists: $OutputDir. Use -Force to replace it."
}
if (Test-Path $OutputDir) {
    Remove-Item -LiteralPath $OutputDir -Recurse -Force
}
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
Copy-Item -Path (Join-Path $SourceDir "*") -Destination $OutputDir -Recurse -Force

$infFiles = @(Get-ChildItem -Path $OutputDir -Filter "*.inf" -File)
$sysFiles = @(Get-ChildItem -Path $OutputDir -Filter "*.sys" -File)
if ($infFiles.Count -eq 0) {
    throw "No INF files found in $OutputDir."
}
if ($sysFiles.Count -eq 0) {
    throw "No SYS files found in $OutputDir."
}

if (!$SkipCatalog) {
    Write-Host "[catalog] Regenerating catalog with inf2cat"
    & $inf2cat /driver:$OutputDir /os:10_X64,Server2016_X64,Server2019_X64,Server2022_X64
    if ($LASTEXITCODE -ne 0) {
        throw "inf2cat failed with exit code $LASTEXITCODE."
    }
}

$catFiles = @(Get-ChildItem -Path $OutputDir -Filter "*.cat" -File)
if ($catFiles.Count -eq 0) {
    throw "No CAT files found in $OutputDir after catalog step."
}

foreach ($sys in $sysFiles) {
    Sign-Target -Path $sys.FullName -Thumbprint $thumbprint
    Verify-Target -Path $sys.FullName
}
foreach ($cat in $catFiles) {
    Sign-Target -Path $cat.FullName -Thumbprint $thumbprint
    Verify-Target -Path $cat.FullName
}

Write-Host "[done] Signed driver package: $OutputDir"
