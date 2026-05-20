param(
    [string]$PackageDir = "artifacts/driver/synctranslate_virtual_audio/package",
    [string]$CertificateSubject = "CN=SyncTranslate Virtual Audio Test",
    [string]$CertificatePath = "artifacts/driver/synctranslate_virtual_audio/SyncTranslateVirtualAudioTest.cer",
    [string]$OsTargets = "10_X64,10_19H1_X64,10_VB_X64,10_CO_X64,10_NI_X64,10_GE_X64"
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$prereq = & (Join-Path $scriptRoot "check_prereqs.ps1") -Json | ConvertFrom-Json
if (!$prereq.inf2cat -or !$prereq.signtool) {
    throw "inf2cat.exe and signtool.exe are required. Install the Windows Driver Kit first."
}

if (!(Test-Path $PackageDir)) {
    throw "Missing driver package directory: $PackageDir"
}

$infFiles = Get-ChildItem -Path $PackageDir -Filter "*.inf" -File
if (!$infFiles) {
    throw "No INF files found in $PackageDir"
}

Write-Host "[driver-sign] generating catalog with Inf2Cat"
& $prereq.inf2cat /driver:$PackageDir /os:$OsTargets /uselocaltime
if ($LASTEXITCODE -ne 0) {
    throw "Inf2Cat failed with exit code $LASTEXITCODE"
}

$cert = Get-ChildItem Cert:\CurrentUser\My |
    Where-Object { $_.Subject -eq $CertificateSubject } |
    Sort-Object NotAfter -Descending |
    Select-Object -First 1

if (!$cert) {
    Write-Host "[driver-sign] creating self-signed test certificate: $CertificateSubject"
    $cert = New-SelfSignedCertificate `
        -Subject $CertificateSubject `
        -Type CodeSigningCert `
        -CertStoreLocation Cert:\CurrentUser\My `
        -HashAlgorithm SHA256 `
        -KeyLength 2048 `
        -KeyExportPolicy Exportable `
        -NotAfter (Get-Date).AddYears(3)
}

$certParent = Split-Path -Parent $CertificatePath
if ($certParent) {
    New-Item -ItemType Directory -Path $certParent -Force | Out-Null
}
Export-Certificate -Cert $cert -FilePath $CertificatePath -Force | Out-Null

$catFiles = Get-ChildItem -Path $PackageDir -Filter "*.cat" -File
if (!$catFiles) {
    throw "No CAT files found after Inf2Cat."
}

foreach ($cat in $catFiles) {
    Write-Host "[driver-sign] signing $($cat.FullName)"
    & $prereq.signtool sign /v /fd SHA256 /s My /n ($CertificateSubject -replace "^CN=", "") /t http://timestamp.digicert.com $cat.FullName
    if ($LASTEXITCODE -ne 0) {
        throw "SignTool failed for $($cat.FullName) with exit code $LASTEXITCODE"
    }
}

Write-Host "[driver-sign] certificate exported: $CertificatePath"
