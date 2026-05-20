param(
    [string]$PackageDir = "artifacts/driver/synctranslate_virtual_audio/package",
    [string]$CertificatePath = "artifacts/driver/synctranslate_virtual_audio/SyncTranslateVirtualAudioTest.cer",
    [switch]$SkipSignatureVerify
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $PackageDir)) {
    throw "Missing driver package directory: $PackageDir"
}

$infFiles = @(Get-ChildItem -Path $PackageDir -Filter "*.inf" -File)
$catFiles = @(Get-ChildItem -Path $PackageDir -Filter "*.cat" -File)
$sysFiles = @(Get-ChildItem -Path $PackageDir -Filter "*.sys" -File -Recurse)

if ($infFiles.Count -eq 0) {
    throw "Driver package must contain at least one INF file."
}
if ($catFiles.Count -eq 0) {
    throw "Driver package must contain at least one CAT file."
}
if ($sysFiles.Count -eq 0) {
    throw "Driver package must contain at least one SYS file."
}

$infText = ($infFiles | ForEach-Object { Get-Content -Raw -Encoding UTF8 $_.FullName }) -join "`n"
$requiredText = @(
    "Root\SyncTranslateVirtualAudio",
    "SyncTranslate Virtual Audio",
    "SyncTranslate Virtual Speaker",
    "SyncTranslate Virtual Microphone"
)

foreach ($item in $requiredText) {
    if ($infText -notlike "*$item*") {
        throw "Driver INF does not contain required SyncTranslate marker: $item"
    }
}

if (!(Test-Path $CertificatePath)) {
    Write-Warning "Certificate file not found: $CertificatePath"
}

if (!$SkipSignatureVerify) {
    $scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
    $prereq = & (Join-Path $scriptRoot "check_prereqs.ps1") -Json | ConvertFrom-Json
    if (!$prereq.signtool) {
        Write-Warning "signtool.exe not found; CAT signature verification was skipped."
    }
    else {
        foreach ($cat in $catFiles) {
            Write-Host "[driver-validate] verifying signature: $($cat.FullName)"
            $previousErrorActionPreference = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            $verifyOutput = & $prereq.signtool verify /pa /v $cat.FullName 2>&1
            $verifyExitCode = $LASTEXITCODE
            $ErrorActionPreference = $previousErrorActionPreference
            if ($verifyExitCode -ne 0) {
                $joinedOutput = ($verifyOutput | Out-String)
                if ((Test-Path $CertificatePath) -and $joinedOutput -match "not trusted by the trust provider") {
                    Write-Warning "CAT is signed, but the test certificate is not trusted on this machine yet. Import $CertificatePath during driver install/test-mode setup."
                }
                else {
                    $verifyOutput | Write-Host
                    throw "SignTool verification failed for $($cat.FullName)"
                }
            }
            else {
                $verifyOutput | Write-Host
            }
        }
    }
}

Write-Host "[driver-validate] package OK: $PackageDir"
Write-Host "[driver-validate] INF: $($infFiles.Count), CAT: $($catFiles.Count), SYS: $($sysFiles.Count)"
