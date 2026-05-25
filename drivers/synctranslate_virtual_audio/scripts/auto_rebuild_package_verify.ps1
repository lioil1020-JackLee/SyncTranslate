param(
    [string]$SamplesRoot = "downloads/driver_samples/Windows-driver-samples",
    [string]$Configuration = "Release",
    [string]$Platform = "x64",
    [string]$ArtifactsDir = "artifacts/driver/synctranslate_virtual_audio",
    [string]$ProductVersion = "2.1.1",
    [string]$PythonExe = "e:/py/SyncTranslate/.venv/Scripts/python.exe",
    [int]$MaxAttempts = 2,
    [switch]$SkipBuild,
    [switch]$SkipFetch,
    [switch]$SkipSign,
    [switch]$SkipInstall,
    [switch]$AllowHostInstall
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptRoot "../../..")).Path

function Assert-Elevated {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    if (!$principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw "Run this script from an elevated PowerShell."
    }
}

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Action
    )
    Write-Host ""
    Write-Host "========== $Name =========="
    & $Action
}

function Run-External {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$WorkingDirectory = ""
    )
    if ($WorkingDirectory) {
        Push-Location $WorkingDirectory
    }
    try {
        & $FilePath @Arguments
        $exitCode = $LASTEXITCODE
        if ($exitCode -ne 0) {
            throw "Command failed ($exitCode): $FilePath $($Arguments -join ' ')"
        }
    }
    finally {
        if ($WorkingDirectory) {
            Pop-Location
        }
    }
}

function Assert-LastExitCode {
    param([string]$Context)
    if ($LASTEXITCODE -ne 0) {
        throw "$Context failed with exit code $LASTEXITCODE"
    }
}

function Read-JsonFile {
    param([string]$Path)
    if (!(Test-Path $Path)) {
        throw "Missing JSON file: $Path"
    }
    return (Get-Content -Raw -Encoding UTF8 $Path | ConvertFrom-Json)
}

if (!$SkipInstall) {
    Assert-Elevated
}

$artifactsFull = (Resolve-Path (Join-Path $repoRoot $ArtifactsDir)).Path
$logsDir = Join-Path $artifactsFull "automation_logs"
New-Item -ItemType Directory -Path $logsDir -Force | Out-Null

$summary = [ordered]@{
    started_at = (Get-Date).ToString("o")
    attempts = @()
    ok = $false
    fail_reason = ""
}

Invoke-Step "Prerequisites" {
    $prereqScript = Join-Path $scriptRoot "check_prereqs.ps1"
    $prereqJson = & $prereqScript -Json | ConvertFrom-Json
    if (!$SkipBuild -and !$prereqJson.build_ready) {
        throw "Build prerequisites are incomplete. Run check_prereqs.ps1 for details."
    }
    if (!$prereqJson.msi_ready) {
        throw "MSI prerequisites are incomplete. Install wix.exe or .NET SDK."
    }
}

$msiPath = Join-Path $repoRoot "$ArtifactsDir/SyncTranslateVirtualAudioDriver.msi"
$packageDir = Join-Path $repoRoot "$ArtifactsDir/package"
$certPath = Join-Path $repoRoot "$ArtifactsDir/SyncTranslateVirtualAudioTest.cer"

if (!$SkipBuild) {
    Invoke-Step "Rebuild and package MSI" {
        $buildScript = Join-Path $scriptRoot "build_driver_msi.ps1"
        $args = @(
            "-SamplesRoot", $SamplesRoot,
            "-Configuration", $Configuration,
            "-Platform", $Platform,
            "-ArtifactsDir", (Join-Path $repoRoot $ArtifactsDir),
            "-ProductVersion", $ProductVersion
        )
        if ($SkipFetch) { $args += "-SkipFetch" }
        if ($SkipSign) { $args += "-SkipSign" }
        Push-Location $repoRoot
        try {
            & $buildScript @args
            Assert-LastExitCode -Context "build_driver_msi.ps1"
        }
        finally {
            Pop-Location
        }
    }
}
else {
    Invoke-Step "Repackage MSI from existing package" {
        $packageMsiScript = Join-Path $scriptRoot "package_driver_msi.ps1"
        Push-Location $repoRoot
        try {
            & $packageMsiScript -PackageDir $packageDir -CertificatePath $certPath -OutputMsi $msiPath -ProductVersion $ProductVersion
            Assert-LastExitCode -Context "package_driver_msi.ps1"
        }
        finally {
            Pop-Location
        }
    }
}

Invoke-Step "Validate package" {
    $validateScript = Join-Path $scriptRoot "validate_driver_package.ps1"
    Push-Location $repoRoot
    try {
        if ($SkipSign) {
            & $validateScript -PackageDir $packageDir -CertificatePath $certPath -SkipSignatureVerify
        }
        else {
            & $validateScript -PackageDir $packageDir -CertificatePath $certPath
        }
        Assert-LastExitCode -Context "validate_driver_package.ps1"
    }
    finally {
        Pop-Location
    }
    if (!(Test-Path $msiPath)) {
        throw "MSI was not generated: $msiPath"
    }
}

for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
    $attemptLog = [ordered]@{
        attempt = $attempt
        installed = $false
        endpoint_verify_ok = $false
        bridge_smoke_ok = $false
        ioctl_smoke_ok = $false
        notes = @()
    }

    try {
        if (!$SkipInstall) {
            if (!$AllowHostInstall) {
                throw "Install/test on host is blocked unless -AllowHostInstall is provided."
            }

            Invoke-Step "Install attempt #$attempt" {
                $installLog = Join-Path $logsDir "msi_install_attempt_${attempt}.log"
                Run-External -FilePath "msiexec.exe" -Arguments @("/i", $msiPath, "/qn", "/norestart", "/L*v", $installLog) -WorkingDirectory $repoRoot

                # Ensure root-enumerated device exists even if MSI custom action was skipped.
                $fallbackInstallScript = Join-Path $scriptRoot "install_driver_package.ps1"
                Run-External -FilePath $fallbackInstallScript -Arguments @(
                    "-PackageDir", $packageDir,
                    "-CertificatePath", $certPath,
                    "-AllowHostInstall",
                    "-SkipPreflight"
                ) -WorkingDirectory $repoRoot
            }
            $attemptLog.installed = $true
        }
        else {
            $attemptLog.notes += "Skipped install by request"
        }

        Invoke-Step "Verify endpoints attempt #$attempt" {
            $verifyScript = Join-Path $scriptRoot "verify_driver_install.ps1"
            Push-Location $repoRoot
            try {
                & $verifyScript
                Assert-LastExitCode -Context "verify_driver_install.ps1"
            }
            finally {
                Pop-Location
            }
        }
        $attemptLog.endpoint_verify_ok = $true

        $bridgeJson = Join-Path $logsDir "bridge_smoke_attempt_${attempt}.json"
        Invoke-Step "Bridge smoke (speaker path) attempt #$attempt" {
            $bridgeSmoke = Join-Path $repoRoot "tools/runtime_smoke/virtual_audio_driver_bridge_smoke.py"
            Push-Location $repoRoot
            try {
                & $PythonExe $bridgeSmoke --bridge-path "runtimes/audio/sync_audio_bridge.exe" --sample-rate 48000 --duration-sec 1.2 --frequency 440 --json-output $bridgeJson
                Assert-LastExitCode -Context "virtual_audio_driver_bridge_smoke.py"
            }
            finally {
                Pop-Location
            }
        }
        $bridgeResult = Read-JsonFile -Path $bridgeJson
        $attemptLog.bridge_smoke_ok = [bool]$bridgeResult.ok
        if (!$attemptLog.bridge_smoke_ok) {
            throw "Bridge smoke failed: remote_input_delta_frames <= 0"
        }

        $ioctlJson = Join-Path $logsDir "ioctl_smoke_attempt_${attempt}.json"
        $ioctlWav = Join-Path $logsDir "ioctl_smoke_attempt_${attempt}.wav"
        Invoke-Step "IOCTL smoke (microphone path) attempt #$attempt" {
            $ioctlSmoke = Join-Path $repoRoot "tools/runtime_smoke/virtual_audio_driver_ioctl_smoke.py"
            Push-Location $repoRoot
            try {
                & $PythonExe $ioctlSmoke --record-duration 2.0 --json-output $ioctlJson --wav-output $ioctlWav
                Assert-LastExitCode -Context "virtual_audio_driver_ioctl_smoke.py"
            }
            finally {
                Pop-Location
            }
        }
        $ioctlResult = Read-JsonFile -Path $ioctlJson
        $attemptLog.ioctl_smoke_ok = [bool]$ioctlResult.overall
        if (!$attemptLog.ioctl_smoke_ok) {
            throw "IOCTL smoke failed"
        }

        $summary.attempts += $attemptLog
        $summary.ok = $true
        $summary.fail_reason = ""
        break
    }
    catch {
        $attemptLog.notes += $_.Exception.Message
        $summary.attempts += $attemptLog
        $summary.fail_reason = $_.Exception.Message

        if ($attempt -lt $MaxAttempts -and !$SkipInstall) {
            Invoke-Step "Cleanup before retry #$($attempt + 1)" {
                $uninstallScript = Join-Path $scriptRoot "uninstall_driver_package.ps1"
                & $uninstallScript
            }
        }
    }
}

$summary.finished_at = (Get-Date).ToString("o")
$summaryPath = Join-Path $logsDir "auto_rebuild_package_verify_summary.json"
$summary | ConvertTo-Json -Depth 8 | Set-Content -Path $summaryPath -Encoding UTF8

Write-Host ""
Write-Host "Summary JSON: $summaryPath"
if ($summary.ok) {
    Write-Host "Automation pipeline PASS"
    exit 0
}

Write-Host "Automation pipeline FAIL: $($summary.fail_reason)"
exit 1
