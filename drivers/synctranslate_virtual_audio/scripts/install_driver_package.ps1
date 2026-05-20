param(
    [string]$PackageDir = "artifacts/driver/synctranslate_virtual_audio/package",
    [string]$MsiPath = "artifacts/driver/synctranslate_virtual_audio/SyncTranslateVirtualAudioDriver.msi",
    [string]$CertificatePath = "artifacts/driver/synctranslate_virtual_audio/SyncTranslateVirtualAudioTest.cer",
    [string]$InfName = "",
    [string]$HardwareId = "Root\SyncTranslateVirtualAudio",
    [string]$DevconPath = "",
    [string]$LogPath = "$env:ProgramData\SyncTranslate\Logs\driver-install.log",
    [switch]$RequireTestSigning,
    [switch]$AllowHostInstall,
    [switch]$SkipPreflight
)

$ErrorActionPreference = "Stop"

function Write-DriverLog {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host $Message
    Add-Content -Path $script:LogPath -Value "[$timestamp] $Message" -Encoding UTF8
}

function Find-Tool {
    param(
        [string]$Name,
        [string[]]$Candidates
    )
    foreach ($candidate in $Candidates) {
        if ($candidate -and (Test-Path $candidate)) {
            return (Resolve-Path $candidate).Path
        }
    }
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }
    return ""
}

function Test-TestSigningEnabled {
    try {
        $bcdedit = Join-Path $env:windir "System32\bcdedit.exe"
        if (!(Test-Path $bcdedit)) {
            $cmd = Get-Command bcdedit.exe -ErrorAction SilentlyContinue
            if (!$cmd) {
                Write-DriverLog "[driver-install] WARNING: bcdedit.exe was not found."
                return $false
            }
            $bcdedit = $cmd.Source
        }
        $bcd = & $bcdedit /enum 2>&1
        $exitCode = $LASTEXITCODE
        if ($exitCode -ne 0) {
            Write-DriverLog "[driver-install] WARNING: bcdedit /enum failed with exit code $exitCode"
            ($bcd | Out-String).Trim() -split "`r?`n" | ForEach-Object {
                if ($_ -and $_.Trim()) {
                    Write-DriverLog "[driver-install] bcdedit: $($_.Trim())"
                }
            }
            return $false
        }
        return [bool]($bcd | Where-Object { $_ -match "testsigning\s+Yes|testsigning\s+on" })
    }
    catch {
        Write-DriverLog "[driver-install] WARNING: Test Mode check failed: $($_.Exception.Message)"
        return $false
    }
}

try {
    $script:LogPath = $LogPath
    $logParent = Split-Path -Parent $script:LogPath
    if ($logParent) {
        New-Item -ItemType Directory -Path $logParent -Force | Out-Null
    }
    Set-Content -Path $script:LogPath -Value "[driver-install] started $(Get-Date -Format o)" -Encoding UTF8

    if (!$AllowHostInstall) {
        throw "Host driver install is blocked by default. Run this only in a disposable Windows VM, or pass -AllowHostInstall after you have a restore point and accept BSOD risk."
    }

    if (!$SkipPreflight) {
        $scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
        $preflightScript = Join-Path $scriptRoot "preflight_driver_install.ps1"
        if (Test-Path $preflightScript) {
            Write-DriverLog "[driver-install] running preflight checks"
            & $preflightScript -PackageDir $PackageDir -MsiPath $MsiPath -CertificatePath $CertificatePath -AllowExistingSyncTranslateDevices 2>&1 |
                ForEach-Object { Write-DriverLog $_ }
            if ($LASTEXITCODE -ne 0) {
                throw "driver preflight failed with exit code $LASTEXITCODE"
            }
        }
        else {
            throw "preflight script not found: $preflightScript"
        }
    }

    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    if (!$principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw "Run this script from an elevated PowerShell."
    }

    Write-DriverLog "[driver-install] identity: $($identity.Name)"
    Write-DriverLog "[driver-install] package: $PackageDir"

    if (!(Test-Path $PackageDir)) {
        throw "Missing driver package directory: $PackageDir"
    }

    $testSigningEnabled = Test-TestSigningEnabled
    if (!$testSigningEnabled) {
        $message = "Windows Test Mode could not be confirmed. If driver install fails with a signature/trust error, run enable_test_mode.ps1 from an elevated PowerShell, reboot, then retry."
        if ($RequireTestSigning) {
            Write-DriverLog "[driver-install] ERROR: $message"
            throw $message
        }
        Write-DriverLog "[driver-install] WARNING: $message"
    }

    if (Test-Path $CertificatePath) {
        Write-DriverLog "[driver-install] importing test certificate into LocalMachine stores"
        Import-Certificate -FilePath $CertificatePath -CertStoreLocation Cert:\LocalMachine\Root | Out-Null
        Import-Certificate -FilePath $CertificatePath -CertStoreLocation Cert:\LocalMachine\TrustedPublisher | Out-Null
    }
    else {
        Write-DriverLog "[driver-install] WARNING: Certificate not found: $CertificatePath. Install may fail unless the CAT is already trusted."
    }

    if ($InfName) {
        $inf = Join-Path $PackageDir $InfName
    }
    else {
        $preferred = @("ComponentizedAudioSample.inf", "TabletAudioSample.inf")
        $inf = ""
        foreach ($candidate in $preferred) {
            $path = Join-Path $PackageDir $candidate
            if (Test-Path $path) {
                $inf = $path
                break
            }
        }
        if (!$inf) {
            $inf = (Get-ChildItem -Path $PackageDir -Filter "*.inf" -File | Select-Object -First 1).FullName
        }
    }

    if (!(Test-Path $inf)) {
        throw "INF not found: $inf"
    }
    Write-DriverLog "[driver-install] inf: $inf"

    $sysnativePnputil = Join-Path $env:windir "Sysnative\pnputil.exe"
    $system32Pnputil = Join-Path $env:windir "System32\pnputil.exe"
    $pnputil = Find-Tool -Name "pnputil.exe" -Candidates @($sysnativePnputil, $system32Pnputil)

    $scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
    $localDevcon = Join-Path $scriptRoot "devcon.exe"
    $wdkDevcon = "${env:ProgramFiles(x86)}\Windows Kits\10\Tools\10.0.26100.0\x64\devcon.exe"
    $devcon = Find-Tool -Name "devcon.exe" -Candidates @($DevconPath, $localDevcon, $wdkDevcon)
    if ($devcon) {
        Write-DriverLog "[driver-install] installing root-enumerated audio device with devcon: $devcon"
        & $devcon install $inf $HardwareId 2>&1 | ForEach-Object { Write-DriverLog $_ }
        if ($LASTEXITCODE -ne 0) {
            throw "devcon install failed with exit code $LASTEXITCODE"
        }
    }
    else {
        Write-DriverLog "[driver-install] WARNING: devcon.exe was not found. pnputil can stage the package, but cannot create the root-enumerated SysVAD device."
        if (!$pnputil) {
            throw "Neither devcon.exe nor pnputil.exe was found."
        }
        Write-DriverLog "[driver-install] staging package with pnputil: $pnputil"
        $pnputilOutput = & $pnputil /add-driver $inf /install 2>&1
        $pnputilOutput | ForEach-Object { Write-DriverLog $_ }
        if ($LASTEXITCODE -ne 0) {
            if ($LASTEXITCODE -eq 259) {
                Write-DriverLog "[driver-install] pnputil returned 259 (already staged / latest version). Continue as success."
                $global:LASTEXITCODE = 0
            }
            else {
                throw "pnputil failed with exit code $LASTEXITCODE"
            }
        }
    }

    Write-DriverLog "[driver-install] done. Check Device Manager and Windows Sound Settings for the virtual audio endpoints."
    Write-Host "[SyncTranslate] Virtual audio driver package installed."
}
catch {
    try {
        if (!$script:LogPath) {
            $script:LogPath = $LogPath
        }
        Write-DriverLog "[driver-install] ERROR: $($_.Exception.Message)"
        Write-DriverLog "[driver-install] ERROR_TYPE: $($_.Exception.GetType().FullName)"
        if ($_.ScriptStackTrace) {
            Write-DriverLog "[driver-install] STACK: $($_.ScriptStackTrace)"
        }
    }
    catch {
        Write-Error $_
    }
    exit 1
}
