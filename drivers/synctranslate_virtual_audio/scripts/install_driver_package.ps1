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
    [switch]$InstallComponentInfs,
    [switch]$StageComponentInfs,
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

function Invoke-NativeForDriverLog {
    param(
        [string]$FilePath,
        [string[]]$Arguments
    )
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $FilePath @Arguments 2>&1
        return [pscustomobject]@{
            Output = @($output)
            ExitCode = $LASTEXITCODE
        }
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
}

function Test-SyncTranslateMediaDeviceBound {
    param([string]$ExpectedHardwareId)
    try {
        $devices = @(Get-CimInstance Win32_PnPEntity -ErrorAction Stop | Where-Object {
            ($_.PNPDeviceID -like "ROOT\MEDIA\*") -and (
                ($_.Service -like "sysvad_componentizedaudiosample") -or
                (($_.HardwareID -join "`n") -like "*$ExpectedHardwareId*") -or
                ($_.Name -like "*SyncTranslate*Virtual*")
            )
        })
        $healthy = @($devices | Where-Object {
            [int]($_.ConfigManagerErrorCode) -eq 0 -and
            $_.Status -in @("OK", $null, "") -and
            $_.Service -like "sysvad_componentizedaudiosample"
        })
        foreach ($device in $devices) {
            Write-DriverLog "[driver-install] media device: name='$($device.Name)' id='$($device.PNPDeviceID)' service='$($device.Service)' status='$($device.Status)' cm_error='$($device.ConfigManagerErrorCode)'"
        }
        return ($healthy.Count -gt 0)
    }
    catch {
        Write-DriverLog "[driver-install] WARNING: Unable to verify MEDIA device binding: $($_.Exception.Message)"
        return $false
    }
}

function Get-SyncTranslateMediaDevices {
    param([string]$ExpectedHardwareId)
    return @(Get-CimInstance Win32_PnPEntity -ErrorAction SilentlyContinue | Where-Object {
        ($_.PNPDeviceID -like "ROOT\MEDIA\*") -and (
            ($_.Service -like "sysvad_componentizedaudiosample") -or
            (($_.HardwareID -join "`n") -like "*$ExpectedHardwareId*") -or
            ($_.Name -like "*SyncTranslate*Virtual*")
        )
    })
}

function Remove-StaleSyncTranslateMediaDevices {
    param(
        [string]$PnPUtilPath,
        [string]$ExpectedHardwareId
    )
    if (!$PnPUtilPath) {
        Write-DriverLog "[driver-install] WARNING: pnputil.exe was not found; stale SyncTranslate ROOT\MEDIA devices cannot be removed before install."
        return
    }
    $devices = Get-SyncTranslateMediaDevices -ExpectedHardwareId $ExpectedHardwareId
    foreach ($device in $devices) {
        Write-DriverLog "[driver-install] removing stale SyncTranslate media device before install: $($device.PNPDeviceID)"
        $removeResult = Invoke-NativeForDriverLog -FilePath $PnPUtilPath -Arguments @("/remove-device", $device.PNPDeviceID)
        $removeResult.Output | ForEach-Object { Write-DriverLog $_ }
        if ($removeResult.ExitCode -ne 0) {
            Write-DriverLog "[driver-install] WARNING: pnputil /remove-device returned exit code $($removeResult.ExitCode) for $($device.PNPDeviceID)"
        }
    }
    if ($devices.Count -gt 0) {
        Start-Sleep -Seconds 1
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

    if (Test-CertificateFile -Path $CertificatePath) {
        Write-DriverLog "[driver-install] importing test certificate before preflight"
        Import-Certificate -FilePath $CertificatePath -CertStoreLocation Cert:\LocalMachine\Root | Out-Null
        Import-Certificate -FilePath $CertificatePath -CertStoreLocation Cert:\LocalMachine\TrustedPublisher | Out-Null
    }
    elseif (Test-Path $CertificatePath) {
        Write-DriverLog "[driver-install] WARNING: Certificate exists but is not a valid X.509 certificate: $CertificatePath. Continuing; preflight will report signature trust status."
    }
    elseif ($RequireTestSigning) {
        Write-DriverLog "[driver-install] WARNING: Certificate not found before preflight: $CertificatePath. Signature verification may fail until the certificate is trusted."
    }

    if (!$SkipPreflight) {
        $scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
        $preflightScript = Join-Path $scriptRoot "preflight_driver_install.ps1"
        if (Test-Path $preflightScript) {
            Write-DriverLog "[driver-install] running preflight checks"
            & $preflightScript -PackageDir $PackageDir -MsiPath $MsiPath -CertificatePath $CertificatePath 2>&1 |
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

    if (Test-CertificateFile -Path $CertificatePath) {
        Write-DriverLog "[driver-install] importing test certificate into LocalMachine stores"
        Import-Certificate -FilePath $CertificatePath -CertStoreLocation Cert:\LocalMachine\Root | Out-Null
        Import-Certificate -FilePath $CertificatePath -CertStoreLocation Cert:\LocalMachine\TrustedPublisher | Out-Null
    }
    elseif (Test-Path $CertificatePath) {
        Write-DriverLog "[driver-install] WARNING: Certificate exists but is invalid and was not imported: $CertificatePath"
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
    Remove-StaleSyncTranslateMediaDevices -PnPUtilPath $pnputil -ExpectedHardwareId $HardwareId
    if ($devcon) {
        Write-DriverLog "[driver-install] installing root-enumerated audio device with devcon: $devcon"
        $devconResult = Invoke-NativeForDriverLog -FilePath $devcon -Arguments @("install", $inf, $HardwareId)
        $devconOutput = $devconResult.Output
        $devconExitCode = $devconResult.ExitCode
        $devconText = ($devconOutput | Out-String).Trim()
        $devconOutput | ForEach-Object { Write-DriverLog $_ }
        $devconReportedSuccess = $devconText -match "Drivers installed successfully|updated successfully"
        if ($devconExitCode -ne 0 -and !$devconReportedSuccess) {
            throw "devcon install failed with exit code $devconExitCode"
        }
        if ($devconExitCode -ne 0 -and $devconReportedSuccess) {
            Write-DriverLog "[driver-install] WARNING: devcon returned exit code $devconExitCode but reported success; continuing after successful device creation."
        }
        if ($pnputil) {
            Write-DriverLog "[driver-install] reconciling driver store binding with pnputil: $pnputil"
            $pnputilResult = Invoke-NativeForDriverLog -FilePath $pnputil -Arguments @("/add-driver", $inf, "/install")
            $pnputilOutput = $pnputilResult.Output
            $pnputilExitCode = $pnputilResult.ExitCode
            $pnputilOutput | ForEach-Object { Write-DriverLog $_ }
            if ($pnputilExitCode -ne 0 -and $pnputilExitCode -ne 259) {
                Write-DriverLog "[driver-install] WARNING: pnputil returned exit code $pnputilExitCode after devcon. Continuing because devcon already created the root device."
            }
        }
    }
    else {
        Write-DriverLog "[driver-install] WARNING: devcon.exe was not found. pnputil can stage the package, but cannot create the root-enumerated SysVAD device."
        if (!$pnputil) {
            throw "Neither devcon.exe nor pnputil.exe was found."
        }
        Write-DriverLog "[driver-install] staging package with pnputil: $pnputil"
        $pnputilResult = Invoke-NativeForDriverLog -FilePath $pnputil -Arguments @("/add-driver", $inf, "/install")
        $pnputilOutput = $pnputilResult.Output
        $pnputilExitCode = $pnputilResult.ExitCode
        $pnputilOutput | ForEach-Object { Write-DriverLog $_ }
        if ($pnputilExitCode -ne 0) {
            if ($pnputilExitCode -eq 259) {
                Write-DriverLog "[driver-install] pnputil returned 259 (already staged / latest version). Continue as success."
                $global:LASTEXITCODE = 0
            }
            else {
                throw "pnputil failed with exit code $pnputilExitCode"
            }
        }

        $rootDeviceScript = Join-Path $scriptRoot "create_root_device.ps1"
        if (!(Test-Path $rootDeviceScript)) {
            throw "devcon.exe was not found and SetupAPI fallback script is missing: $rootDeviceScript"
        }
        Write-DriverLog "[driver-install] creating root-enumerated audio device with SetupAPI fallback: $rootDeviceScript"
        $fallbackResult = Invoke-NativeForDriverLog -FilePath "powershell.exe" -Arguments @(
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            $rootDeviceScript,
            "-HardwareId",
            $HardwareId,
            "-InfPath",
            $inf
        )
        $fallbackResult.Output | ForEach-Object { Write-DriverLog $_ }
        if ($fallbackResult.ExitCode -ne 0) {
            throw "SetupAPI root-device fallback failed with exit code $($fallbackResult.ExitCode)"
        }

        Write-DriverLog "[driver-install] binding staged driver to the SetupAPI-created root device with pnputil"
        $bindResult = Invoke-NativeForDriverLog -FilePath $pnputil -Arguments @("/add-driver", $inf, "/install")
        $bindResult.Output | ForEach-Object { Write-DriverLog $_ }
        if ($bindResult.ExitCode -ne 0 -and $bindResult.ExitCode -ne 259) {
            throw "pnputil driver bind failed after SetupAPI fallback with exit code $($bindResult.ExitCode)"
        }
    }

    if ($pnputil -and $InstallComponentInfs) {
        foreach ($componentInf in @("ComponentizedApoSample.inf", "ComponentizedAudioSampleExtension.inf")) {
            $componentInfPath = Join-Path $PackageDir $componentInf
            if (Test-Path $componentInfPath) {
                Write-DriverLog "[driver-install] installing componentized INF with pnputil: $componentInf"
                $componentResult = Invoke-NativeForDriverLog -FilePath $pnputil -Arguments @("/add-driver", $componentInfPath, "/install")
                $componentOutput = $componentResult.Output
                $componentExitCode = $componentResult.ExitCode
                $componentOutput | ForEach-Object { Write-DriverLog $_ }
                if ($componentExitCode -ne 0 -and $componentExitCode -ne 259) {
                    throw "pnputil failed for $componentInf with exit code $componentExitCode"
                }
            }
        }
    }
    elseif ($pnputil -and $StageComponentInfs) {
        foreach ($componentInf in @("ComponentizedApoSample.inf", "ComponentizedAudioSampleExtension.inf")) {
            $componentInfPath = Join-Path $PackageDir $componentInf
            if (Test-Path $componentInfPath) {
                Write-DriverLog "[driver-install] staging optional componentized INF without /install: $componentInf"
                $componentResult = Invoke-NativeForDriverLog -FilePath $pnputil -Arguments @("/add-driver", $componentInfPath)
                $componentOutput = $componentResult.Output
                $componentExitCode = $componentResult.ExitCode
                $componentOutput | ForEach-Object { Write-DriverLog $_ }
                if ($componentExitCode -ne 0 -and $componentExitCode -ne 259) {
                    throw "pnputil staging failed for $componentInf with exit code $componentExitCode"
                }
            }
        }
    }
    elseif ($pnputil) {
        Write-DriverLog "[driver-install] skipping optional componentized APO/extension INFs; base virtual audio endpoint install is the productized default."
    }

    if (!(Test-SyncTranslateMediaDeviceBound -ExpectedHardwareId $HardwareId)) {
        throw "SyncTranslate MEDIA driver did not bind successfully. Check the package CAT/INF/SYS files, driver signature trust, and Device Manager problem code."
    }

    Write-DriverLog "[driver-install] done. Check Device Manager and Windows Sound Settings for the virtual audio endpoints."
    Write-Host "[SyncTranslate] Virtual audio driver package installed."
    exit 0
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
