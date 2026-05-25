param(
    [Parameter(Mandatory = $true)]
    [string]$VMName,

    [string]$GuestUser = "",
    [string]$GuestPassword = "",
    [System.Management.Automation.PSCredential]$Credential,

    [string]$RepoPath = "",
    [string]$GuestWorkDir = "C:\SyncTranslate",

    [switch]$BuildInGuest,
    [switch]$SkipBuild,
    [switch]$EnableTestSigning,
    [switch]$CreateCheckpoint,
    [switch]$IncludeRuntimes,
    [string]$OutputDir = "downloads/validation/hyperv",
    [switch]$NoReboot,
    [switch]$CollectDiagnostics
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Status, [string]$Name, [string]$Message)
    $line = "[${Status}] ${Name}: ${Message}"
    Write-Host $line
    Add-Content -Path $script:HostLog -Value $line -Encoding UTF8
}

function Resolve-Credential {
    if ($Credential) { return $Credential }
    if (!$GuestUser) {
        throw "GuestUser or Credential is required. Do not put passwords in scripts; use Get-Credential or Read-Host -AsSecureString."
    }
    if (!$GuestPassword) {
        $securePassword = Read-Host -Prompt "Password for $GuestUser" -AsSecureString
    }
    else {
        $securePassword = ConvertTo-SecureString $GuestPassword -AsPlainText -Force
    }
    return [System.Management.Automation.PSCredential]::new($GuestUser, $securePassword)
}

function Assert-HyperVReady {
    if (!(Get-Module -ListAvailable -Name Hyper-V)) {
        throw "Hyper-V PowerShell module was not found. Enable Hyper-V Management Tools on the host."
    }
    Import-Module Hyper-V -ErrorAction Stop
    try {
        $vm = Get-VM -Name $VMName -ErrorAction Stop
    }
    catch {
        throw "Cannot access VM '$VMName': $($_.Exception.Message). Run this script from an elevated Hyper-V host PowerShell session, or verify the VM name with Get-VM."
    }
    if (!$vm) { throw "VM not found: $VMName" }
    if ($vm.State -ne "Running") {
        throw "VM '$VMName' is $($vm.State). Start it before running validation."
    }
    return $vm
}

function New-GuestSession {
    param([System.Management.Automation.PSCredential]$Cred)
    try {
        return New-PSSession -VMName $VMName -Credential $Cred -ErrorAction Stop
    }
    catch {
        throw "PowerShell Direct connection failed for VM '$VMName': $($_.Exception.Message)"
    }
}

function Invoke-GuestCommand {
    param(
        [System.Management.Automation.Runspaces.PSSession]$Session,
        [string]$Name,
        [scriptblock]$ScriptBlock,
        [object[]]$ArgumentList = @(),
        [switch]$ContinueOnError
    )
    Write-Step "PASS" "guest_step_start" $Name
    try {
        $output = Invoke-Command -Session $Session -ScriptBlock $ScriptBlock -ArgumentList $ArgumentList -ErrorAction Stop 2>&1
        foreach ($line in $output) {
            Add-Content -Path $script:GuestLog -Value "[$Name] $line" -Encoding UTF8
        }
        Write-Step "PASS" $Name "completed"
        $script:Summary[$Name] = "PASS"
        return $output
    }
    catch {
        $message = $_.Exception.Message
        Add-Content -Path $script:GuestLog -Value "[$Name] ERROR $message" -Encoding UTF8
        Write-Step "FAIL" $Name $message
        $script:Summary[$Name] = "FAIL: $message"
        if (!$ContinueOnError) { throw }
        return @()
    }
}

function Copy-RepoToGuest {
    param(
        [System.Management.Automation.Runspaces.PSSession]$Session,
        [string]$Source,
        [string]$Destination,
        [bool]$IncludeRuntimeAssets
    )
    if (!(Test-Path $Source)) { throw "RepoPath not found: $Source" }
    $copyVmFile = Get-Command Copy-VMFile -ErrorAction SilentlyContinue
    if (!$copyVmFile) {
        throw "Copy-VMFile is not available. Alternative: git clone the repo inside the VM or mount a shared folder, then pass -SkipBuild/-GuestWorkDir to that path."
    }

    $stage = Join-Path ([System.IO.Path]::GetTempPath()) ("synctranslate_hyperv_" + [guid]::NewGuid().ToString("N"))
    $zip = "$stage.zip"
    New-Item -ItemType Directory -Path $stage -Force | Out-Null
    try {
        $excludeNames = @(
            ".git",
            ".venv",
            "__pycache__",
            ".pytest_cache",
            "artifacts",
            "build",
            "dist",
            "downloads",
            "image",
            "logs",
            "synctranslate.egg-info"
        )
        if (!$IncludeRuntimeAssets) {
            $excludeNames += "runtimes"
        }
        Get-ChildItem -LiteralPath $Source -Force | Where-Object { $excludeNames -notcontains $_.Name } | ForEach-Object {
            Copy-Item -LiteralPath $_.FullName -Destination $stage -Recurse -Force
        }
        Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $zip -Force
        $guestZip = "C:\SyncTranslate_validation_repo.zip"
        Copy-VMFile -Name $VMName -SourcePath $zip -DestinationPath $guestZip -FileSource Host -CreateFullPath -ErrorAction Stop
        Invoke-GuestCommand -Session $Session -Name "expand_repo" -ArgumentList @($guestZip, $Destination, $IncludeRuntimeAssets) -ScriptBlock {
            param($GuestZip, $Destination, $IncludeRuntimeAssets)
            $preserveRoot = "C:\SyncTranslate_preserve_driver_samples"
            $driverSamples = Join-Path $Destination "downloads\driver_samples"
            $runtimeAssets = Join-Path $Destination "runtimes"
            if (Test-Path $preserveRoot) { Remove-Item -LiteralPath $preserveRoot -Recurse -Force }
            if (Test-Path $driverSamples) {
                New-Item -ItemType Directory -Path $preserveRoot -Force | Out-Null
                Move-Item -LiteralPath $driverSamples -Destination (Join-Path $preserveRoot "driver_samples") -Force
            }
            if (!$IncludeRuntimeAssets -and (Test-Path $runtimeAssets)) {
                New-Item -ItemType Directory -Path $preserveRoot -Force | Out-Null
                Move-Item -LiteralPath $runtimeAssets -Destination (Join-Path $preserveRoot "runtimes") -Force
            }
            if (Test-Path $Destination) { Remove-Item -LiteralPath $Destination -Recurse -Force }
            New-Item -ItemType Directory -Path $Destination -Force | Out-Null
            Expand-Archive -LiteralPath $GuestZip -DestinationPath $Destination -Force
            $preservedDriverSamples = Join-Path $preserveRoot "driver_samples"
            if (Test-Path $preservedDriverSamples) {
                New-Item -ItemType Directory -Path (Join-Path $Destination "downloads") -Force | Out-Null
                Move-Item -LiteralPath $preservedDriverSamples -Destination (Join-Path $Destination "downloads\driver_samples") -Force
                Remove-Item -LiteralPath $preserveRoot -Recurse -Force -ErrorAction SilentlyContinue
            }
            $preservedRuntimes = Join-Path $preserveRoot "runtimes"
            if (Test-Path $preservedRuntimes) {
                Move-Item -LiteralPath $preservedRuntimes -Destination (Join-Path $Destination "runtimes") -Force
                Remove-Item -LiteralPath $preserveRoot -Recurse -Force -ErrorAction SilentlyContinue
            }
            Remove-Item -LiteralPath $GuestZip -Force
            "expanded repo to $Destination"
        }
    }
    finally {
        Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $zip -Force -ErrorAction SilentlyContinue
    }
}

function Wait-GuestReconnected {
    param([System.Management.Automation.PSCredential]$Cred)
    Write-Step "WARN" "reboot" "waiting for VM to reboot and PowerShell Direct to reconnect"
    $deadline = (Get-Date).AddMinutes(8)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 10
        try {
            $session = New-PSSession -VMName $VMName -Credential $Cred -ErrorAction Stop
            Write-Step "PASS" "reboot" "VM reconnected"
            return $session
        }
        catch {
            Add-Content -Path $script:HostLog -Value "[reboot] waiting: $($_.Exception.Message)" -Encoding UTF8
        }
    }
    throw "VM did not reconnect through PowerShell Direct within timeout."
}

function Copy-ValidationArtifacts {
    param([System.Management.Automation.Runspaces.PSSession]$Session)
    $target = Resolve-Path -LiteralPath $OutputDir -ErrorAction SilentlyContinue
    if (!$target) {
        New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
        $target = Resolve-Path -LiteralPath $OutputDir
    }
    $guestValidation = Join-Path $GuestWorkDir "downloads\validation"
    try {
        Copy-Item -FromSession $Session -Path $guestValidation -Destination $target.Path -Recurse -Force -ErrorAction Stop
        Write-Step "PASS" "collect_validation" "copied $guestValidation to $($target.Path)"
    }
    catch {
        Write-Step "WARN" "collect_validation" "could not copy validation directory: $($_.Exception.Message)"
    }
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
$script:HostLog = Join-Path $OutputDir "host_driver_vm_validation_$timestamp.log"
$script:GuestLog = Join-Path $OutputDir "guest_driver_vm_validation_$timestamp.log"
$script:Summary = [ordered]@{}
Set-Content -Path $script:HostLog -Value "[host] SyncTranslate Hyper-V driver validation started $(Get-Date -Format o)" -Encoding UTF8
Set-Content -Path $script:GuestLog -Value "[guest] SyncTranslate Hyper-V driver validation log $(Get-Date -Format o)" -Encoding UTF8

try {
    Write-Step "WARN" "safety" "Use a disposable VM. TESTSIGNING is for test VMs only. No certificates, private keys, or passwords are stored by this script."
    $cred = Resolve-Credential
    $repo = if ($RepoPath) { (Resolve-Path -LiteralPath $RepoPath).Path } else { (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path }

    $vm = Assert-HyperVReady
    Write-Step "PASS" "hyperv_vm" "VM '$VMName' is running on host '$env:COMPUTERNAME'"
    if ($CreateCheckpoint) {
        $checkpointName = "SyncTranslate-driver-test-$timestamp"
        Checkpoint-VM -Name $VMName -SnapshotName $checkpointName -ErrorAction Stop
        Write-Step "PASS" "checkpoint" "created checkpoint $checkpointName"
    }

    $session = New-GuestSession -Cred $cred
    Write-Step "PASS" "powershell_direct" "connected with New-PSSession -VMName"

    Copy-RepoToGuest -Session $session -Source $repo -Destination $GuestWorkDir -IncludeRuntimeAssets ([bool]$IncludeRuntimes)

    Invoke-GuestCommand -Session $session -Name "guest_inventory" -ScriptBlock {
        "Windows: $((Get-CimInstance Win32_OperatingSystem).Caption) $((Get-CimInstance Win32_OperatingSystem).Version)"
        "User: $env:USERNAME"
        $principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
        "Admin: $($principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator))"
        $python = Get-Command python -ErrorAction SilentlyContinue
        if ($python) { "python: $((python --version) 2>&1)" } else { "python: missing" }
        $uv = Get-Command uv -ErrorAction SilentlyContinue
        if ($uv) { "uv: $((uv --version) 2>&1)" } else { "uv: missing" }
        $git = Get-Command git -ErrorAction SilentlyContinue
        if ($git) { "git: $((git --version) 2>&1)" } else { "git: missing" }
    } -ContinueOnError

    Invoke-GuestCommand -Session $session -Name "wdk_environment" -ArgumentList @($GuestWorkDir) -ScriptBlock {
        param($WorkDir)
        Set-Location $WorkDir
        powershell -ExecutionPolicy Bypass -File drivers/synctranslate_virtual_audio/scripts/check_wdk_environment.ps1
        if ($LASTEXITCODE -ne 0) { throw "check_wdk_environment.ps1 failed with exit code $LASTEXITCODE" }
    } -ContinueOnError

    if ($BuildInGuest -and !$SkipBuild) {
        Invoke-GuestCommand -Session $session -Name "fetch_sysvad" -ArgumentList @($GuestWorkDir) -ScriptBlock {
            param($WorkDir)
            Set-Location $WorkDir
            $sysvadSolution = Join-Path $WorkDir "downloads\driver_samples\Windows-driver-samples\audio\sysvad\sysvad.sln"
            if (Test-Path $sysvadSolution) {
                Write-Host "PASS SysVAD samples already present at $sysvadSolution; fetch_sysvad.ps1 skipped."
                return
            }
            powershell -ExecutionPolicy Bypass -File drivers/synctranslate_virtual_audio/scripts/fetch_sysvad.ps1
            if ($LASTEXITCODE -ne 0) { throw "fetch_sysvad.ps1 failed with exit code $LASTEXITCODE" }
        } -ContinueOnError
        Invoke-GuestCommand -Session $session -Name "build_driver_package" -ArgumentList @($GuestWorkDir) -ScriptBlock {
            param($WorkDir)
            Set-Location $WorkDir
            powershell -ExecutionPolicy Bypass -File drivers/synctranslate_virtual_audio/scripts/build_driver_package.ps1
            if ($LASTEXITCODE -ne 0) { throw "build_driver_package.ps1 failed with exit code $LASTEXITCODE" }
        } -ContinueOnError
    }
    elseif ($SkipBuild) {
        Write-Step "WARN" "build_driver_package" "skipped by -SkipBuild"
        $script:Summary["build_driver_package"] = "SKIPPED"
    }

    if ($EnableTestSigning) {
        Invoke-GuestCommand -Session $session -Name "enable_testsigning" -ScriptBlock {
            bcdedit /set testsigning on
            if ($LASTEXITCODE -ne 0) { throw "bcdedit /set testsigning on failed with exit code $LASTEXITCODE" }
        } -ContinueOnError
        if (!$NoReboot) {
            Invoke-GuestCommand -Session $session -Name "restart_vm" -ScriptBlock { Restart-Computer -Force } -ContinueOnError
            Remove-PSSession $session -ErrorAction SilentlyContinue
            $session = Wait-GuestReconnected -Cred $cred
        }
        else {
            Write-Step "WARN" "reboot" "NoReboot specified; Test Mode may not be active until the VM reboots."
        }
    }

    Invoke-GuestCommand -Session $session -Name "cleanup_existing_driver" -ArgumentList @($GuestWorkDir) -ScriptBlock {
        param($WorkDir)
        Set-Location $WorkDir
        if (Test-Path drivers/synctranslate_virtual_audio/scripts/uninstall_test_driver.ps1) {
            powershell -ExecutionPolicy Bypass -File drivers/synctranslate_virtual_audio/scripts/uninstall_test_driver.ps1 -AllowHostInstall
        }
    } -ContinueOnError

    Invoke-GuestCommand -Session $session -Name "install_test_driver" -ArgumentList @($GuestWorkDir, [bool]$SkipBuild) -ScriptBlock {
        param($WorkDir, $SkipBuildRequested)
        Set-Location $WorkDir
        $packageDir = Join-Path $WorkDir "artifacts/driver/synctranslate_virtual_audio/package"
        if ($SkipBuildRequested -and !(Test-Path $packageDir)) {
            Write-Host "WARN install skipped because -SkipBuild was specified and no driver package exists at $packageDir."
            return
        }
        powershell -ExecutionPolicy Bypass -File drivers/synctranslate_virtual_audio/scripts/install_test_driver.ps1 -AllowHostInstall
        if ($LASTEXITCODE -ne 0) { throw "install_test_driver.ps1 failed with exit code $LASTEXITCODE" }
    } -ContinueOnError

    Invoke-GuestCommand -Session $session -Name "verify_driver_format" -ArgumentList @($GuestWorkDir) -ScriptBlock {
        param($WorkDir)
        Set-Location $WorkDir
        powershell -ExecutionPolicy Bypass -File drivers/synctranslate_virtual_audio/scripts/verify_driver_format.ps1 -JsonOutput downloads/validation/driver_format.json
        if ($LASTEXITCODE -ne 0) { throw "verify_driver_format.ps1 failed with exit code $LASTEXITCODE" }
    } -ContinueOnError

    Invoke-GuestCommand -Session $session -Name "preflight_dialogue" -ArgumentList @($GuestWorkDir) -ScriptBlock {
        param($WorkDir)
        Set-Location $WorkDir
        if (Get-Command uv -ErrorAction SilentlyContinue) {
            uv run python tools/validation/preflight_release_check.py --mode dialogue --json downloads/validation/preflight_dialogue.json
        }
        else {
            python tools/validation/preflight_release_check.py --mode dialogue --json downloads/validation/preflight_dialogue.json
        }
        if ($LASTEXITCODE -ne 0) { throw "preflight_release_check.py --mode dialogue failed with exit code $LASTEXITCODE" }
    } -ContinueOnError

    Invoke-GuestCommand -Session $session -Name "windows_audio_runtime" -ArgumentList @($GuestWorkDir) -ScriptBlock {
        param($WorkDir)
        Set-Location $WorkDir
        if (Get-Command uv -ErrorAction SilentlyContinue) {
            uv run python tools/validation/validate_windows_audio_runtime.py --json downloads/validation/windows_audio_runtime.json
        }
        else {
            python tools/validation/validate_windows_audio_runtime.py --json downloads/validation/windows_audio_runtime.json
        }
        if ($LASTEXITCODE -ne 0) { throw "validate_windows_audio_runtime.py failed with exit code $LASTEXITCODE" }
    } -ContinueOnError

    Invoke-GuestCommand -Session $session -Name "dialogue_passthrough_smoke" -ArgumentList @($GuestWorkDir) -ScriptBlock {
        param($WorkDir)
        Set-Location $WorkDir
        if (Get-Command uv -ErrorAction SilentlyContinue) {
            uv run python tools/validation/audio_smoke_test.py dialogue-passthrough --duration 0.1
        }
        else {
            python tools/validation/audio_smoke_test.py dialogue-passthrough --duration 0.1
        }
        if ($LASTEXITCODE -ne 0) { throw "audio_smoke_test.py dialogue-passthrough failed with exit code $LASTEXITCODE" }
    } -ContinueOnError

    if ($CollectDiagnostics) {
        Invoke-GuestCommand -Session $session -Name "export_diagnostics_bundle" -ArgumentList @($GuestWorkDir) -ScriptBlock {
            param($WorkDir)
            Set-Location $WorkDir
            if (Get-Command uv -ErrorAction SilentlyContinue) {
                uv run python tools/validation/export_diagnostics_bundle.py
            }
            else {
                python tools/validation/export_diagnostics_bundle.py
            }
            if ($LASTEXITCODE -ne 0) { throw "export_diagnostics_bundle.py failed with exit code $LASTEXITCODE" }
        } -ContinueOnError
    }
    else {
        Write-Step "WARN" "export_diagnostics_bundle" "skipped; pass -CollectDiagnostics to export diagnostics bundle"
    }

    Copy-ValidationArtifacts -Session $session
    Remove-PSSession $session -ErrorAction SilentlyContinue

    $summaryPath = Join-Path $OutputDir "summary_$timestamp.json"
    $script:Summary["host_log"] = $script:HostLog
    $script:Summary["guest_log"] = $script:GuestLog
    $script:Summary["output_dir"] = (Resolve-Path $OutputDir).Path
    $script:Summary | ConvertTo-Json -Depth 5 | Set-Content -Path $summaryPath -Encoding UTF8

    Write-Host ""
    Write-Host "Summary:"
    foreach ($key in $script:Summary.Keys) {
        Write-Host "  $key = $($script:Summary[$key])"
    }
    Write-Host "  summary_json = $summaryPath"
    exit 0
}
catch {
    Write-Step "FAIL" "driver_vm_validation" $_.Exception.Message
    Write-Host "Host log: $script:HostLog"
    Write-Host "Guest log: $script:GuestLog"
    exit 1
}
