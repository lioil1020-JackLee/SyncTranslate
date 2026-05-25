param(
    [string]$PackageDir = "artifacts/driver/synctranslate_virtual_audio/package",
    [string]$MsiPath = "artifacts/driver/synctranslate_virtual_audio/SyncTranslateVirtualAudioDriver.msi",
    [string]$CertificatePath = "artifacts/driver/synctranslate_virtual_audio/SyncTranslateVirtualAudioTest.cer",
    [switch]$Json,
    [switch]$AllowExistingSyncTranslateDevices,
    [switch]$AllowKernelBridgeEnabled,
    [switch]$AllowMsiInstallCustomAction,
    [switch]$AllowSecureBoot
)

$ErrorActionPreference = "Stop"

$checks = New-Object System.Collections.Generic.List[object]
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptRoot "../../..")

function Add-Check {
    param(
        [string]$Name,
        [bool]$Passed,
        [string]$Message,
        [string]$Severity = "error",
        [object]$Details = $null
    )
    $script:checks.Add([pscustomobject]@{
        name = $Name
        passed = $Passed
        severity = $Severity
        message = $Message
        details = $Details
    }) | Out-Null
}

function Resolve-OptionalPath {
    param([string]$Path)
    if ($Path -and (Test-Path $Path)) {
        return (Resolve-Path $Path).Path
    }
    return ""
}

function Test-ScriptParse {
    param([string]$Path)
    try {
        [scriptblock]::Create((Get-Content -Raw -Path $Path)) | Out-Null
        return $true
    }
    catch {
        Add-Check "script_parse:$Path" $false $_.Exception.Message
        return $false
    }
}

function Get-MsiRows {
    param(
        [string]$DatabasePath,
        [string]$Sql
    )
    $installer = New-Object -ComObject WindowsInstaller.Installer
    $database = $installer.GetType().InvokeMember("OpenDatabase", "InvokeMethod", $null, $installer, @($DatabasePath, 0))
    $view = $database.GetType().InvokeMember("OpenView", "InvokeMethod", $null, $database, @($Sql))
    $view.GetType().InvokeMember("Execute", "InvokeMethod", $null, $view, $null) | Out-Null
    $rows = @()
    while ($true) {
        $record = $view.GetType().InvokeMember("Fetch", "InvokeMethod", $null, $view, $null)
        if ($null -eq $record) {
            break
        }
        $fieldCount = $record.GetType().InvokeMember("FieldCount", "GetProperty", $null, $record, $null)
        $values = @()
        for ($i = 1; $i -le $fieldCount; $i++) {
            $values += $record.GetType().InvokeMember("StringData", "GetProperty", $null, $record, @($i))
        }
        $rows += ,$values
    }
    return $rows
}

function Find-Tool {
    param(
        [string]$Name,
        [string[]]$Roots
    )
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }
    foreach ($root in $Roots) {
        if (!(Test-Path $root)) {
            continue
        }
        $match = Get-ChildItem -Path $root -Filter $Name -Recurse -File -ErrorAction SilentlyContinue |
            Sort-Object @{ Expression = { if ($_.FullName -match "\\x64\\") { 0 } elseif ($_.FullName -match "\\x86\\") { 1 } else { 2 } } }, FullName |
            Select-Object -First 1
        if ($match) {
            return $match.FullName
        }
    }
    return ""
}

$packageDirResolved = Resolve-OptionalPath $PackageDir
$msiResolved = Resolve-OptionalPath $MsiPath
$certificateResolved = Resolve-OptionalPath $CertificatePath

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
Add-Check "administrator" ($principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) "Driver preflight should run from an elevated PowerShell."

Add-Check "package_exists" ([bool]$packageDirResolved) "Package directory: $PackageDir"
Add-Check "msi_exists" ([bool]$msiResolved) "MSI: $MsiPath" "warning"
Add-Check "certificate_exists" ([bool]$certificateResolved) "Certificate: $CertificatePath" "warning"

if ($packageDirResolved) {
    $infFiles = @(Get-ChildItem -Path $packageDirResolved -Filter "*.inf" -File)
    $catFiles = @(Get-ChildItem -Path $packageDirResolved -Filter "*.cat" -File)
    $sysFiles = @(Get-ChildItem -Path $packageDirResolved -Filter "*.sys" -File -Recurse)
    Add-Check "package_has_inf" ($infFiles.Count -gt 0) "INF files: $($infFiles.Count)" "error" $infFiles.FullName
    Add-Check "package_has_cat" ($catFiles.Count -gt 0) "CAT files: $($catFiles.Count)" "error" $catFiles.FullName
    Add-Check "package_has_sys" ($sysFiles.Count -eq 1) "SYS files: $($sysFiles.Count)" "error" $sysFiles.FullName

    if ($infFiles.Count -gt 0) {
        $infText = ($infFiles | ForEach-Object { Get-Content -Raw -Path $_.FullName }) -join "`n"
        foreach ($marker in @("Root\SyncTranslateVirtualAudio", "SyncTranslate Virtual Audio", "SyncTranslate Virtual Speaker", "SyncTranslate Virtual Microphone")) {
            Add-Check "inf_marker:$marker" ($infText -like "*$marker*") "Required INF marker: $marker"
        }
    }

    $kitsRoots = @("${env:ProgramFiles(x86)}\Windows Kits\10", "${env:ProgramFiles}\Windows Kits\10")
    $signtool = Find-Tool "signtool.exe" $kitsRoots
    if ($signtool -and $catFiles.Count -gt 0) {
        foreach ($cat in $catFiles) {
            $previousErrorActionPreference = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            try {
                $verifyOutput = & $signtool verify /pa /v $cat.FullName 2>&1
                $verifyExitCode = $LASTEXITCODE
            }
            finally {
                $ErrorActionPreference = $previousErrorActionPreference
            }
            Add-Check "cat_signature:$($cat.Name)" ($verifyExitCode -eq 0) "signtool verify /pa $($cat.Name)" "error" (($verifyOutput | Out-String).Trim())
        }
    }
    else {
        Add-Check "signtool_available" $false "signtool.exe not found; CAT signature cannot be verified." "warning"
    }

    $infverif = Find-Tool "infverif.exe" $kitsRoots
    if ($infverif -and $infFiles.Count -gt 0) {
        foreach ($inf in $infFiles) {
            $previousErrorActionPreference = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            try {
                $infOutput = & $infverif /w $inf.FullName 2>&1
                $infExitCode = $LASTEXITCODE
            }
            finally {
                $ErrorActionPreference = $previousErrorActionPreference
            }
            Add-Check "infverif:$($inf.Name)" ($infExitCode -eq 0) "infverif /w $($inf.Name)" "error" (($infOutput | Out-String).Trim())
        }
    }
    else {
        Add-Check "infverif_available" $false "infverif.exe not found; INF structural validation was skipped." "warning"
    }
}

foreach ($scriptPath in @(
    "drivers/synctranslate_virtual_audio/scripts/install_driver_package.ps1",
    "drivers/synctranslate_virtual_audio/scripts/uninstall_driver_package.ps1",
    "drivers/synctranslate_virtual_audio/scripts/verify_driver_install.ps1"
)) {
    $resolvedScriptPath = Join-Path $repoRoot $scriptPath
    if (Test-Path $resolvedScriptPath) {
        $ok = Test-ScriptParse $resolvedScriptPath
        if ($ok) {
            Add-Check "script_parse:$scriptPath" $true "PowerShell script parses."
        }
    }
    else {
        Add-Check "script_exists:$scriptPath" $false "Missing script: $scriptPath"
    }
}

$bridgeSource = "drivers/synctranslate_virtual_audio/overlay/EndpointsCommon/minwavertstream.cpp"
$bridgeSourceResolved = Join-Path $repoRoot $bridgeSource
if (Test-Path $bridgeSourceResolved) {
    $sourceText = Get-Content -Raw -Path $bridgeSourceResolved
    $kernelBridgeEnabled = [bool]($sourceText -match "#define\s+SYNC_BRIDGE_KERNEL_ENABLED\s+1")
    $kernelTtsInjectionEnabled = [bool]($sourceText -match "#define\s+SYNC_TTS_INJECTION_ENABLED\s+1")
    Add-Check "kernel_bridge_disabled" (!$kernelBridgeEnabled -or $AllowKernelBridgeEnabled) "Kernel bridge must stay disabled until VM/driver-verifier testing passes." "error"
    Add-Check "kernel_tts_injection_disabled" (!$kernelTtsInjectionEnabled) "Kernel TTS injection must stay disabled; use user-mode audio routing until the replacement path passes VM testing." "error"
    if (!$kernelBridgeEnabled) {
        $unguardedBridgeCalls = @()
        $readIntoIndex = $sourceText.IndexOf("(void)SyncBridgeReadIntoDma")
        if ($readIntoIndex -ge 0) {
            $before = $sourceText.Substring([Math]::Max(0, $readIntoIndex - 80), [Math]::Min(80, $readIntoIndex))
            if ($before -notmatch "#if\s+SYNC_BRIDGE_KERNEL_ENABLED") {
                $unguardedBridgeCalls += "SyncBridgeReadIntoDma"
            }
        }
        $writeFromIndex = $sourceText.IndexOf("(void)SyncBridgeWriteFromDma")
        if ($writeFromIndex -ge 0) {
            $before = $sourceText.Substring([Math]::Max(0, $writeFromIndex - 80), [Math]::Min(80, $writeFromIndex))
            if ($before -notmatch "#if\s+SYNC_BRIDGE_KERNEL_ENABLED") {
                $unguardedBridgeCalls += "SyncBridgeWriteFromDma"
            }
        }
        Add-Check "kernel_bridge_dpc_calls_guarded" ($unguardedBridgeCalls.Count -eq 0) "DPC/audio timer path must not call SyncBridge functions while the kernel bridge is disabled." "error" $unguardedBridgeCalls
    }
}
else {
    Add-Check "kernel_bridge_source_exists" $false "Missing source guard file: $bridgeSource" "warning"
}

if ($msiResolved) {
    try {
        $customActions = @(Get-MsiRows $msiResolved "SELECT Action, Type, Source, Target FROM CustomAction")
        $installActions = @($customActions | Where-Object {
            $_[0] -eq "InstallDriverPackage" -or
            (($_[3] -match "install_driver_package\.ps1") -and ($_[3] -notmatch "uninstall_driver_package\.ps1"))
        })
        Add-Check "msi_has_install_custom_action" ($installActions.Count -gt 0) "Driver MSI must provide a one-click elevated install custom action." "error" $installActions

        $sequenceRows = @(Get-MsiRows $msiResolved "SELECT Action, Condition, Sequence FROM InstallExecuteSequence")
        $rebootRows = @($sequenceRows | Where-Object { $_[0] -eq "ScheduleReboot" -or $_[0] -eq "ForceReboot" })
        Add-Check "msi_no_reboot_action" ($rebootRows.Count -eq 0) "MSI must not schedule a reboot." "error" $rebootRows
    }
    catch {
        Add-Check "msi_table_scan" $false "Unable to inspect MSI tables: $($_.Exception.Message)"
    }
}

try {
    $secureBootValue = $null
    try {
        $secureBootState = Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\SecureBoot\State" -ErrorAction Stop
        $secureBootValue = $secureBootState.UEFISecureBootEnabled
    }
    catch {
        $secureBootValue = "unknown"
    }
    Add-Check "secure_boot_allows_test_mode" (($secureBootValue -ne 1) -or $AllowSecureBoot) "UEFISecureBootEnabled: $secureBootValue" "error"

    $bcd = & "$env:windir\System32\bcdedit.exe" /enum 2>&1
    $testSigningEnabled = [bool]($bcd | Where-Object { $_ -match "testsigning\s+Yes|testsigning\s+on" })
    Add-Check "test_signing_enabled" $testSigningEnabled "Windows Test Mode must be enabled before loading a test-signed driver." "error" (($bcd | Out-String).Trim())
}
catch {
    Add-Check "boot_policy_check" $false "Unable to check boot policy: $($_.Exception.Message)" "warning"
}

try {
    $devices = @(pnputil /enum-devices /class MEDIA 2>&1)
    $syncDeviceBlocks = @()
    $currentBlock = New-Object System.Collections.Generic.List[string]
    foreach ($line in $devices) {
        if ($line -match "^\s*Instance ID:") {
            if ($currentBlock.Count -gt 0) {
                $block = ($currentBlock -join "`n")
                if ($block -match "SyncTranslate") {
                    $syncDeviceBlocks += $block
                }
                $currentBlock.Clear()
            }
        }
        $currentBlock.Add($line) | Out-Null
    }
    if ($currentBlock.Count -gt 0) {
        $block = ($currentBlock -join "`n")
        if ($block -match "SyncTranslate") {
            $syncDeviceBlocks += $block
        }
    }
    $problemDevices = @($syncDeviceBlocks | Where-Object { $_ -match "Status:\s+Problem|Problem Code:" })
    Add-Check "no_existing_sync_translate_problem_devices" (($problemDevices.Count -eq 0) -or $AllowExistingSyncTranslateDevices) "Existing SyncTranslate problem devices: $($problemDevices.Count)" "error" $problemDevices
    Add-Check "no_duplicate_sync_translate_devices" (($syncDeviceBlocks.Count -le 1) -or $AllowExistingSyncTranslateDevices) "Existing SyncTranslate media devices: $($syncDeviceBlocks.Count)" "error" $syncDeviceBlocks
}
catch {
    Add-Check "existing_device_scan" $false "Unable to scan existing media devices: $($_.Exception.Message)" "warning"
}

try {
    $drivers = @(pnputil /enum-drivers 2>&1)
    $syncDrivers = @()
    $currentBlock = New-Object System.Collections.Generic.List[string]
    foreach ($line in $drivers) {
        if ($line -match "^\s*Published Name:") {
            if ($currentBlock.Count -gt 0) {
                $block = ($currentBlock -join "`n")
                if ($block -match "Provider Name:\s+SyncTranslate") {
                    $syncDrivers += $block
                }
                $currentBlock.Clear()
            }
        }
        $currentBlock.Add($line) | Out-Null
    }
    if ($currentBlock.Count -gt 0) {
        $block = ($currentBlock -join "`n")
        if ($block -match "Provider Name:\s+SyncTranslate") {
            $syncDrivers += $block
        }
    }
    Add-Check "no_duplicate_sync_translate_driver_packages" (($syncDrivers.Count -le 1) -or $AllowExistingSyncTranslateDevices) "Existing SyncTranslate driver packages: $($syncDrivers.Count)" "error" $syncDrivers
}
catch {
    Add-Check "existing_driver_scan" $false "Unable to scan existing driver packages: $($_.Exception.Message)" "warning"
}

$failed = @($checks | Where-Object { !$_.passed -and $_.severity -eq "error" })
$warnings = @($checks | Where-Object { !$_.passed -and $_.severity -eq "warning" })
$result = [pscustomobject]@{
    passed = ($failed.Count -eq 0)
    failed_count = $failed.Count
    warning_count = $warnings.Count
    checks = @($checks.ToArray())
}

if ($Json) {
    $result | ConvertTo-Json -Depth 8
}
else {
    foreach ($check in $checks) {
        $prefix = if ($check.passed) { "[PASS]" } elseif ($check.severity -eq "warning") { "[WARN]" } else { "[FAIL]" }
        Write-Host "$prefix $($check.name): $($check.message)"
    }
    if (!$result.passed) {
        Write-Host ""
        Write-Host "[driver-preflight] FAILED: $($failed.Count) blocking issue(s)."
    }
    else {
        Write-Host ""
        Write-Host "[driver-preflight] PASSED with $($warnings.Count) warning(s)."
    }
}

if (!$result.passed) {
    exit 1
}
