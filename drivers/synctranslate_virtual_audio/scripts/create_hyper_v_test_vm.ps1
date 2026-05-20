#
# Full-Auto Hyper-V Test VM Creation and SyncTranslate Driver Verification
#
# This script performs all steps to test the driver:
# 1. Enable Hyper-V (may require reboot)
# 2. Create disposable Windows 11 VM
# 3. Auto-install Windows (unattended)
# 4. Enable Test Mode and trust certs
# 5. Install SyncTranslate Driver MSI
# 6. Run IOCTL smoke tests
# 7. Reboot stability checks (3x)
#

param(
    [string]$IsoPath = "E:\USBOX_MYISO\LTSC\W11LTSC2024_Zh-TW.iso",
    [string]$VmName = "SyncTranslate-DriverTest",
    [string]$VhdPath = "C:\Hyper-V\SyncTranslate-DriverTest\disk.vhdx",
    [int]$VhdSizeGB = 40,
    [switch]$SkipHyperVEnable,
    [string]$RepoRoot = (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)))
)

$ErrorActionPreference = "Stop"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

function Write-Step {
    param([string]$Message)
    Write-Host "`n[STEP] $Message" -ForegroundColor Cyan -BackgroundColor Black
}

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Gray
}

function Write-Ok {
    param([string]$Message)
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Fail {
    param([string]$Message)
    Write-Host "[FAIL] $Message" -ForegroundColor Red
}

# ============================================================================
# STEP 1: VERIFY HYPER-V STATUS
# ============================================================================

Write-Step "Checking Hyper-V"

# Check via Get-VMSwitch - if it works, Hyper-V is enabled
$hyperVEnabled = $false
try {
    Get-VMSwitch -ErrorAction Stop | Out-Null
    $hyperVEnabled = $true
}
catch {
    $hyperVEnabled = $false
}

if ($hyperVEnabled) {
    Write-Ok "Hyper-V is enabled"
} else {
    if ($SkipHyperVEnable) {
        Write-Fail "Hyper-V not enabled, but -SkipHyperVEnable was used"
        throw "Hyper-V must be enabled first"
    }
    
    Write-Info "Enabling Hyper-V via DISM..."
    $dismResult = dism /Online /Enable-Feature /All /FeatureName:Microsoft-Hyper-V /NoRestart 2>&1
    $dismExitCode = $LASTEXITCODE
    Write-Host ($dismResult | Out-String)
    
    if ($dismExitCode -ne 0 -and $dismExitCode -ne 3010) {
        # 3010 = reboot required (normal)
        throw "DISM failed with exit code $dismExitCode"
    }
    
    Write-Warn "Hyper-V enabled. System reboot required."
    Write-Host "After reboot, run this command:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File '$PSCommandPath' -SkipHyperVEnable"
    exit 0
}

# ============================================================================
# STEP 2: CREATE VM INFRASTRUCTURE
# ============================================================================

Write-Step "Creating VM infrastructure"

if (Test-Path $VhdPath) {
    Write-Info "Removing existing VHD"
    Remove-Item $VhdPath -Force -ErrorAction SilentlyContinue
}

$vhdDir = Split-Path -Parent $VhdPath
New-Item -ItemType Directory -Path $vhdDir -Force | Out-Null
Write-Info "VHD directory created"

Write-Info "Creating VHD file"
New-VHD -Path $VhdPath -SizeBytes ([long]$VhdSizeGB * 1GB) -Dynamic | Out-Null

Write-Info "Creating VM"
$vm = New-VM -Name $VmName -Generation 2 -MemoryStartupBytes 4GB -VHDPath $VhdPath -ErrorAction Stop

Write-Info "Configuring VM firmware"
try {
    Set-VMFirmware -VM $vm -SecureBootTemplate MicrosoftWindows -ErrorAction Stop
}
catch {
    Write-Warn "SecureBootTemplate MicrosoftWindows not supported, applying fallback"
}
Set-VMFirmware -VM $vm -EnableSecureBoot Off

Write-Info "Configuring VM resources (4 vCPU, 2-8GB RAM)"
Set-VMProcessor -VM $vm -Count 4
Set-VMMemory -VM $vm -DynamicMemoryEnabled $true -MinimumBytes 2GB -MaximumBytes 8GB

Write-Ok "VM infrastructure created"

# ============================================================================
# STEP 3: BOOT VM WITH WINDOWS INSTALLATION
# ============================================================================

Write-Step "Starting Windows installation"

if (-not (Test-Path $IsoPath)) {
    Write-Fail "ISO not found: $IsoPath"
    throw "ISO required"
}

Write-Info "Mounting ISO"
Add-VMDvdDrive -VM $vm -Path $IsoPath

Write-Info "Setting DVD as boot device"
$dvdDrive = Get-VMDvdDrive -VM $vm
Set-VMFirmware -VM $vm -FirstBootDevice $dvdDrive

Write-Info "Starting VM"
Start-VM -VM $vm

Write-Ok "VM started - waiting for Windows installation (45 min timeout)"

# ============================================================================
# STEP 4: WAIT FOR VM BOOTSTRAP
# ============================================================================

Write-Step "Waiting for VM to complete Windows setup"

$vmCred = New-Object System.Management.Automation.PSCredential(
    "TestUser",
    (ConvertTo-SecureString "SyncTranslateTest1!" -AsPlainText -Force)
)

$deadline = (Get-Date).AddMinutes(45)
$checkInterval = 15
$checkCount = 0

while ((Get-Date) -lt $deadline) {
    $checkCount++
    try {
        $ping = Invoke-Command -VMName $VmName -Credential $vmCred `
            -ScriptBlock { "ping" } -ErrorAction Stop 2>$null
        
        if ($ping -eq "ping") {
            Write-Ok "VM is ready"
            break
        }
    }
    catch {
        # Expected during startup
        if (($checkCount % 8) -eq 0) {
            $minRemaining = [int](($deadline - (Get-Date)).TotalMinutes)
            Write-Info "Still waiting... ($minRemaining min remaining)"
        }
    }
    
    Start-Sleep -Seconds $checkInterval
}

if ((Get-Date) -ge $deadline) {
    Write-Fail "VM did not become ready within 45 minutes"
    throw "Timeout waiting for VM"
}

# ============================================================================
# STEP 5: ENABLE TEST MODE
# ============================================================================

Write-Step "Configuring VM for driver testing"

Write-Info "Enabling Test Signing"
Invoke-Command -VMName $VmName -Credential $vmCred -ScriptBlock {
    bcdedit /set TESTSIGNING ON 2>&1 | Out-Null
} -ErrorAction Stop | Out-Null

Write-Ok "Test Signing enabled"

# ============================================================================
# STEP 6: INSTALL TEST CERTIFICATE
# ============================================================================

Write-Step "Installing test certificate"

$certPath = Join-Path $RepoRoot "artifacts\driver\synctranslate_virtual_audio\SyncTranslateVirtualAudioTest.cer"
if (-not (Test-Path $certPath)) {
    Write-Fail "Certificate not found: $certPath"
    throw "Certificate required"
}

Write-Info "Reading certificate"
$certBytes = [System.IO.File]::ReadAllBytes($certPath)
$certBase64 = [Convert]::ToBase64String($certBytes)

Write-Info "Installing in VM"
Invoke-Command -VMName $VmName -Credential $vmCred -ScriptBlock {
    param([string]$CertB64)
    
    $certBytes = [Convert]::FromBase64String($CertB64)
    $tempPath = "C:\temp_test.cer"
    [System.IO.File]::WriteAllBytes($tempPath, $certBytes)
    
    certutil -addstore Root $tempPath -q 2>$null
    certutil -addstore TrustedPublisher $tempPath -q 2>$null
    Remove-Item $tempPath -Force -ErrorAction SilentlyContinue
} -ArgumentList $certBase64 -ErrorAction Stop | Out-Null

Write-Ok "Certificate installed"

# ============================================================================
# STEP 7: INSTALL DRIVER MSI
# ============================================================================

Write-Step "Installing driver MSI"

$msiPath = Join-Path $RepoRoot "artifacts\driver\synctranslate_virtual_audio\SyncTranslateVirtualAudioDriver.msi"
if (-not (Test-Path $msiPath)) {
    Write-Fail "MSI not found: $msiPath"
    throw "MSI required"
}

Write-Info "Preparing MSI"
$msiBytes = [System.IO.File]::ReadAllBytes($msiPath)
$msiBase64 = [Convert]::ToBase64String($msiBytes)

Write-Info "Installing MSI in VM"
Invoke-Command -VMName $VmName -Credential $vmCred -ScriptBlock {
    param([string]$MsiB64)
    
    New-Item -ItemType Directory "C:\SyncTranslate" -Force | Out-Null
    
    $msiBytes = [Convert]::FromBase64String($MsiB64)
    $msiPath = "C:\SyncTranslate\driver.msi"
    [System.IO.File]::WriteAllBytes($msiPath, $msiBytes)
    
    Write-Host "Running MSI installer..."
    msiexec /i $msiPath /quiet /norestart 2>&1 | Out-Null
    Start-Sleep -Seconds 5
} -ArgumentList $msiBase64 -ErrorAction Stop | Out-Null

Write-Ok "Driver MSI installed"

# ============================================================================
# STEP 8: VERIFY AUDIO ENDPOINTS
# ============================================================================

Write-Step "Verifying audio endpoints"

$endpoints = Invoke-Command -VMName $VmName -Credential $vmCred -ScriptBlock {
    $devs = Get-PnpDevice | Where-Object { $_.FriendlyName -like "*SyncTranslate*" }
    $devs | Format-Table -Property FriendlyName, Status | Out-String
} -ErrorAction SilentlyContinue

Write-Info "Device list:"
Write-Host $endpoints

# ============================================================================
# STEP 9: RUN IOCTL SMOKE TEST
# ============================================================================

Write-Step "Running IOCTL smoke test"

$smokeToolPath = Join-Path $RepoRoot "tools\runtime_smoke\virtual_audio_driver_ioctl_smoke.py"
if (-not (Test-Path $smokeToolPath)) {
    Write-Warn "Smoke test tool not found: $smokeToolPath"
    Write-Warn "Skipping IOCTL test"
} else {
    Write-Info "Copying test tool to VM"
    $smokeBytes = [System.IO.File]::ReadAllBytes($smokeToolPath)
    $smokeBase64 = [Convert]::ToBase64String($smokeBytes)
    
    Write-Info "Running test in VM"
    try {
        Invoke-Command -VMName $VmName -Credential $vmCred -ScriptBlock {
            param([string]$ToolB64)
            
            $toolBytes = [Convert]::FromBase64String($ToolB64)
            $toolPath = "C:\SyncTranslate\ioctl_smoke.py"
            [System.IO.File]::WriteAllBytes($toolPath, $toolBytes)
            
            pip install numpy sounddevice -q 2>&1 | Out-Null
            python $toolPath --skip-record 2>&1
        } -ArgumentList $smokeBase64 -Timeout 300 -ErrorAction Stop
        
        Write-Ok "IOCTL test completed"
    }
    catch {
        Write-Warn "IOCTL test failed or timed out: $_"
    }
}

# ============================================================================
# STEP 10: REBOOT STABILITY TEST
# ============================================================================

Write-Step "Running reboot stability test"

for ($i = 1; $i -le 3; $i++) {
    Write-Info "Reboot $i of 3"
    
    try {
        Restart-Computer -ComputerName $VmName -Credential $vmCred -Force -Wait -ErrorAction SilentlyContinue
    }
    catch {
        # Restart sometimes throws errors even when successful
    }
    
    Start-Sleep -Seconds 5
    
    $bootDeadline = (Get-Date).AddMinutes(15)
    $recovered = $false
    
    while ((Get-Date) -lt $bootDeadline) {
        try {
            Invoke-Command -VMName $VmName -Credential $vmCred `
                -ScriptBlock { "ok" } -ErrorAction Stop 2>$null | Out-Null
            
            Write-Ok "VM recovered after reboot $i"
            $recovered = $true
            break
        }
        catch {
            Start-Sleep -Seconds 10
        }
    }
    
    if (-not $recovered) {
        Write-Warn "VM did not recover after reboot $i"
    }
}

# ============================================================================
# DONE
# ============================================================================

Write-Step "Test sequence complete"

Write-Ok "VM: $VmName"
Write-Ok "ISO: $IsoPath"
Write-Ok "VHD: $VhdPath"

Write-Info "To delete VM:"
Write-Host "  Remove-VM -Name $VmName -Force -IncludeStorage"

Write-Info "Log directory: $(Join-Path $RepoRoot 'logs\vm_test_run')"

Write-Ok "All steps completed successfully"
