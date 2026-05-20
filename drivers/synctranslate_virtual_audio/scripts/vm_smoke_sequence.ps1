<#
.SYNOPSIS
    SyncTranslate Virtual Audio Driver VM 安裝驗證完整流程。

.DESCRIPTION
    只在 disposable Windows VM 中執行本 script。
    不要在開發主機執行。

    本 script 執行以下步驟：
      Phase 1 - Preflight        （環境 gate，不改變任何狀態）
      Phase 2 - Driver Install   （安裝 MSI 或 driver package）
      Phase 3 - Endpoint Verify  （驗證 SyncTranslate 裝置出現）
      Phase 4 - IOCTL Smoke      （Python smoke test）
      Phase 5 - Reboot Gate      （提示重啟，重啟後重跑 Phase 3+4 直到 3 次）
      Phase 6 - Driver Verifier  （啟用 Driver Verifier + 再次 smoke）
      Phase 7 - Uninstall Gate   （解除安裝，驗證裝置消失）

    每個 Phase 完成後寫 JSON 結果到 LogDir。

    使用方式（VM 內 elevated PowerShell）：

        cd <repo_root>
        powershell -ExecutionPolicy Bypass -File `
            drivers/synctranslate_virtual_audio/scripts/vm_smoke_sequence.ps1 `
            -Phase all

        # 只跑某個 phase（例如 install 掛掉重試）
        powershell -ExecutionPolicy Bypass -File `
            drivers/synctranslate_virtual_audio/scripts/vm_smoke_sequence.ps1 `
            -Phase install

.PARAMETER Phase
    all | preflight | install | verify | ioctl | verifier | uninstall

.PARAMETER MsiPath
    MSI 路徑（預設 artifacts/driver/synctranslate_virtual_audio/SyncTranslateVirtualAudioDriver.msi）

.PARAMETER PackageDir
    driver package 目錄（預設 artifacts/driver/synctranslate_virtual_audio/package）

.PARAMETER CertificatePath
    測試憑證路徑（預設 artifacts/driver/synctranslate_virtual_audio/SyncTranslateVirtualAudioTest.cer）

.PARAMETER LogDir
    結果 JSON 輸出目錄（預設 logs/vm_smoke_sequence）

.PARAMETER PythonExe
    Python 執行檔路徑（預設 python）
#>

param(
    [ValidateSet("all", "preflight", "install", "verify", "ioctl", "verifier", "uninstall")]
    [string]$Phase = "all",
    [string]$MsiPath = "artifacts/driver/synctranslate_virtual_audio/SyncTranslateVirtualAudioDriver.msi",
    [string]$PackageDir = "artifacts/driver/synctranslate_virtual_audio/package",
    [string]$CertificatePath = "artifacts/driver/synctranslate_virtual_audio/SyncTranslateVirtualAudioTest.cer",
    [string]$LogDir = "logs/vm_smoke_sequence",
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "../../..")

function Write-Step {
    param([string]$Msg)
    Write-Host ""
    Write-Host "=========================================="
    Write-Host $Msg
    Write-Host "=========================================="
}

function Assert-Elevated {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $p = New-Object Security.Principal.WindowsPrincipal($id)
    if (!$p.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw "elevated PowerShell required."
    }
}

function Save-PhaseResult {
    param([string]$PhaseName, [hashtable]$Data)
    $outDir = Join-Path $repoRoot $LogDir
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null
    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $outPath = Join-Path $outDir "${PhaseName}_${ts}.json"
    $Data | ConvertTo-Json -Depth 8 | Set-Content -Path $outPath -Encoding UTF8
    Write-Host "[seq] 結果已寫入：$outPath"
    return $outPath
}

function Assert-PhaseOk {
    param([hashtable]$Result, [string]$PhaseName)
    if (!$Result.ok) {
        $reason = $Result.fail_reason
        throw "Phase '$PhaseName' FAILED. reason=$reason"
    }
}

# ---------------------------------------------------------------------------
# Phase 1 - Preflight
# ---------------------------------------------------------------------------

function Invoke-PhasePreflight {
    Write-Step "[Phase 1/7] Preflight"
    $preflightScript = Join-Path $scriptDir "preflight_driver_install.ps1"
    $output = & $preflightScript `
        -PackageDir (Join-Path $repoRoot $PackageDir) `
        -MsiPath (Join-Path $repoRoot $MsiPath) `
        -CertificatePath (Join-Path $repoRoot $CertificatePath) `
        -Json 2>&1
    $exitCode = $LASTEXITCODE
    $output | ForEach-Object { Write-Host $_ }

    $result = @{
        ok = ($exitCode -eq 0)
        exit_code = $exitCode
        fail_reason = if ($exitCode -ne 0) { "preflight_blocking_checks" } else { $null }
    }
    Save-PhaseResult -PhaseName "phase1_preflight" -Data $result | Out-Null
    Assert-PhaseOk -Result $result -PhaseName "preflight"
    Write-Host "[Phase 1] PASS"
}

# ---------------------------------------------------------------------------
# Phase 2 - Driver Install
# ---------------------------------------------------------------------------

function Invoke-PhaseInstall {
    Write-Step "[Phase 2/7] Driver Install"
    $installScript = Join-Path $scriptDir "install_driver_package.ps1"
    $output = & $installScript `
        -MsiPath (Join-Path $repoRoot $MsiPath) `
        -PackageDir (Join-Path $repoRoot $PackageDir) `
        -CertificatePath (Join-Path $repoRoot $CertificatePath) `
        -AllowHostInstall `
        -LogPath (Join-Path $repoRoot "$LogDir/driver-install.log") 2>&1
    $exitCode = $LASTEXITCODE
    $output | ForEach-Object { Write-Host $_ }

    $result = @{
        ok = ($exitCode -eq 0)
        exit_code = $exitCode
        fail_reason = if ($exitCode -ne 0) { "install_script_nonzero_exit" } else { $null }
    }
    Save-PhaseResult -PhaseName "phase2_install" -Data $result | Out-Null
    Assert-PhaseOk -Result $result -PhaseName "install"
    Write-Host "[Phase 2] PASS"
}

# ---------------------------------------------------------------------------
# Phase 3 - Endpoint Verify
# ---------------------------------------------------------------------------

function Invoke-PhaseVerify {
    param([int]$RebootCount = 0)
    Write-Step "[Phase 3/7] Endpoint Verify (after $RebootCount reboot(s))"
    $verifyScript = Join-Path $scriptDir "verify_driver_install.ps1"
    $output = & $verifyScript -Json 2>&1
    $exitCode = $LASTEXITCODE
    $output | ForEach-Object { Write-Host $_ }

    try {
        $jsonStr = ($output | Where-Object { $_ -match "^\s*\{" } | Out-String).Trim()
        if (!$jsonStr) { $jsonStr = ($output | Out-String).Trim() }
        $parsed = $jsonStr | ConvertFrom-Json
    }
    catch {
        $parsed = $null
    }

    $ok = $exitCode -eq 0
    $result = @{
        ok = $ok
        reboot_count = $RebootCount
        exit_code = $exitCode
        verify_output = ($output | Out-String).Trim()
        fail_reason = if (!$ok) { "endpoint_not_found_or_error" } else { $null }
        has_virtual_speaker = if ($parsed) { [bool]$parsed.has_virtual_speaker_endpoint } else { $null }
        has_virtual_microphone = if ($parsed) { [bool]$parsed.has_virtual_microphone_endpoint } else { $null }
    }
    Save-PhaseResult -PhaseName "phase3_verify_r${RebootCount}" -Data $result | Out-Null
    Assert-PhaseOk -Result $result -PhaseName "verify"
    Write-Host "[Phase 3] PASS  speaker=$($result.has_virtual_speaker)  microphone=$($result.has_virtual_microphone)"
}

# ---------------------------------------------------------------------------
# Phase 4 - IOCTL Smoke
# ---------------------------------------------------------------------------

function Invoke-PhaseIoctlSmoke {
    param([int]$RebootCount = 0)
    Write-Step "[Phase 4/7] IOCTL Smoke Test (after $RebootCount reboot(s))"
    $smokePy = Join-Path $repoRoot "tools/runtime_smoke/virtual_audio_driver_ioctl_smoke.py"
    $jsonOut = Join-Path $repoRoot "$LogDir/phase4_ioctl_smoke_r${RebootCount}.json"
    $wavOut = Join-Path $repoRoot "$LogDir/phase4_capture_r${RebootCount}.wav"

    $output = & $PythonExe $smokePy `
        --json-output $jsonOut `
        --wav-output $wavOut 2>&1
    $exitCode = $LASTEXITCODE
    $output | ForEach-Object { Write-Host $_ }

    $result = @{
        ok = ($exitCode -eq 0)
        reboot_count = $RebootCount
        exit_code = $exitCode
        json_output = $jsonOut
        wav_output = $wavOut
        fail_reason = if ($exitCode -ne 0) { "ioctl_smoke_nonzero_exit" } else { $null }
    }
    Save-PhaseResult -PhaseName "phase4_ioctl_smoke_r${RebootCount}" -Data $result | Out-Null
    Assert-PhaseOk -Result $result -PhaseName "ioctl_smoke"
    Write-Host "[Phase 4] PASS"
}

# ---------------------------------------------------------------------------
# Phase 5 - Reboot Gate（提示 3 次重啟）
# ---------------------------------------------------------------------------

function Invoke-PhaseRebootGate {
    Write-Step "[Phase 5/7] Reboot Gate (3 reboots required)"
    Write-Host ""
    Write-Host "請依序手動重啟 VM 三次，每次重啟後重新執行："
    Write-Host ""
    Write-Host "  powershell -ExecutionPolicy Bypass -File \"$($MyInvocation.ScriptName)\" -Phase verify"
    Write-Host "  powershell -ExecutionPolicy Bypass -File \"$($MyInvocation.ScriptName)\" -Phase ioctl"
    Write-Host ""
    Write-Host "確認三次重啟後均無 BSOD，且 Phase 3+4 通過，再執行 -Phase verifier"
    Write-Host ""
    $result = @{
        ok = $true
        message = "reboot_gate_requires_manual_reboots"
        reboot_count_required = 3
    }
    Save-PhaseResult -PhaseName "phase5_reboot_gate" -Data $result | Out-Null
    Write-Host "[Phase 5] 提示已顯示。請重啟 VM 三次並驗證後再繼續。"
}

# ---------------------------------------------------------------------------
# Phase 6 - Driver Verifier
# ---------------------------------------------------------------------------

function Invoke-PhaseVerifier {
    Write-Step "[Phase 6/7] Driver Verifier"
    $dvScript = Join-Path $scriptDir "setup_driver_verifier.ps1"
    Write-Host "[Phase 6] 啟用 Driver Verifier..."
    & $dvScript -Enable
    Write-Host ""
    Write-Host "[Phase 6] Driver Verifier 已設定。請手動重啟 VM，然後執行 IOCTL smoke test："
    Write-Host ""
    Write-Host "  powershell -ExecutionPolicy Bypass -File \"$($MyInvocation.ScriptName)\" -Phase ioctl"
    Write-Host ""
    Write-Host "Smoke 完成後查詢違規："
    Write-Host ""
    Write-Host "  powershell -ExecutionPolicy Bypass -File \"$($MyInvocation.ScriptName)\" -Phase verifier_query"
    Write-Host ""
    $result = @{
        ok = $true
        message = "driver_verifier_enabled_reboot_required"
    }
    Save-PhaseResult -PhaseName "phase6_verifier_enable" -Data $result | Out-Null
}

function Invoke-PhaseVerifierQuery {
    Write-Step "[Phase 6b/7] Driver Verifier Query"
    $dvScript = Join-Path $scriptDir "setup_driver_verifier.ps1"
    & $dvScript -Query
    $result = @{
        ok = $true
        message = "driver_verifier_query_complete"
    }
    Save-PhaseResult -PhaseName "phase6_verifier_query" -Data $result | Out-Null
}

# ---------------------------------------------------------------------------
# Phase 7 - Uninstall Gate
# ---------------------------------------------------------------------------

function Invoke-PhaseUninstall {
    Write-Step "[Phase 7/7] Uninstall Gate"
    $uninstallScript = Join-Path $scriptDir "uninstall_driver_package.ps1"
    $output = & $uninstallScript 2>&1
    $exitCode = $LASTEXITCODE
    $output | ForEach-Object { Write-Host $_ }

    Write-Host "[Phase 7] 驗證裝置已移除..."
    Start-Sleep -Seconds 3
    $verifyScript = Join-Path $scriptDir "verify_driver_install.ps1"
    $verifyOutput = & $verifyScript -Json 2>&1
    $verifyExit = $LASTEXITCODE
    # 解除安裝後 verify 應該失敗（exit ≠ 0）才是正確的
    $ok = ($exitCode -eq 0) -and ($verifyExit -ne 0)

    $result = @{
        ok = $ok
        uninstall_exit_code = $exitCode
        verify_exit_after_uninstall = $verifyExit
        fail_reason = if (!$ok) {
            if ($exitCode -ne 0) { "uninstall_script_failed" }
            else { "device_still_present_after_uninstall" }
        } else { $null }
    }
    Save-PhaseResult -PhaseName "phase7_uninstall" -Data $result | Out-Null
    Assert-PhaseOk -Result $result -PhaseName "uninstall"
    Write-Host "[Phase 7] PASS — 裝置已清除"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

Assert-Elevated

$outDir = Join-Path $repoRoot $LogDir
New-Item -ItemType Directory -Path $outDir -Force | Out-Null

Write-Host ""
Write-Host "============================================================"
Write-Host " SyncTranslate VM Smoke Sequence"
Write-Host " Phase: $Phase"
Write-Host " 警告：只在 disposable Windows VM 中執行。"
Write-Host " 請先建立 VM snapshot 再繼續。"
Write-Host "============================================================"

switch ($Phase) {
    "preflight"  { Invoke-PhasePreflight }
    "install"    { Invoke-PhaseInstall }
    "verify"     { Invoke-PhaseVerify -RebootCount 0 }
    "ioctl"      { Invoke-PhaseIoctlSmoke -RebootCount 0 }
    "verifier"   { Invoke-PhaseVerifier }
    "uninstall"  { Invoke-PhaseUninstall }
    "all" {
        Invoke-PhasePreflight
        Invoke-PhaseInstall
        Invoke-PhaseVerify -RebootCount 0
        Invoke-PhaseIoctlSmoke -RebootCount 0
        Invoke-PhaseRebootGate
        # Phase 5 (reboot gate) 之後的步驟需要手動重啟後再執行：
        #   -Phase verify / -Phase ioctl（各 3 次）
        #   -Phase verifier
        #   -Phase uninstall
    }
}

Write-Host ""
Write-Host "Done. 結果存放在：$(Join-Path $repoRoot $LogDir)"
