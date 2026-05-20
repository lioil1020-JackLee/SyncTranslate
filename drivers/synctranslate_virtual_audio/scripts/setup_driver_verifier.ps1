param(
    [switch]$Enable,
    [switch]$Disable,
    [switch]$Query,
    [string]$DriverBinary = "tabletaudiosample.sys"
)

$ErrorActionPreference = "Stop"

function Assert-Elevated {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($id)
    if (!$principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw "elevated PowerShell required"
    }
}

function Get-VerifierPath {
    $path = Join-Path $env:windir "System32\verifier.exe"
    if (Test-Path $path) {
        return $path
    }
    $cmd = Get-Command verifier.exe -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }
    throw "verifier.exe not found"
}

function Invoke-Verifier {
    param([string[]]$Arguments)
    $verifier = Get-VerifierPath
    Write-Host "[dv] verifier $($Arguments -join ' ')"
    $output = & $verifier @Arguments 2>&1
    $exitCode = $LASTEXITCODE
    $output | ForEach-Object { Write-Host "    $_" }
    if ($exitCode -ne 0 -and $exitCode -ne 2) {
        throw "verifier.exe exited with code $exitCode"
    }
    return $output
}

function Enable-DriverVerifier {
    param([string]$Driver)
    Write-Host ""
    Write-Host "[dv] WARNING: run only in a disposable VM"
    Write-Host "[dv] Enabling verifier for: $Driver"
    Write-Host ""

    # 0x209bb = Special Pool + IRQL checking + Pool tracking
    #         + I/O verification + Deadlock detection + standard checks
    $flags = "0x209bb"

    try {
        Invoke-Verifier -Arguments @("/reset") | Out-Null
    }
    catch {
        Write-Host "[dv] reset skipped: $($_.Exception.Message)"
    }

    Invoke-Verifier -Arguments @("/flags", $flags, "/driver", $Driver) | Out-Null
    Write-Host "[dv] verifier configuration applied; reboot required"
}

function Disable-DriverVerifier {
    Write-Host "[dv] disabling verifier settings"
    Invoke-Verifier -Arguments @("/reset") | Out-Null
    Write-Host "[dv] verifier reset done; reboot recommended"
}

function Query-DriverVerifier {
    Write-Host "[dv] querysettings"
    Invoke-Verifier -Arguments @("/querysettings") | Out-Null
    Write-Host ""
    Write-Host "[dv] query"
    Invoke-Verifier -Arguments @("/query") | Out-Null
}

if (!$Enable -and !$Disable -and !$Query) {
    Write-Host "Usage: -Enable | -Disable | -Query"
    exit 0
}

Assert-Elevated

if ($Enable) {
    Enable-DriverVerifier -Driver $DriverBinary
}
elseif ($Disable) {
    Disable-DriverVerifier
}
elseif ($Query) {
    Query-DriverVerifier
}