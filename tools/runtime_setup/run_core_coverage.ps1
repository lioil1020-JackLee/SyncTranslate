#!/usr/bin/env powershell
# Run core-module coverage in isolated subprocesses to reduce runtime path interference.

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\.." )).Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Python not found: $python"
}

Write-Host "[coverage] repo: $repoRoot"
Write-Host "[coverage] python: $python"

# Avoid test-time external runtime path injection if present.
$env:SYNC_TRANSLATE_SKIP_EXTERNAL_RUNTIME = "1"

function Run-Coverage {
    param(
        [string]$Module,
        [string[]]$Tests,
        [string]$DataFile
    )

    $args = @("-m", "pytest") + $Tests + @("--cov=$Module", "--cov-report=term", "-q")
    $env:COVERAGE_FILE = $DataFile
    Write-Host "`n[coverage] module=$Module"
    Write-Host "[coverage] tests=$($Tests -join ', ')"
    & $python @args
    if ($LASTEXITCODE -ne 0) {
        throw "Coverage run failed for $Module (exit=$LASTEXITCODE)"
    }
}

# session_service
Run-Coverage -Module "app.application.session_service" -Tests @("tests/test_session_service.py") -DataFile ".coverage.session_service"

# translation_dispatcher
Run-Coverage -Module "app.application.translation_dispatcher" -Tests @("tests/test_translation_manager_profiles.py", "tests/test_pipeline_integration.py") -DataFile ".coverage.translation_dispatcher"

# worker_v2
# NOTE: In some Windows environments numpy/native extension state can still conflict under coverage.
# Keep this run separate so a failure does not hide previous successful module reports.
try {
    Run-Coverage -Module "app.infra.asr.worker_v2" -Tests @("tests/test_asr_v2_endpointing.py", "tests/test_asr_streaming_and_profiles.py") -DataFile ".coverage.worker_v2"
}
catch {
    Write-Warning "worker_v2 coverage failed in current environment: $($_.Exception.Message)"
    Write-Warning "Try running this script in a fresh shell with no preloaded Python processes."
}

Write-Host "`n[coverage] done"
