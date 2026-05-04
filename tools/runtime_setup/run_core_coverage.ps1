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
Run-Coverage -Module "app.application.translation_dispatcher" -Tests @("tests/test_audio_router_refactor_smoke.py") -DataFile ".coverage.translation_dispatcher"

# worker_v2
# worker_v2 — use coverage run (not pytest-cov) to avoid numpy C-extension double-load
Write-Host "`n[coverage] module=app.infra.asr.worker_v2 (via coverage run)"
$env:COVERAGE_FILE = ".coverage.worker_v2"
& $python -m coverage run --data-file=".coverage.worker_v2" -m pytest tests/test_asr_v2_endpointing.py tests/test_asr_streaming_and_profiles.py -q
& $python -m coverage report --data-file=".coverage.worker_v2" --include="app/infra/asr/worker_v2.py"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "worker_v2 coverage failed (exit=$LASTEXITCODE)"
}

Write-Host "`n[coverage] done"
