param(
    [string]$Destination = "downloads/driver_samples/Windows-driver-samples"
)

$ErrorActionPreference = "Stop"

if (!(Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git.exe is required to fetch the Microsoft Windows-driver-samples repository."
}

$repoUrl = "https://github.com/microsoft/Windows-driver-samples.git"

if (Test-Path $Destination) {
    Write-Host "[sysvad] repository already exists: $Destination"
    Push-Location $Destination
    try {
        git sparse-checkout set audio/sysvad
        git submodule update --init
    }
    finally {
        Pop-Location
    }
    exit 0
}

$parent = Split-Path -Parent $Destination
if ($parent) {
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
}

Write-Host "[sysvad] cloning Microsoft Windows-driver-samples sparse checkout"
git clone --filter=blob:none --sparse $repoUrl $Destination
if ($LASTEXITCODE -ne 0) {
    throw "git clone failed with exit code $LASTEXITCODE"
}

Push-Location $Destination
try {
    git sparse-checkout set audio/sysvad
    git submodule update --init
}
finally {
    Pop-Location
}

Write-Host "[sysvad] ready: $Destination\audio\sysvad"
