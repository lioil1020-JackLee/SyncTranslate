param(
    [string]$OutputDir = "runtimes/audio",
    [string]$WorkDir = "build/audio_bridge",
    [string]$SpecDir = "build/audio_bridge_spec"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
New-Item -ItemType Directory -Path $WorkDir -Force | Out-Null
New-Item -ItemType Directory -Path $SpecDir -Force | Out-Null

$pyInstaller = @(
    "run", "--group", "build", "pyinstaller",
    "--onefile",
    "--clean",
    "--name", "sync_audio_bridge",
    "--distpath", $OutputDir,
    "--workpath", $WorkDir,
    "--specpath", $SpecDir,
    "--hidden-import", "app.infra.audio.bridge_protocol",
    "--hidden-import", "app.infra.audio.bridge_ring_buffer",
    "--hidden-import", "app.infra.audio.windows_named_event",
    "--hidden-import", "soundcard",
    "--exclude-module", "torch",
    "--exclude-module", "torchaudio",
    "--exclude-module", "onnxruntime",
    "--exclude-module", "faster_whisper",
    "--exclude-module", "ctranslate2",
    "--exclude-module", "pytest",
    "--exclude-module", "PySide6",
    "--exclude-module", "edge_tts",
    "--exclude-module", "matplotlib",
    "--exclude-module", "pandas",
    "--exclude-module", "scipy",
    "app/infra/audio/bridge_process.py"
)

Write-Host "[audio-bridge] packaging sync_audio_bridge.exe"
& uv @pyInstaller
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}

$exe = Join-Path $OutputDir "sync_audio_bridge.exe"
if (!(Test-Path $exe)) {
    throw "Expected bridge executable was not produced: $exe"
}

Get-Item $exe | Select-Object FullName,Length | Format-Table -AutoSize
