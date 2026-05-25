# SyncTranslate

SyncTranslate is a local Windows desktop tool for real-time captions, translation, and two-way voice translation.

## Product Modes

- `meeting`: monitor-only captions, translation, and transcript recording from a selected system input or WASAPI output loopback device. This mode does not use the SyncTranslate virtual audio driver, bridge, virtual speaker, or virtual microphone.
- `dialogue`: two-way call translation through SyncTranslate Virtual Speaker and SyncTranslate Virtual Microphone. This mode requires the virtual audio driver and bridge.

ASR uses faster-whisper/CTranslate2. App-internal audio uses float32 `AudioFrame` data with sample-rate/channel metadata. The ASR branch converts to 16 kHz mono float32. The Windows/driver/bridge boundary is fixed at `48000Hz / PCM16 / 2ch`.

In meeting mode, TTS off still runs ASR, translation, captions, and transcript recording. In dialogue mode, `tts_voice: none` means direct passthrough for that direction and does not produce captions or translations for that direction.

## Quick Start

```powershell
uv sync
powershell -ExecutionPolicy Bypass -File .\tools\runtime_setup\prepare_external_runtimes.ps1
uv run python main.py --check
uv run python main.py
```

The expected runtime layout is:

```text
runtimes/
  shared/
  faster_whisper/
  models/
    asr/
      large-v3-turbo/
    llm/
      hy-mt1.5-7b.gguf
  audio/
    sync_audio_bridge.exe
```

## Driver And Dialogue Mode

Meeting mode is the portable no-driver path.

Dialogue mode requires:

- SyncTranslate Virtual Speaker
- SyncTranslate Virtual Microphone
- `runtimes/audio/sync_audio_bridge.exe`
- driver endpoint format `48000Hz / PCM16 / 2ch`

Development builds use a test-signed driver. Test Mode is acceptable for disposable VMs and lab machines only. A public release needs production driver signing/WHQL or equivalent Microsoft-approved signing; normal users should not be asked to enable Test Mode manually.

The current one-click development MSI is expected at:

```text
artifacts/driver/synctranslate_virtual_audio/SyncTranslateVirtualAudioDriver.msi
```

After MSI install and reboot, a healthy dialogue-mode driver install shows:

- `SyncTranslate Virtual Audio Device` under **Sound, video and game controllers** with no warning icon.
- `SyncTranslate Virtual Speaker` under **Audio inputs and outputs**.
- `SyncTranslate Virtual Microphone` under **Audio inputs and outputs**.

Verify it with:

```powershell
powershell -ExecutionPolicy Bypass -File .\drivers\synctranslate_virtual_audio\scripts\verify_driver_format.ps1 -JsonOutput .\downloads\validation\driver_format.json
uv run python .\tools\validation\validate_windows_audio_runtime.py --no-capture-probe --no-bridge-probe
uv run python .\tools\validation\audio_smoke_test.py dialogue-passthrough --duration 0.1
```

The format verifier checks `PKEY_AudioEngine_DeviceFormat` for the v2 boundary. WASAPI shared-mode mix may report 48 kHz float32 because Windows mixes in float; that value is diagnostic detail, not the driver boundary.

## Validation

Run these before release or on a new Windows test machine:

```powershell
python -m compileall -q app tests tools
uv run pytest -q
uv run python main.py --check
uv run python .\tools\validation\preflight_release_check.py --mode meeting
uv run python .\tools\validation\preflight_release_check.py --mode dialogue
uv run python .\tools\validation\export_diagnostics_bundle.py
```

Useful validation tools:

- `tools/validation/validate_windows_audio_runtime.py`: Windows device, meeting, dialogue, bridge, driver checks.
- `tools/validation/audio_smoke_test.py`: meeting input, meeting loopback, and dialogue passthrough smoke tests.
- `tools/validation/preflight_release_check.py`: portable release readiness and runtime/model checks.
- `tools/validation/export_diagnostics_bundle.py`: sanitized diagnostic zip for issue reports.
- `tools/hyperv/run_driver_vm_validation.ps1`: Hyper-V driver build/install/format validation harness.

Generated diagnostics and smoke outputs go under `downloads/validation/`, which is intentionally ignored by Git.

## Release Zip Layout

```text
SyncTranslate-onedir/
  SyncTranslate.exe
  config.yaml
  runtimes/
    shared/
    faster_whisper/
    models/
      asr/
        large-v3-turbo/
      llm/
        hy-mt1.5-7b.gguf
    audio/
      sync_audio_bridge.exe
  tools/
    validation/
    runtime_setup/
```

`meeting` mode must start on a normal Windows PC without the virtual audio driver. `dialogue` mode must clearly report missing driver/bridge instead of crashing.

## Documentation

- [Configuration](docs/設定說明.md)
- [Architecture](docs/架構說明.md)
- [Testing](docs/測試說明.md)
- [Windows v2 acceptance checklist](docs/v2_windows_驗收清單.md)
- [Driver build and signing](docs/driver_build_and_signing.md)
- [Virtual audio protocol v2](docs/virtual_audio_protocol_v2.md)

## Important Scope Boundaries

This v2 product line does not add whisper.cpp, Windows built-in ASR, or cloud ASR. Runtime ASR remains faster-whisper/CTranslate2 with fixed UI languages: `zh-TW`, `en`, `ja`, `ko`, and `th`.
