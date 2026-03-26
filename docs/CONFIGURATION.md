# Configuration

## Main file

Use `config.yaml` for your local machine.

A canonical example is provided in [`config.example.yaml`](../config.example.yaml).

## Canonical external keys

### Language

- `language.remote_translation_target`
- `language.local_translation_target`

### Runtime

- `runtime.remote_translation_enabled`
- `runtime.local_translation_enabled`
- `runtime.tts_output_mode`

## Fixed behavior

These behaviors are now fixed by the application and are not intended to be user-tuned:

- `direction.mode = bidirectional`
- `runtime.asr_language_mode = auto`
- direction-specific ASR strategy is always enabled
- direction-specific LLM strategy is always enabled
- startup warmup is disabled

## Audio routing

`audio` contains four device selections:

- `meeting_in`
- `microphone_in`
- `speaker_out`
- `meeting_out`

Each route also has gain / volume fields.

## ASR

Shared defaults live under `asr`.

Direction overrides live under:

- `asr_channels.local`
- `asr_channels.remote`

## LLM

Shared defaults live under `llm`.

Direction-specific runtime selections live under:

- `llm_channels.local`
- `llm_channels.remote`

Each direction can use a different LM Studio model.

## TTS

TTS uses `edge-tts`.

Relevant sections:

- `tts`
- `meeting_tts`
- `local_tts`
- `tts_channels.local`
- `tts_channels.remote`

`style_preset` controls speech pacing policy for edge-tts.

## Runtime artifacts

The app writes runtime artifacts to the system temp directory under `SyncTranslate`:

- crash log
- runtime event log
- system-check snapshots
