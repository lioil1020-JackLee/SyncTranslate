# Remote Audio Passthrough 16kHz Issue - Root Cause Analysis

## Executive Summary

Remote audio passthrough is being resampled from 48kHz to 16kHz **at the bridge level** because the AudioRouter is passing the configured sample rate (which defaults to 48kHz but can be set to 16kHz for ASR optimization) directly to the remote audio source.

**The Issue:** The same `sample_rate` parameter used for ASR processing (16kHz) is being applied to remote speaker audio capture from the bridge, causing the bridge to resample the original 48kHz speaker audio down to 16kHz before sending it.

## Call Chain: Where 16kHz Originates

```
1. UI Layer (app/ui/main_window.py:424)
   └─→ Reads: self.config.runtime.sample_rate
       └─→ If config has "sample_rate: 16000" → uses 16000

2. Session Service (app/application/session_service.py:70)
   └─→ Calls: self._audio_router.start(mode, routes, sample_rate, ...)
       └─→ Receives 16000 if that was in config

3. Audio Router (app/application/audio_router.py:109)
   └─→ Sets: self._sample_rate = int(sample_rate)
       └─→ Stores 16000

4. Audio Router Reconcile (app/application/audio_router.py:627)
   └─→ Calls: self._input_manager.start(source, ..., sample_rate=self._sample_rate, ...)
       └─→ Passes 16000 for BOTH local AND remote

5. Input Manager (app/infra/audio/routing.py:47)
   └─→ Routes to: capture.start(device_name, sample_rate=sample_rate, ...)
       └─→ For "remote": calls VirtualSpeakerSource.start(..., sample_rate=16000, ...)

6. VirtualSpeakerSource (app/infra/audio/sources.py:77)
   └─→ Calls: self._bridge.start_remote_input(sample_rate=int(sample_rate), ...)
       └─→ **CRITICAL: Passes 16000 to the bridge**

7. Virtual Bridge Client (app/infra/audio/virtual_bridge_client.py:94)
   └─→ Sets: self._remote_input_sample_rate = int(sample_rate)
       └─→ **Stores: 16000**

8. Remote Input Poll (app/infra/audio/virtual_bridge_client.py:380)
   └─→ Requests: self._request({"cmd": "read_remote_input", "sample_rate": int(self._remote_input_sample_rate)})
       └─→ **Tells bridge: "Send me remote audio at 16000 Hz"**

9. Bridge Handler (app/infra/audio/bridge_process.py)
   └─→ Receives command to capture remote audio at 16000 Hz
       └─→ **Resamples 48kHz speaker audio → 16kHz**

10. Audio Returned to App (app/infra/audio/virtual_bridge_client.py:389)
    └─→ audio, sample_rate = decode_audio_packet(packet)
        └─→ Receives 16kHz audio

11. VirtualSpeakerSource Emission (app/infra/audio/sources.py:161)
    └─→ Calls: consumer(chunk, sample_rate)
        └─→ **Passes 16kHz to _on_remote_audio_chunk()**

12. TTS Passthrough (app/application/audio_router.py:289)
    └─→ Calls: self._tts_manager.submit_passthrough(passthrough_channel, chunk, sample_rate)
        └─→ **TTS gets 16kHz audio**

13. Playback (app/infra/tts/playback_queue.py:347, 502)
    └─→ Calls: self._sinks[key].push_passthrough(audio, sample_rate=float(sample_rate))
        └─→ **Playback device receives 16kHz**
```

## The Core Problem

**File: `app/application/audio_router.py` line 109 and 627**

```python
# AudioRouter.start() receives sample_rate parameter
def start(self, mode: str, routes: AudioRouteConfig, sample_rate: int, chunk_ms: int = ASR_DEFAULT_CHUNK_MS) -> None:
    ...
    self._sample_rate = int(sample_rate)  # Stores it (could be 16000 if configured for ASR)
    ...

# Later, in _reconcile_single_source()
def _reconcile_single_source(self, *, source: str, capture_needed: bool, asr_needed: bool) -> None:
    ...
    self._input_manager.start(
        source,
        self._device_of(source),
        sample_rate=self._sample_rate,  # ← SAME sample_rate for BOTH local and remote!
        chunk_ms=self._chunk_ms,
    )
```

**The Problem:** This `self._sample_rate` applies to **BOTH** local and remote sources, but:
- **Local audio:** 16kHz is fine (user is speaking, ASR processes at 16kHz)
- **Remote audio:** Should be 48kHz (speaker audio quality), NOT resampled down to 16kHz

## What Happens When config.runtime.sample_rate = 16000

1. User configures `sample_rate: 16000` for ASR optimization
2. This is intended for ASR processing of local microphone audio
3. But it gets applied to remote speaker audio capture too
4. The bridge receives request: "Send me remote input at 16000 Hz"
5. Bridge resamples 48kHz speaker audio → 16kHz
6. Audio arrives at router at 16kHz
7. Passthrough outputs 16kHz audio (quality loss, bandwidth reduction)

## Diagnostic Log Evidence

```
passthrough_playback_started: sample_rate=48000.0 channels=2 requested_rate=16000.0
```

- `sample_rate=48000.0`: The output device plays back at 48kHz
- `requested_rate=16000.0`: But the audio was **requested/provided** at 16kHz
- The playback layer had to upsample 16kHz → 48kHz, introducing quality issues

This confirms the audio arrived at the TTS manager at 16kHz.

## Architecture Issue

The current design assumes one sample rate for all audio processing:

```
AudioRouter.sample_rate
  ├─ Local mic capture     ← ASR needs 16kHz, can use configured rate
  ├─ Local ASR processing  ← Needs 16kHz
  ├─ Remote speaker audio  ← ⚠️ SHOULD BE 48kHz (NOT resampled)
  ├─ Remote ASR processing ← Needs 16kHz
  └─ Passthrough output    ← Should preserve original 48kHz
```

## Correct Flow (What Should Happen)

```
Remote Speaker (48kHz)
    ↓
VirtualSpeakerSource.start() should request 48kHz, NOT the configured sample_rate
    ↓
Bridge provides 48kHz audio
    ↓
_on_remote_audio_chunk() receives 48kHz
    ↓
TTS passthrough gets 48kHz → outputs 48kHz (quality preserved) ✓
    ↓
ASR processing: If needed, resample 48kHz → 16kHz for recognition ✓
```

## Code Locations Involved

| Component | File | Line | Issue |
|-----------|------|------|-------|
| UI Start | `app/ui/main_window.py` | 424 | Passes config.runtime.sample_rate |
| Session Service | `app/application/session_service.py` | 70 | Forwards sample_rate to router |
| Audio Router | `app/application/audio_router.py` | 109, 627 | Applies same rate to all sources |
| Input Manager | `app/infra/audio/routing.py` | 47, 53 | Routes to VirtualSpeakerSource |
| VirtualSpeakerSource | `app/infra/audio/sources.py` | 77 | **[KEY]** Passes sample_rate to bridge |
| Virtual Bridge Client | `app/infra/audio/virtual_bridge_client.py` | 94, 380 | **[KEY]** Requests bridge at specific rate |
| Remote Input Poll | `app/infra/audio/virtual_bridge_client.py` | 389 | Receives resampled audio |
| TTS Passthrough | `app/infra/tts/playback_queue.py` | 347, 502 | Outputs received rate |

## Solution Options

### Option 1: Hardcode 48kHz for Remote Audio
```python
# In VirtualSpeakerSource.start(), always use 48kHz for remote
self._bridge.start_remote_input(
    sample_rate=48000,  # ← Always 48kHz, ignore passed sample_rate
    device_name=str(device_name or ""),
    chunk_ms=int(chunk_ms),
)
```

### Option 2: Separate Configuration Parameter
Add to `RuntimeConfig`:
```python
remote_sample_rate: int = 48000  # Always 48kHz for remote speaker
asr_sample_rate: int = 16000     # Process ASR at 16kHz
```

Then resample after capture when needed for ASR.

### Option 3: Intelligent Parameter Routing
Have `AudioRouter.start()` accept separate parameters:
```python
def start(self, mode: str, routes: AudioRouteConfig, 
          asr_sample_rate: int = 16000,  # For ASR/local processing
          remote_sample_rate: int = 48000,  # For remote speaker
          chunk_ms: int = ASR_DEFAULT_CHUNK_MS) -> None:
```

## Why This Bug Exists

The audio architecture was designed assuming a single sample rate throughout, but the actual requirements are:
- **Local mic → ASR:** 16kHz (Whisper optimal, bandwidth efficient)
- **Remote speaker → Passthrough:** 48kHz (Quality preservation)
- **Remote speaker → ASR:** Can be either, but should preserve original then resample

The `config.runtime.sample_rate` is an ASR optimization parameter, not meant to apply globally to remote audio.
