# Remote Audio 16kHz Issue - Visual Data Flow Diagram

## Problem Scenario: User Sets sample_rate: 16000

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    USER CONFIG: sample_rate: 16000                           │
│                  (Intended for ASR optimization only)                        │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │  app/ui/main_window.py:424   │
                   │ sample_rate from config      │
                   │ (16000)                      │
                   └──────────────┬───────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │ SessionService.start()       │
                   │ app/application/             │
                   │ session_service.py:70        │
                   │ Passes: sample_rate=16000    │
                   └──────────────┬───────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │ AudioRouter.start()          │
                   │ app/application/             │
                   │ audio_router.py:109          │
                   │ self._sample_rate = 16000    │
                   └──────────────┬───────────────┘
                                  │
                   ┌──────────────┴──────────────┐
                   │ (Applied to BOTH sources)  │
                   ▼                             ▼
        ┌──────────────────────┐   ┌──────────────────────────┐
        │  Local Source (OK)   │   │  Remote Source (WRONG!)  │
        │                      │   │                          │
        │ Local microphone:    │   │ Remote speaker audio:    │
        │ Mic → 16kHz OK ✓     │   │ Speaker (48kHz) →        │
        │                      │   │ Bridge requests 16kHz    │
        │ Local ASR: 16kHz ✓   │   │ Bridge resamples:        │
        │                      │   │ 48kHz → 16kHz ✗          │
        └──────────────────────┘   └───────────┬──────────────┘
                                               │
                                   ┌───────────┴────────────┐
                                   │  PROBLEM ZONE          │
                                   └───────────┬────────────┘
                                               │
                                               ▼
                        ┌─────────────────────────────────┐
                        │ AudioRouter._reconcile_single_  │
                        │ source() line 627               │
                        │                                 │
                        │ self._input_manager.start(      │
                        │   source="remote",              │
                        │   sample_rate=16000  ◄──────┐  │
                        │ )                             │  │
                        └────────────┬──────────────────┘  │
                                     │                     │
                                     │          (Wrong:    │
                                     │       should be      │
                                     │        48000)        │
                                     │                     │
                                     ▼                     │
                        ┌─────────────────────────────────┘
                        │ AudioInputManager.start()
                        │ app/infra/audio/routing.py:47
                        │
                        │ capture.start(..., sample_rate=16000)
                        │         ▲
                        │         │
                        │ For remote: VirtualSpeakerSource
                        └─────────┬────────────────────────
                                  │
                                  ▼
                        ┌──────────────────────────┐
                        │ VirtualSpeakerSource.    │
                        │ start()                  │
                        │ app/infra/audio/         │
                        │ sources.py:77            │
                        │                          │
                        │ self._bridge.            │
                        │ start_remote_input(      │
                        │   sample_rate=16000  ◄──┐
                        │ )                        │
                        └────────────┬─────────────┘
                                     │
                                     │ [KEY DECISION POINT]
                                     │ Passes 16000 to bridge
                                     │
                                     ▼
                        ┌──────────────────────────┐
                        │ VirtualBridgeClient      │
                        │ start_remote_input()     │
                        │ virtual_bridge_client.   │
                        │ py:94                    │
                        │                          │
                        │ self._remote_input_      │
                        │ sample_rate = 16000      │
                        └────────────┬─────────────┘
                                     │
                                     ▼
                        ┌──────────────────────────┐
                        │ Bridge Command Sent:     │
                        │                          │
                        │ {                        │
                        │   "cmd":                 │
                        │   "start_remote_input",  │
                        │   "sample_rate": 16000   │
                        │ }                        │
                        │                          │
                        │ Bridge says:             │
                        │ "Send me 16kHz audio"    │
                        └────────────┬─────────────┘
                                     │
                                     ▼
                        ┌──────────────────────────────┐
                        │ BRIDGE (External Process)    │
                        │                              │
                        │ Receives: 48kHz speaker audio│
                        │ Sees request: 16kHz          │
                        │                              │
                        │ ACTION: RESAMPLE!            │
                        │ 48kHz → 16kHz                │
                        │                              │
                        │ (Loses quality!)             │
                        └────────────┬─────────────────┘
                                     │
                                     ▼
                        ┌──────────────────────────┐
                        │ Remote Input Poll Thread │
                        │ virtual_bridge_client.   │
                        │ py:380-389               │
                        │                          │
                        │ payload = self._request({│
                        │   "cmd":                 │
                        │   "read_remote_input",   │
                        │   "sample_rate": 16000   │
                        │ })                       │
                        │                          │
                        │ audio, sample_rate =     │
                        │ decode_audio_packet(p)   │
                        │                          │
                        │ sample_rate = 16000      │
                        │ (From bridge response)   │
                        │                          │
                        │ for consumer in          │
                        │   consumers:             │
                        │   consumer(audio, 16000) │
                        └────────────┬─────────────┘
                                     │
                                     ▼
                        ┌──────────────────────────┐
                        │ VirtualSpeakerSource     │
                        │ ._dispatch_audio()       │
                        │ sources.py:130           │
                        │                          │
                        │ _emit_chunk(chunk, 16000)
                        └────────────┬─────────────┘
                                     │
                                     ▼
                        ┌──────────────────────────┐
                        │ VirtualSpeakerSource     │
                        │ ._emit_chunk()           │
                        │ sources.py:161           │
                        │                          │
                        │ consumer(chunk, 16000)   │
                        └────────────┬─────────────┘
                                     │
                                     ▼
                        ┌──────────────────────────┐
                        │ AudioRouter              │
                        │ ._on_remote_audio_chunk()│
                        │ audio_router.py:266      │
                        │                          │
                        │ def _on_remote_audio_    │
                        │ chunk(self, chunk, 16000)│
                        │                          │
                        │ Receives: 16kHz ✗        │
                        └────────────┬─────────────┘
                                     │
                                     ▼
                        ┌──────────────────────────┐
                        │ AudioRouter              │
                        │ ._handle_source_audio_   │
                        │ chunk()                  │
                        │ audio_router.py:275      │
                        │                          │
                        │ TTS passthrough gets:    │
                        │ 16kHz audio ✗            │
                        └────────────┬─────────────┘
                                     │
                                     ▼
                        ┌──────────────────────────┐
                        │ TTSManager               │
                        │ .submit_passthrough()    │
                        │ playback_queue.py:319    │
                        │                          │
                        │ Queue task with:         │
                        │ audio, sample_rate=16000 │
                        └────────────┬─────────────┘
                                     │
                                     ▼
                        ┌──────────────────────────┐
                        │ Passthrough Playback     │
                        │ playback.py:141          │
                        │ .push_passthrough(       │
                        │   audio, 16000          │
                        │ )                        │
                        │                          │
                        │ DIAGNOSTIC LOG SHOWS:    │
                        │ requested_rate=16000 ✗   │
                        └────────────┬─────────────┘
                                     │
                                     ▼
                        ┌──────────────────────────┐
                        │ Output Device (48kHz)    │
                        │                          │
                        │ Receives: 16kHz audio    │
                        │ Must upsample: 16→48kHz  │
                        │ Quality Loss! ✗          │
                        └──────────────────────────┘
```

## What SHOULD Happen (Fixed Flow)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    USER CONFIG: sample_rate: 16000                           │
│                  (For ASR optimization)                                      │
│                    remote_sample_rate: 48000  (NEW)                          │
│                  (For speaker audio quality)                                 │
└─────────────────┬──────────────────────────────────┬────────────────────────┘
                  │                                  │
        ┌─────────▼──────────┐          ┌────────────▼──────────┐
        │ For local ASR:     │          │ For remote speaker:   │
        │ 16000 Hz ✓         │          │ 48000 Hz ✓            │
        └─────────┬──────────┘          └────────────┬──────────┘
                  │                                  │
                  │   ┌─────────────────────────────┘
                  │   │
                  ▼   ▼
        ┌──────────────────────────────────────┐
        │ AudioRouter.start(                   │
        │   sample_rate=16000,   [for ASR]     │
        │   remote_sample_rate=48000  [NEW]    │
        │ )                                    │
        │ audio_router.py:109                  │
        │                                      │
        │ self._sample_rate = 16000            │
        │ self._remote_sample_rate = 48000     │
        └────────┬──────────────────────────────┘
                 │
                 ▼
        ┌──────────────────────────────┐
        │ _reconcile_single_source()   │
        │ audio_router.py:627          │
        │                              │
        │ source_sample_rate =         │
        │   (48000 if source=="remote" │
        │    else 16000)  ◄─── NEW!    │
        │                              │
        │ self._input_manager.start(   │
        │   sample_rate=               │
        │   source_sample_rate         │
        │ )                            │
        └────────┬─────────────────────┘
                 │
        ┌────────┴──────────┐
        │                   │
        ▼                   ▼
     LOCAL            REMOTE (48kHz!)
     (16kHz)
        │                   │
        │                   ▼
        │         ┌──────────────────────┐
        │         │ VirtualSpeaker       │
        │         │ Source.start()       │
        │         │                      │
        │         │ Bridge request:      │
        │         │ "Send 48kHz" ✓       │
        │         └──────────┬───────────┘
        │                    │
        │                    ▼
        │         ┌──────────────────────┐
        │         │ Bridge Response:     │
        │         │ 48kHz audio ✓        │
        │         └──────────┬───────────┘
        │                    │
        │                    ▼
        │         ┌──────────────────────┐
        │         │ TTS Passthrough:     │
        │         │ 48kHz audio ✓        │
        │         │                      │
        │         │ DIAGNOSTIC LOG:      │
        │         │ requested_rate=48000 │
        │         │ sample_rate=48000    │
        │         │ Match! ✓             │
        │         └──────────┬───────────┘
        │                    │
        │                    ▼
        │         ┌──────────────────────┐
        │         │ Output Device:       │
        │         │ 48kHz → 48kHz ✓      │
        │         │ No upsampling! ✓     │
        │         │ Full quality! ✓      │
        │         └──────────────────────┘
        │
        ▼
     ASR (if needed):
     May downsample
     16kHz ✓
```

## Key Differences

| Aspect | Before Fix (16kHz) | After Fix (48kHz) |
|--------|-------------------|------------------|
| **Bridge request** | 16000 Hz (Wrong!) | 48000 Hz (Correct!) |
| **Resampling at bridge** | 48kHz → 16kHz ✗ | No resampling ✓ |
| **Remote audio arrives at** | 16 kHz ✗ | 48 kHz ✓ |
| **Passthrough receives** | 16 kHz (quality loss) ✗ | 48 kHz (full quality) ✓ |
| **Playback upsamples** | 16→48 kHz (artifacts) ✗ | 48→48 kHz (clean) ✓ |
| **ASR gets** | 16 kHz (OK, but could be better) | Can receive 48 kHz and downsample as needed ✓ |
| **Diagnostic log** | requested_rate ≠ sample_rate ✗ | requested_rate = sample_rate ✓ |

## The Core Issue in One Sentence

**The configured ASR sample rate (16kHz) is being passed directly to the remote speaker audio source, causing the bridge to resample high-quality 48kHz speaker audio down to 16kHz, losing quality.**

## The Core Fix in One Sentence

**Remote speaker audio should always be captured at its original 48kHz quality; resampling to match ASR requirements should happen separately in the ASR processing pipeline.**
