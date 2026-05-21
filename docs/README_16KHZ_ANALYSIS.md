# Remote Audio 16kHz Passthrough Issue - Complete Analysis Index

## Quick Summary

**Problem:** Remote audio passthrough is sampled at 16kHz instead of 48kHz because the configured ASR sample rate parameter (`config.runtime.sample_rate`) is being globally applied to remote speaker audio capture.

**Root Cause:** `AudioRouter._sample_rate` (set from user config) is passed to both local AND remote sources. For remote sources, this causes `VirtualSpeakerSource` to tell the bridge "send me remote input at 16kHz", which resamples the original 48kHz speaker audio down to 16kHz before transmission.

**Diagnostic Evidence:** Logs show `passthrough_playback_started: sample_rate=48000.0 requested_rate=16000.0` - the audio is being requested at 16kHz but output device is 48kHz, requiring upsampling.

---

## Documentation Files (In This Directory)

### 1. **ROOT_CAUSE_ANALYSIS_DETAILED.md** ⭐ START HERE
- **For:** Understanding exactly where the bug is
- **Contains:** 
  - Direct answer to each of your 4 questions
  - Complete call chain from UI to bridge
  - Code locations with line numbers
  - Proof of where sample_rate originates

### 2. **DEBUGGING_16KHZ_ISSUE.md**
- **For:** Executive summary and overview
- **Contains:**
  - Problem statement
  - Root cause chain (simplified)
  - Why the bug exists (architecture issue)
  - What should happen vs what does happen

### 3. **DATA_FLOW_DIAGRAM.md**
- **For:** Visual understanding of the data flow
- **Contains:**
  - ASCII flow diagram showing problem scenario
  - Step-by-step path from config to output
  - Comparison with correct flow
  - Key differences table

### 4. **RECOMMENDED_FIXES.md** ⭐ IMPLEMENTATION GUIDE
- **For:** Choosing and implementing a solution
- **Contains:**
  - 3 different fix options with code changes
  - Solution 1: Quick fix (hardcode 48kHz)
  - Solution 2: Best fix (separate config parameter)
  - Solution 3: Smart heuristic
  - Testing strategy
  - Backward compatibility notes

---

## Answer Summary to Your Questions

### ❓ Question 1: Where does _on_remote_audio_chunk() get its sample_rate from?

**Answer:** From the bridge, via this chain:
1. `app/ui/main_window.py:424` - config.runtime.sample_rate
2. `app/application/audio_router.py:109` - stored in self._sample_rate
3. `app/application/audio_router.py:627` - passed to input manager
4. `app/infra/audio/sources.py:77` - VirtualSpeakerSource passes to bridge
5. `app/infra/audio/virtual_bridge_client.py:94` - stored as _remote_input_sample_rate
6. `app/infra/audio/virtual_bridge_client.py:380` - sent in bridge request
7. `app/infra/audio/virtual_bridge_client.py:389` - received back from bridge
8. `app/infra/audio/sources.py:161` - emitted to _on_remote_audio_chunk()

**See:** ROOT_CAUSE_ANALYSIS_DETAILED.md > "Your Question 1" for full trace

---

### ❓ Question 2: Is there resampling in the audio input flow?

**Answer:** **YES, but at the BRIDGE level**, not in Python code.

- The bridge receives request: "send me remote input at 16kHz" (from VirtualBridgeClient)
- Bridge resamples: 48kHz speaker audio → 16kHz
- Python receives already-resampled audio

**See:** ROOT_CAUSE_ANALYSIS_DETAILED.md > "Your Question 2" for detailed explanation

---

### ❓ Question 3: How does InputManager handle sample_rate?

**Answer:** It **doesn't do anything** with sample_rate in add_consumer(). The sample_rate was already set in start().

- `AudioInputManager.add_consumer()` just delegates to the source
- `VirtualSpeakerSource.add_consumer()` just registers listeners
- The sample_rate is already configured in the prior start() call

**See:** ROOT_CAUSE_ANALYSIS_DETAILED.md > "Your Question 3"

---

### ❓ Question 4: Mismatch between VirtualSpeakerSource output and _on_remote_audio_chunk() input?

**Answer:** **NO mismatch** - both correctly pass through what the bridge sends. The problem is **upstream**: the bridge is being told to send 16kHz instead of 48kHz.

- VirtualSpeakerSource._emit_chunk() passes exactly what it received
- _on_remote_audio_chunk() receives exactly what was emitted
- The real mismatch is at the config level: wrong sample_rate is being passed to VirtualSpeakerSource.start()

**See:** ROOT_CAUSE_ANALYSIS_DETAILED.md > "Your Question 4"

---

## Code Location Reference

| Component | File | Line | Issue |
|-----------|------|------|-------|
| **UI** | `app/ui/main_window.py` | 424 | Reads config.runtime.sample_rate (16000) |
| **Session Service** | `app/application/session_service.py` | 70 | Passes sample_rate to AudioRouter |
| **AudioRouter.start()** | `app/application/audio_router.py` | 109 | Stores sample_rate in self._sample_rate |
| **AudioRouter._reconcile** | `app/application/audio_router.py` | 627 | **[KEY]** Passes same rate to all sources |
| **AudioInputManager** | `app/infra/audio/routing.py` | 47, 53 | Routes to VirtualSpeakerSource |
| **VirtualSpeakerSource.start()** | `app/infra/audio/sources.py` | 77 | **[ROOT CAUSE]** Passes to bridge.start_remote_input() |
| **VirtualBridgeClient.start_remote_input()** | `app/infra/audio/virtual_bridge_client.py` | 94 | **[KEY]** Sets _remote_input_sample_rate = 16000 |
| **VirtualBridgeClient._poll_remote_input()** | `app/infra/audio/virtual_bridge_client.py` | 380, 389 | Requests and receives at 16kHz |
| **VirtualSpeakerSource._dispatch_audio()** | `app/infra/audio/sources.py` | 130 | Passes 16kHz to _emit_chunk() |
| **VirtualSpeakerSource._emit_chunk()** | `app/infra/audio/sources.py` | 161 | Calls consumer with 16kHz |
| **AudioRouter._on_remote_audio_chunk()** | `app/application/audio_router.py` | 266 | Receives 16kHz audio |
| **TTS.submit_passthrough()** | `app/infra/tts/playback_queue.py` | 319 | Gets 16kHz audio |

---

## Fix Recommendations

### If You Need Quick Fix (< 1 hour):
**→ Use SOLUTION 1** from RECOMMENDED_FIXES.md
- Hardcode 48kHz in [app/infra/audio/sources.py](app/infra/audio/sources.py#L77)
- Single file change
- Solves the problem immediately
- Can be refactored later

### If You Have Time for Proper Fix (< 1 day):
**→ Use SOLUTION 2** from RECOMMENDED_FIXES.md
- Add separate `remote_sample_rate` config parameter
- Update AudioRouter, SessionService, UI to pass through
- Better architecture
- Fully configurable
- Recommended long-term solution

### Testing the Fix:
- Look for diagnostic log: `passthrough_playback_started: sample_rate=48000.0 requested_rate=48000.0`
- If `requested_rate` ≠ `sample_rate`, the issue persists
- See RECOMMENDED_FIXES.md for unit and integration tests

---

## Repository Memory

Added to `/memories/repo/asr_tts_routing_notes.md`:
- Bug description with root cause chain
- All key file locations
- Evidence (diagnostic log pattern)
- Recommended fix approach

---

## Session Memory

Created `/memories/session/remote_audio_16khz_root_cause.md`:
- Problem statement with diagnostic log
- Root cause chain (numbered steps 1-9)
- Why the issue exists
- What the fix requires

---

## Architecture Issue Explanation

The system was designed with a single `sample_rate` parameter for all audio processing:
- **Intended use:** ASR optimization parameter (16kHz)
- **Actual impact:** Applied globally to all sources
- **Should be:** Separate parameters for ASR (16kHz) and remote passthrough (48kHz)

### Current (Broken):
```
config.runtime.sample_rate = 16000
  ├─ Local mic → 16kHz ✓
  ├─ Local ASR → 16kHz ✓
  ├─ Remote speaker → 16kHz ✗ (Should be 48kHz!)
  ├─ Remote ASR → 16kHz ✓
  └─ Passthrough → 16kHz ✗ (Should be 48kHz!)
```

### Should Be:
```
config.runtime.sample_rate = 16000 (ASR rate)
config.runtime.remote_sample_rate = 48000 (NEW - speaker rate)
  ├─ Local mic → 16kHz ✓
  ├─ Local ASR → 16kHz ✓
  ├─ Remote speaker → 48kHz ✓
  ├─ Remote ASR → May receive 48kHz and downsample ✓
  └─ Passthrough → 48kHz ✓
```

---

## Key Insights

1. **The bridge is the culprit**: It resamples audio based on the requested sample rate
2. **The bridge request comes from Python**: VirtualSpeakerSource asks for 16kHz
3. **VirtualSpeakerSource was passed the wrong rate**: From AudioRouter._sample_rate
4. **AudioRouter used the wrong rate**: From config.runtime.sample_rate (ASR parameter)
5. **Architecture issue**: Single sample_rate parameter used for multiple purposes
6. **Quality impact**: 48kHz speaker audio → resampled to 16kHz → upsampled for output = quality loss

---

## Next Steps

1. **Read ROOT_CAUSE_ANALYSIS_DETAILED.md** - Understand the full chain
2. **Review DATA_FLOW_DIAGRAM.md** - Visualize the problem
3. **Choose fix from RECOMMENDED_FIXES.md** - Pick solution based on timeline
4. **Implement and test** - Add unit tests from RECOMMENDED_FIXES.md
5. **Verify in logs** - Check diagnostic output shows requested_rate = sample_rate

---

## Questions?

- **For understanding the bug:** See ROOT_CAUSE_ANALYSIS_DETAILED.md
- **For visualization:** See DATA_FLOW_DIAGRAM.md
- **For implementation:** See RECOMMENDED_FIXES.md
- **For summary:** See DEBUGGING_16KHZ_ISSUE.md

All documents use exact file paths and line numbers for easy reference.
