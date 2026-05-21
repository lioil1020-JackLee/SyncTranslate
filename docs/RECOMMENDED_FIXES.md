# Recommended Fixes for 16kHz Remote Audio Issue

## The Problem (Summary)

Remote audio passthrough receives audio at 16kHz instead of 48kHz because:
1. `config.runtime.sample_rate` (an ASR optimization parameter) is applied globally
2. This low sample rate is passed to `VirtualSpeakerSource.start()`
3. `VirtualSpeakerSource` tells the bridge to send remote input at 16kHz
4. The bridge resamples 48kHz speaker audio down to 16kHz
5. Quality is lost, and passthrough receives degraded audio

## Three Recommended Solutions

### Solution 1: Quick Fix - Hardcode 48kHz for Remote Audio ⭐ RECOMMENDED (Simplest)

**File:** [app/infra/audio/sources.py](app/infra/audio/sources.py#L77-80)

**Change:** Always request 48kHz from the bridge, ignore the passed sample_rate for remote audio.

```python
def start(self, device_name: str, sample_rate: int, chunk_ms: int) -> None:
    import logging
    logger = logging.getLogger(__name__)
    
    if not device_name:
        logger.error("VirtualSpeakerSource.start() called with empty device_name")
        raise ValueError("device_name cannot be empty for VirtualSpeakerSource")
    
    logger.info(
        f"VirtualSpeakerSource.start: device_name={device_name!r}, "
        f"sample_rate={sample_rate}, chunk_ms={chunk_ms}"
    )
    
    try:
        # FIX: Always use 48kHz for remote speaker audio, regardless of ASR config
        remote_sample_rate = 48000  # Remote speaker is always 48kHz, not ASR rate
        
        self._bridge.start_remote_input(
            sample_rate=int(remote_sample_rate),  # Changed from sample_rate to remote_sample_rate
            device_name=str(device_name or ""),
            chunk_ms=int(chunk_ms),
        )
        self._sample_rate = int(remote_sample_rate)  # Also update internal rate
        self._chunk_frames = max(1, int(self._sample_rate * max(5, int(chunk_ms)) / 1000))
        self._pending = np.zeros((0, 1), dtype=np.float32)
        self._running = True
        self._last_error = ""
    except Exception as exc:
        self._running = False
        self._last_error = str(exc)
        logger.exception(f"VirtualSpeakerSource.start() failed: {exc}")
        raise
```

**Pros:**
- Minimal code change
- Solves the immediate problem
- Remote audio always at maximum quality

**Cons:**
- Hardcoded value not configurable
- Doesn't address architectural issue

---

### Solution 2: Add Separate Remote Sample Rate Configuration ⭐ RECOMMENDED (Best)

**File:** [app/infra/config/schema.py](app/infra/config/schema.py#L275-320)

**Change 1:** Add new config parameter to RuntimeConfig:

```python
class RuntimeConfig:
    asr_pipeline: str = "v2"
    asr_v2_backend: str = "faster_whisper_v2"
    asr_v2_endpointing: str = "neural_endpoint"
    sample_rate: int = 48000  # ASR processing sample rate (16kHz for optimization, 48kHz for quality)
    remote_sample_rate: int = 48000  # ADD THIS: Remote speaker audio sample rate (always 48kHz)
    chunk_ms: int = ASR_DEFAULT_CHUNK_MS
    # ... rest of config
```

**File:** [app/application/audio_router.py](app/application/audio_router.py#L102)

**Change 2:** Update AudioRouter to accept remote_sample_rate:

```python
def start(self, mode: str, routes: AudioRouteConfig, sample_rate: int, 
          remote_sample_rate: int = 48000,  # NEW PARAMETER
          chunk_ms: int = ASR_DEFAULT_CHUNK_MS) -> None:
    self.stop()
    self._state.start_session()
    self._tts_manager.start()
    self._mode = self._normalize_mode(mode)
    self._routes = routes
    self._sample_rate = int(sample_rate)
    self._remote_sample_rate = int(remote_sample_rate)  # NEW: Store separately
    self._chunk_ms = int(chunk_ms)
    # ... rest of method
```

**File:** [app/application/audio_router.py](app/application/audio_router.py#L610-630)

**Change 3:** Use remote_sample_rate for remote sources:

```python
def _reconcile_single_source(self, *, source: str, capture_needed: bool, asr_needed: bool) -> None:
    # ... existing code ...
    
    running = self._capture_running.get(source, False)
    if capture_needed and not running:
        consumer = self._consumer_of(source)
        self._input_manager.add_consumer(source, consumer)
        try:
            # Use different sample rate for remote sources
            source_sample_rate = self._remote_sample_rate if source == "remote" else self._sample_rate  # NEW
            
            self._input_manager.start(
                source,
                self._device_of(source),
                sample_rate=source_sample_rate,  # Changed
                chunk_ms=self._chunk_ms,
            )
        except Exception:
            try:
                self._input_manager.remove_consumer(source, consumer)
            except Exception:
                pass
            raise
        self._capture_running[source] = True
    # ... rest of method
```

**File:** [app/application/session_service.py](app/application/session_service.py#L56-70)

**Change 4:** Pass remote_sample_rate through the call chain:

```python
def start(
    self,
    routes: AudioRouteConfig,
    sample_rate: int,
    chunk_ms: int = ASR_DEFAULT_CHUNK_MS,
    *,
    mode: str = "bidirectional",
    remote_sample_rate: int = 48000,  # NEW PARAMETER
) -> SessionResult:
    # ... validation code ...
    try:
        self._audio_router.start(mode, routes, sample_rate, 
                                 remote_sample_rate=remote_sample_rate,  # NEW
                                 chunk_ms=chunk_ms)
```

**File:** [app/ui/main_window.py](app/ui/main_window.py#L1270-1280)

**Change 5:** Pass remote_sample_rate from config:

```python
def _worker() -> None:
    try:
        if action == "stop":
            result = self.session_controller.stop()
        else:
            if route is None or sample_rate is None:
                raise ValueError("missing session start parameters")
            # Get remote_sample_rate from config (defaults to 48000)
            remote_sample_rate = getattr(self.config.runtime, 'remote_sample_rate', 48000)
            
            result = self.session_controller.start(
                route,
                sample_rate=sample_rate,
                remote_sample_rate=remote_sample_rate,  # NEW
                chunk_ms=chunk_ms or self.config.runtime.chunk_ms,
                mode=str(mode or getattr(self.config.direction, "mode", "bidirectional") or "bidirectional"),
            )
        self._session_action_queue.put((action, True, result))
    except Exception as exc:
        self._session_action_queue.put((action, False, exc))
```

**Pros:**
- Fully configurable
- Addresses architectural issue
- Allows per-configuration tuning
- Clear separation of concerns

**Cons:**
- More code changes
- Requires config migration handling

---

### Solution 3: Dynamic Configuration in start() Method

**File:** [app/infra/audio/sources.py](app/infra/audio/sources.py#L77-80)

**Change:** Detect if this is ASR sample rate and always use 48kHz for remote:

```python
def start(self, device_name: str, sample_rate: int, chunk_ms: int) -> None:
    import logging
    logger = logging.getLogger(__name__)
    
    if not device_name:
        logger.error("VirtualSpeakerSource.start() called with empty device_name")
        raise ValueError("device_name cannot be empty for VirtualSpeakerSource")
    
    logger.info(
        f"VirtualSpeakerSource.start: device_name={device_name!r}, "
        f"sample_rate={sample_rate}, chunk_ms={chunk_ms}"
    )
    
    try:
        # FIX: Remote speaker audio should always be at full quality (48kHz)
        # If sample_rate < 32kHz, it's likely an ASR optimization - ignore for remote
        remote_sample_rate = 48000 if sample_rate < 32000 else sample_rate
        
        logger.info(
            f"VirtualSpeakerSource requesting remote input at {remote_sample_rate}Hz "
            f"(ASR rate was {sample_rate}Hz)"
        )
        
        self._bridge.start_remote_input(
            sample_rate=int(remote_sample_rate),
            device_name=str(device_name or ""),
            chunk_ms=int(chunk_ms),
        )
        self._sample_rate = int(remote_sample_rate)
        self._chunk_frames = max(1, int(self._sample_rate * max(5, int(chunk_ms)) / 1000))
        self._pending = np.zeros((0, 1), dtype=np.float32)
        self._running = True
        self._last_error = ""
    except Exception as exc:
        self._running = False
        self._last_error = str(exc)
        logger.exception(f"VirtualSpeakerSource.start() failed: {exc}")
        raise
```

**Pros:**
- Smart heuristic: detects ASR optimization
- Minimal code change
- No configuration changes needed

**Cons:**
- Magic number (32000) might not always be correct
- Less explicit than separate parameters

---

## Testing the Fix

### 1. Unit Test to Add

**File:** Create or update test file

```python
def test_remote_audio_preserves_48khz_when_asr_optimized_to_16khz():
    """Verify that remote speaker audio stays at 48kHz even when ASR is optimized to 16kHz."""
    mock_bridge = Mock()
    source = VirtualSpeakerSource(mock_bridge)
    
    # ASR optimized to 16kHz
    source.start(device_name="test_device", sample_rate=16000, chunk_ms=10)
    
    # Bridge should be asked for 48kHz (or 48000)
    mock_bridge.start_remote_input.assert_called_once()
    call_args = mock_bridge.start_remote_input.call_args
    
    # The sample_rate in the call should be 48000, not 16000
    assert call_args[1]['sample_rate'] == 48000, \
        f"Expected 48000Hz but got {call_args[1]['sample_rate']}Hz"
```

### 2. Integration Test

```python
def test_remote_passthrough_gets_48khz_with_16khz_asr_config():
    """
    Scenario: User configures sample_rate: 16000 for ASR optimization.
    Expected: Passthrough still receives 48kHz remote audio.
    Evidence: diagnostic log shows requested_rate = 48000 (not 16000).
    """
    # Setup with ASR sample rate of 16kHz
    config = RuntimeConfig(sample_rate=16000)
    router = AudioRouter(...)
    routes = AudioRouteConfig(...)
    
    router.start(mode="bidirectional", routes=routes, sample_rate=16000, chunk_ms=40)
    
    # Simulate bridge sending 48kHz audio
    # (In real scenario, bridge would send at requested rate)
    received_sample_rates = []
    
    def capture_sample_rate(chunk, sample_rate):
        received_sample_rates.append(sample_rate)
    
    # Push simulated 48kHz remote audio
    # ... verification that router received 48kHz, not 16kHz
    
    assert 48000 in received_sample_rates, \
        f"Expected 48000Hz remote audio, got {received_sample_rates}"
```

### 3. Diagnostic Verification

After fix, the diagnostic log should show:
```
passthrough_playback_started: sample_rate=48000.0 channels=2 requested_rate=48000.0
```

(Both values should match at 48000)

Before fix:
```
passthrough_playback_started: sample_rate=48000.0 channels=2 requested_rate=16000.0
```

(Mismatch indicates upsampling/quality loss)

---

## Implementation Order (If Using Solution 2)

1. Add `remote_sample_rate` to [app/infra/config/schema.py](app/infra/config/schema.py)
2. Update [app/infra/audio/sources.py](app/infra/audio/sources.py) to accept remote_sample_rate (as parameter, optional)
3. Update [app/application/audio_router.py](app/application/audio_router.py) to track and use remote_sample_rate
4. Update [app/application/session_service.py](app/application/session_service.py) to pass through
5. Update [app/ui/main_window.py](app/ui/main_window.py) to read from config
6. Add/update tests in [tests/](tests/) directory
7. Update config documentation in [docs/](docs/)

---

## Backward Compatibility Consideration

For Solution 2, ensure config migration:
- Old configs without `remote_sample_rate` should default to 48000
- Existing tests with hardcoded `sample_rate` should still work

Add to config migration logic:
```python
def migrate_runtime_config(old_config):
    # ... existing migrations ...
    if 'remote_sample_rate' not in old_config.get('runtime', {}):
        old_config['runtime']['remote_sample_rate'] = 48000
    return old_config
```

---

## Recommendation

**Go with Solution 2** (separate remote_sample_rate parameter):
- ✅ Solves root cause, not just symptom
- ✅ Fully configurable for future needs
- ✅ Clear architectural separation
- ✅ Easy to understand and maintain
- ✅ Best long-term solution

**Fallback to Solution 1** (hardcode 48kHz) if timeline is tight:
- ✅ Quick fix (minimal changes)
- ✅ Solves the immediate problem
- ✅ Can be refactored to Solution 2 later
