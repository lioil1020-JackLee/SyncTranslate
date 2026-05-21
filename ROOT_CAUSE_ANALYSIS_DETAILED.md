# Remote Audio 16kHz Issue - Answers to Your Questions

## Your Question 1: Where does _on_remote_audio_chunk() get its sample_rate parameter from?

### Direct Answer
`_on_remote_audio_chunk()` at [app/application/audio_router.py](app/application/audio_router.py#L266) receives `sample_rate` from:

**VirtualSpeakerSource._emit_chunk()** ([app/infra/audio/sources.py](app/infra/audio/sources.py#L161)):
```python
def _emit_chunk(self, chunk: np.ndarray, sample_rate: float) -> None:
    ...
    for consumer in list(self._consumers):
        consumer(chunk, sample_rate)  # ← Calls _on_remote_audio_chunk(chunk, sample_rate)
```

The `sample_rate` passed to `_emit_chunk()` comes from `_dispatch_audio()` ([app/infra/audio/sources.py](app/infra/audio/sources.py#L130)):
```python
def _dispatch_audio(self, audio: np.ndarray, sample_rate: float) -> None:
    ...
    self._emit_chunk(chunk, sample_rate)  # ← Directly passes received sample_rate
```

### The Source: Bridge Callback
`_dispatch_audio()` is registered as a bridge callback ([app/infra/audio/sources.py](app/infra/audio/sources.py#L110)):
```python
self._bridge.add_remote_input_consumer(self._dispatch_audio)
```

The bridge calls this with: `_dispatch_audio(audio, sample_rate)` where `sample_rate` comes from the remote input polling thread.

### Tracing Back: Where Bridge Gets sample_rate

**Virtual Bridge Client** polls remote input ([app/infra/audio/virtual_bridge_client.py](app/infra/audio/virtual_bridge_client.py#L377-389)):
```python
def _poll_remote_input(self) -> None:
    while not self._remote_input_stop.wait(0.02):
        ...
        payload = self._request({
            "cmd": "read_remote_input",
            "sample_rate": int(self._remote_input_sample_rate),  # ← Requests at this rate
        })
        ...
        audio, sample_rate = decode_audio_packet(packet)
        sample_rate_float = float(sample_rate or self._remote_input_sample_rate or 48000)
        for consumer in consumers:
            consumer(audio.astype(np.float32, copy=False), sample_rate_float)  # ← Sends to _dispatch_audio
```

### **CRITICAL FINDING:** Where _remote_input_sample_rate Gets Set

At [app/infra/audio/virtual_bridge_client.py](app/infra/audio/virtual_bridge_client.py#L94):
```python
def start_remote_input(self, *, sample_rate: int, device_name: str = "", chunk_ms: int = 10) -> None:
    ...
    self._remote_input_sample_rate = int(sample_rate)  # ← This is the value that flows through!
    self._request({
        "cmd": "start_remote_input",
        "sample_rate": int(sample_rate),  # ← Tells bridge to resample to this rate
        "device_name": str(device_name or ""),
        "chunk_ms": int(chunk_ms),
    })
```

### Who Calls start_remote_input()?

**VirtualSpeakerSource.start()** ([app/infra/audio/sources.py](app/infra/audio/sources.py#L77-80)):
```python
def start(self, device_name: str, sample_rate: int, chunk_ms: int) -> None:
    ...
    self._bridge.start_remote_input(
        sample_rate=int(sample_rate),  # ← PASSES THE PARAMETER
        device_name=str(device_name or ""),
        chunk_ms=int(chunk_ms),
    )
```

### Who Calls VirtualSpeakerSource.start()?

**AudioInputManager.start()** ([app/infra/audio/routing.py](app/infra/audio/routing.py#L47-48)):
```python
def start(self, source: str, device_name: str, sample_rate: int, chunk_ms: int) -> None:
    capture = self._capture_of(source)
    capture.start(device_name, sample_rate=sample_rate, chunk_ms=chunk_ms)
```

For remote sources, `self._capture_of("remote")` returns `VirtualSpeakerSource`.

### Who Calls AudioInputManager.start()?

**AudioRouter._reconcile_single_source()** ([app/application/audio_router.py](app/application/audio_router.py#L623-630)):
```python
def _reconcile_single_source(self, *, source: str, capture_needed: bool, asr_needed: bool) -> None:
    ...
    if capture_needed and not running:
        consumer = self._consumer_of(source)
        self._input_manager.add_consumer(source, consumer)
        try:
            self._input_manager.start(
                source,
                self._device_of(source),
                sample_rate=self._sample_rate,  # ← THIS IS THE PROBLEM!
                chunk_ms=self._chunk_ms,
            )
```

### Where AudioRouter.sample_rate Comes From

**AudioRouter.start()** at [app/application/audio_router.py](app/application/audio_router.py#L102-109):
```python
def start(self, mode: str, routes: AudioRouteConfig, sample_rate: int, chunk_ms: int = ASR_DEFAULT_CHUNK_MS) -> None:
    ...
    self._sample_rate = int(sample_rate)  # ← Stores the parameter
```

### Full Chain to Root:

```
1. UI: app/ui/main_window.py:424
   sample_rate=self.config.runtime.sample_rate
   
2. SessionService: app/application/session_service.py:70
   self._audio_router.start(mode, routes, sample_rate, ...)
   
3. AudioRouter.start(): app/application/audio_router.py:109
   self._sample_rate = int(sample_rate)
   
4. AudioRouter._reconcile_single_source(): app/application/audio_router.py:627
   self._input_manager.start(..., sample_rate=self._sample_rate, ...)
   
5. AudioInputManager.start(): app/infra/audio/routing.py:47
   capture.start(..., sample_rate=sample_rate, ...)
   
6. VirtualSpeakerSource.start(): app/infra/audio/sources.py:77
   self._bridge.start_remote_input(sample_rate=int(sample_rate), ...)
   
7. VirtualBridgeClient.start_remote_input(): app/infra/audio/virtual_bridge_client.py:94
   self._remote_input_sample_rate = int(sample_rate)
   
8. VirtualBridgeClient._poll_remote_input(): app/infra/audio/virtual_bridge_client.py:380, 389
   Tells bridge to send at self._remote_input_sample_rate
   Receives sample_rate_float from bridge response
   
9. VirtualBridgeClient callback → VirtualSpeakerSource._dispatch_audio(): app/infra/audio/sources.py:130
   self._emit_chunk(chunk, sample_rate)
   
10. VirtualSpeakerSource._emit_chunk(): app/infra/audio/sources.py:161
    consumer(chunk, sample_rate)
    
11. AudioRouter._on_remote_audio_chunk(): app/application/audio_router.py:266
    Receives (chunk, sample_rate) with WRONG sample_rate
```

---

## Your Question 2: Is there resampling happening in the audio input flow for remote audio?

### Direct Answer
**YES, resampling happens at the BRIDGE LEVEL**, not in the Python app code itself.

### Where Resampling Occurs

The bridge (not the Python app) performs resampling in response to the requested sample rate:

**Virtual Bridge Client requests resampling** ([app/infra/audio/virtual_bridge_client.py](app/infra/audio/virtual_bridge_client.py#L380)):
```python
payload = self._request({
    "cmd": "read_remote_input",
    "sample_rate": int(self._remote_input_sample_rate),  # ← Bridge resamples to this rate
})
```

When the bridge receives this request with a sample rate different from the source audio (48kHz speaker), it resamples:
- **Source audio rate:** 48kHz (from Voicemeeter/speaker)
- **Requested rate:** Could be 48kHz or 16kHz (depending on config)
- **Bridge does:** Resample 48kHz → 16kHz when `_remote_input_sample_rate = 16000`

### No Python-level Resampling for Remote Audio

I searched the codebase and found **NO resampling** in the Python layer for remote audio input:
- `VirtualSpeakerSource._dispatch_audio()` does NOT resample
- `AudioRouter._on_remote_audio_chunk()` does NOT resample
- `TTS.submit_passthrough()` does NOT resample

The audio flows through as-is with the sample rate provided by the bridge.

### ASR Gets Resampling Separately

ASR processing may have its own resampling layer (in the ASR manager), but that's separate from the audio input path.

---

## Your Question 3: How does InputManager handle sample_rate when calling add_consumer?

### Direct Answer
InputManager does **NOT** do anything with sample_rate when calling add_consumer. It just passes the consumer through.

### InputManager.add_consumer() Code
([app/infra/audio/routing.py](app/infra/audio/routing.py#L56-57)):
```python
def add_consumer(self, source: str, consumer: Callable[[np.ndarray, float], None]) -> None:
    self._capture_of(source).add_consumer(consumer)
```

It just delegates to the underlying capture/source object (VirtualSpeakerSource for remote).

### VirtualSpeakerSource.add_consumer() Code
([app/infra/audio/sources.py](app/infra/audio/sources.py#L107-111)):
```python
def add_consumer(self, consumer: AudioConsumer) -> None:
    import logging
    logger = logging.getLogger(__name__)
    
    if consumer not in self._consumers:
        self._consumers.append(consumer)
        logger.info(f"VirtualSpeakerSource.add_consumer: total_consumers={len(self._consumers)}")
    
    # Always register bridge callback to ensure remote input thread is started/maintained
    logger.debug("Registering bridge remote_input_consumer callback")
    self._bridge.add_remote_input_consumer(self._dispatch_audio)
```

**Key point:** The sample_rate is **NOT** handled here at all. It was already set in `start()`. The `add_consumer()` just:
1. Adds the consumer to a list
2. Registers the internal bridge callback `_dispatch_audio`

When `_dispatch_audio` is called later by the bridge, it receives whatever sample_rate the bridge sends.

### Sample Rate Is Set Earlier in start()
([app/infra/audio/sources.py](app/infra/audio/sources.py#L63-82)):
```python
def start(self, device_name: str, sample_rate: int, chunk_ms: int) -> None:
    ...
    self._bridge.start_remote_input(
        sample_rate=int(sample_rate),
        device_name=str(device_name or ""),
        chunk_ms=int(chunk_ms),
    )
    self._sample_rate = int(sample_rate)
    ...
```

So the `start()` method tells the bridge what sample rate to use, and then `add_consumer()` just registers listeners for the audio that will arrive at that sample rate.

---

## Your Question 4: Is there a mismatch between VirtualSpeakerSource output and what _on_remote_audio_chunk() receives?

### Direct Answer
**NO mismatch** - they both correctly pass through what the bridge sends. The problem is **upstream**: the bridge is being TOLD to send 16kHz instead of 48kHz.

### What VirtualSpeakerSource Outputs

The `_emit_chunk()` method outputs whatever sample rate it received:

([app/infra/audio/sources.py](app/infra/audio/sources.py#L155-161)):
```python
def _emit_chunk(self, chunk: np.ndarray, sample_rate: float) -> None:
    self._frame_count += int(chunk.shape[0]) if chunk.ndim else int(chunk.size)
    self._level = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
    for consumer in list(self._consumers):
        consumer(chunk, sample_rate)  # ← Passes the sample_rate it received
```

### What _on_remote_audio_chunk() Receives

([app/application/audio_router.py](app/application/audio_router.py#L266-272)):
```python
def _on_remote_audio_chunk(self, chunk: np.ndarray, sample_rate: float) -> None:
    self._handle_source_audio_chunk(
        source="remote",
        passthrough_channel="local",
        chunk=chunk,
        sample_rate=sample_rate,  # ← Uses exactly what was passed
    )
```

### Where the Real Mismatch Is

The mismatch is at a **higher level**:

| Component | Expected | Actual | Why |
|-----------|----------|--------|-----|
| VirtualSpeakerSource.start() call | sample_rate=48000 | sample_rate=16000 (from config) | AudioRouter._sample_rate is used for both ASR and remote |
| Bridge request | Request 48kHz speaker audio | Request 16kHz (resampled) | VirtualBridgeClient._remote_input_sample_rate set to 16000 |
| Audio arriving at bridge callback | 48kHz quality | 16kHz (resampled from 48kHz) | Bridge resampled per request |
| _dispatch_audio receives | 48kHz | 16kHz | Bridge sent resampled audio |
| _on_remote_audio_chunk receives | 48kHz | 16kHz | VirtualSpeakerSource just passes through |

**The pipeline is internally consistent**, but it's being fed the wrong configuration upstream.

---

## Summary: The Root Cause

When `config.runtime.sample_rate = 16000` (for ASR optimization):

1. AudioRouter stores `_sample_rate = 16000`
2. Both local AND remote inputs are told to use 16kHz
3. VirtualSpeakerSource asks bridge for 16kHz remote input
4. Bridge resamples 48kHz speaker audio → 16kHz
5. Resampled 16kHz audio flows through the pipeline
6. Passthrough outputs 16kHz instead of original 48kHz

**The fix:** Don't pass `self._sample_rate` to VirtualSpeakerSource. Instead:
- Always request 48kHz from the bridge (original speaker quality)
- Let ASR handle its own resampling if needed
- Preserve original quality for passthrough output
