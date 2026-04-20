"""Quick smoke test for startup_suppress_ms in EndpointingRuntime."""
from app.bootstrap.external_runtime import configure_external_ai_runtime
configure_external_ai_runtime()
from app.infra.asr.endpointing_v2 import EndpointingRuntime, EndpointingDescriptor
from app.infra.config.schema import VadSettings
import numpy as np

cfg = VadSettings(neural_threshold=0.55, min_silence_duration_ms=1000, startup_suppress_ms=3000)
desc = EndpointingDescriptor(name="test", mode="test")
rt = EndpointingRuntime(desc, cfg)

sr = 16000
# 100ms chunk of loud audio (would normally trigger speech)
chunk = np.ones(1600, dtype=np.float32) * 0.5
events = []
for i in range(60):  # 6 seconds
    sig = rt.update(chunk, sr)
    if sig.speech_started:
        events.append(rt._elapsed_ms)

if not events:
    print("FAIL: no speech_started events at all")
else:
    first = events[0]
    if first > 3000:
        print(f"PASS: first speech_started at {first:.0f}ms (after 3000ms suppress)")
    else:
        print(f"FAIL: speech_started at {first:.0f}ms (expected > 3000ms)")

print(f"All speech_started events (ms): {[round(e) for e in events]}")

# Test reset clears elapsed timer
rt.reset()
assert rt._elapsed_ms == 0.0, "reset() should clear _elapsed_ms"
print("PASS: reset() clears _elapsed_ms")
