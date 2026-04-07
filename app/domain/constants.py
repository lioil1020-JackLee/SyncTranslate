"""Central string constants for SyncTranslate.

Use these instead of bare string literals to avoid typos and ease refactoring.
New code should import from here; existing code can be migrated incrementally.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# TTS output mode values
# ---------------------------------------------------------------------------
OUTPUT_MODE_TTS = "tts"
OUTPUT_MODE_PASSTHROUGH = "passthrough"
OUTPUT_MODE_SUBTITLE_ONLY = "subtitle_only"

# ---------------------------------------------------------------------------
# Display partial strategy values
# ---------------------------------------------------------------------------
DISPLAY_PARTIAL_ALL = "all"
DISPLAY_PARTIAL_NONE = "none"
DISPLAY_PARTIAL_STABLE_ONLY = "stable_only"

# ---------------------------------------------------------------------------
# ASR language special values
# ---------------------------------------------------------------------------
ASR_LANGUAGE_AUTO = "auto"
ASR_LANGUAGE_NONE = "none"

# ---------------------------------------------------------------------------
# Translation target special values
# ---------------------------------------------------------------------------
TRANSLATION_TARGET_NONE = "none"

# ---------------------------------------------------------------------------
# TTS voice special values
# ---------------------------------------------------------------------------
TTS_VOICE_NONE = "none"

# ---------------------------------------------------------------------------
# Queue overflow drop strategy
# ---------------------------------------------------------------------------
QUEUE_DROP_STRATEGY_DROP_OLDEST = "drop_oldest"

# ---------------------------------------------------------------------------
# Drop reasons (used in diagnostics events)
# ---------------------------------------------------------------------------
DROP_REASON_OVER_LATENCY = "over_latency"
DROP_REASON_QUEUE_OVERFLOW = "queue_overflow"
