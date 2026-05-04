# Configure external AI runtimes before test collection so that
# tests importing torch / faster_whisper can find the packages
# even when they are not installed inside .venv.
import os

from app.bootstrap.external_runtime import configure_external_ai_runtime

if os.environ.get("SYNC_TRANSLATE_SKIP_EXTERNAL_RUNTIME", "0") != "1":
	configure_external_ai_runtime()
