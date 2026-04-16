# Configure external AI runtimes before test collection so that
# tests importing torch / faster_whisper can find the packages
# even when they are not installed inside .venv.
from app.bootstrap.external_runtime import configure_external_ai_runtime

configure_external_ai_runtime()
