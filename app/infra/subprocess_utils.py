"""Utilities for spawning subprocesses consistently across the app.

All child Python processes should receive PYTHONUTF8 and PYTHONIOENCODING so
that stdout/stderr are UTF-8 even on Traditional Chinese Windows (CP950).
"""
from __future__ import annotations

import os
import subprocess
import sys


def safe_subprocess_env(base: dict | None = None) -> dict:
    """Return an env dict suitable for subprocess.run / Popen calls.

    Inherits the current environment and ensures UTF-8 I/O variables are set.
    Pass *base* to override specific variables before the UTF-8 defaults are applied.
    """
    env = os.environ.copy()
    if base:
        env.update(base)
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def hidden_subprocess_kwargs() -> dict[str, object]:
    """Return kwargs that hide the subprocess console window on Windows."""
    if sys.platform != "win32":
        return {}
    kwargs: dict[str, object] = {
        "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),
    }
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = 0
    kwargs["startupinfo"] = startupinfo
    return kwargs


__all__ = ["safe_subprocess_env", "hidden_subprocess_kwargs"]
