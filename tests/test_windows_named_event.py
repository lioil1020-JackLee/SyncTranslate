from __future__ import annotations

import sys
import uuid

import pytest

from app.infra.audio.windows_named_event import WindowsNamedEvent


@pytest.mark.skipif(sys.platform != "win32", reason="Windows named events are Windows-only")
def test_windows_named_event_can_signal_and_reset() -> None:
    event_name = rf"Local\SyncTranslateTest{uuid.uuid4().hex}"

    with WindowsNamedEvent(event_name) as event:
        assert event.wait(timeout_ms=0) is False
        event.set()
        assert event.wait(timeout_ms=0) is True
        event.reset()
        assert event.wait(timeout_ms=0) is False
