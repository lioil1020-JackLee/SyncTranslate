from __future__ import annotations

import unittest

from app.infra.asr.stream_worker import _format_asr_exception_message


class AsrErrorFormattingTests(unittest.TestCase):
    def test_ssl_import_failure_message_contains_recovery_hint(self) -> None:
        exc = RuntimeError("DLL load failed while importing _ssl: 找不到指定的程式。")

        message = _format_asr_exception_message(exc)

        self.assertIn("_internal/libssl-3-x64.dll", message)
        self.assertIn("_internal/libcrypto-3-x64.dll", message)
        self.assertIn("dist/SyncTranslate-onedir/SyncTranslate.exe", message)

    def test_non_ssl_error_message_keeps_original_text(self) -> None:
        exc = RuntimeError("cuda out of memory")

        message = _format_asr_exception_message(exc)

        self.assertEqual(message, "cuda out of memory")


if __name__ == "__main__":
    unittest.main()
