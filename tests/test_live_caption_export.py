from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.ui.pages.live_caption_page import LiveCaptionPage


class _QtTestCase(unittest.TestCase):
    _app: QApplication | None = None

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._app = QApplication.instance() or QApplication([])


class LiveCaptionExportTests(_QtTestCase):
    def test_export_editor_writes_panel_text_to_selected_file(self) -> None:
        page = LiveCaptionPage()
        page.remote_original.setPlainText("[final] line two\n[final] line one")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "remote_original.txt"
            with patch(
                "app.ui.pages.live_caption_page.QFileDialog.getSaveFileName",
                return_value=(str(path), "文字檔 (*.txt)"),
            ):
                page._export_editor(page.remote_original, "remote_original")

            self.assertTrue(path.exists())
            self.assertEqual(path.read_text(encoding="utf-8"), "line one\nline two")

    def test_export_editor_keeps_filesystem_untouched_when_user_cancels(self) -> None:
        page = LiveCaptionPage()
        page.local_translated.setPlainText("translated text")

        with tempfile.TemporaryDirectory() as tmp:
            existing_files = set(Path(tmp).iterdir())
            with patch(
                "app.ui.pages.live_caption_page.QFileDialog.getSaveFileName",
                return_value=("", ""),
            ):
                page._export_editor(page.local_translated, "local_translated")

            self.assertEqual(existing_files, set(Path(tmp).iterdir()))

    def test_export_all_four_panels_preserves_mixed_partial_and_final_content(self) -> None:
        page = LiveCaptionPage()

        page.set_remote_original_lines(["[partial] remote hello"])
        page.set_remote_original_lines(["[final] remote hello"])
        page.set_remote_original_lines(["[final] remote hello", "[partial] remote next"])

        page.set_remote_translated_lines(["[partial] 遠端翻譯一"])
        page.set_remote_translated_lines(["[final] 遠端翻譯一"])
        page.set_remote_translated_lines(["[final] 遠端翻譯一", "[final] 遠端翻譯二"])

        page.set_local_original_lines(["[partial] local draft"])
        page.set_local_translated_lines(["[final] 本地翻譯完成"])

        with tempfile.TemporaryDirectory() as tmp:
            paths = {
                "remote_original": Path(tmp) / "remote_original.txt",
                "remote_translated": Path(tmp) / "remote_translated.txt",
                "local_original": Path(tmp) / "local_original.txt",
                "local_translated": Path(tmp) / "local_translated.txt",
            }

            for panel_name, editor in (
                ("remote_original", page.remote_original),
                ("remote_translated", page.remote_translated),
                ("local_original", page.local_original),
                ("local_translated", page.local_translated),
            ):
                with patch(
                    "app.ui.pages.live_caption_page.QFileDialog.getSaveFileName",
                    return_value=(str(paths[panel_name]), "文字檔 (*.txt)"),
                ):
                    page._export_editor(editor, panel_name)

            self.assertEqual(
                paths["remote_original"].read_text(encoding="utf-8"),
                "remote hello\n[partial] remote next",
            )
            self.assertEqual(
                paths["remote_translated"].read_text(encoding="utf-8"),
                "遠端翻譯二\n遠端翻譯一",
            )
            self.assertEqual(
                paths["local_original"].read_text(encoding="utf-8"),
                "[partial] local draft",
            )
            self.assertEqual(
                paths["local_translated"].read_text(encoding="utf-8"),
                "本地翻譯完成",
            )


if __name__ == "__main__":
    unittest.main()
