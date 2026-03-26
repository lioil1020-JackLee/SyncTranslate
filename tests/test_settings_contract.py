from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QGroupBox, QTabWidget, QWidget

from app.infra.config.schema import AudioRouteConfig
from app.ui.main_window import MainWindow
from app.ui.pages.settings_page import SettingsPage


class _QtTestCase(unittest.TestCase):
    _app: QApplication | None = None

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._app = QApplication.instance() or QApplication([])


class SettingsContractTests(_QtTestCase):
    def test_settings_page_is_single_scroll_page_without_nested_tabs(self) -> None:
        page = SettingsPage(QWidget(), QWidget(), QWidget())

        self.assertEqual(len(page.findChildren(QTabWidget)), 0)
        section_titles = {group.title() for group in page.findChildren(QGroupBox)}
        self.assertIn("音訊裝置", section_titles)
        self.assertIn("翻譯與輸出", section_titles)
        self.assertIn("儲存與診斷", section_titles)
        self.assertEqual(page._scroll.horizontalScrollBarPolicy(), Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def test_route_validation_reports_missing_and_unavailable_devices(self) -> None:
        fake_window = type(
            "FakeWindow",
            (),
            {
                "input_device_names": {"USB Mic"},
                "output_device_names": {"Desk Speaker"},
            },
        )()
        route = AudioRouteConfig(
            meeting_in="USB Mic",
            microphone_in="",
            speaker_out="Desk Speaker",
            meeting_out="Missing Headset",
        )

        message, has_error = MainWindow._describe_route_validation(fake_window, route)

        self.assertTrue(has_error)
        self.assertIn("未設定裝置：本地輸入", message)
        self.assertIn("找不到裝置：遠端輸出: Missing Headset", message)


if __name__ == "__main__":
    unittest.main()
