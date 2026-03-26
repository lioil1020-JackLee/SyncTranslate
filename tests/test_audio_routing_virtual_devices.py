from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.infra.audio.device_registry import DeviceInfo, DeviceManager, encode_device_selector, parse_device_selector
from app.ui.pages.audio_routing_page import AudioRoutingPage


class _QtTestCase(unittest.TestCase):
    _app: QApplication | None = None

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._app = QApplication.instance() or QApplication([])


def _device(
    *,
    index: int,
    name: str,
    hostapi_name: str = "Windows WASAPI",
    input_channels: int = 2,
    output_channels: int = 2,
) -> DeviceInfo:
    return DeviceInfo(
        index=index,
        name=name,
        hostapi_index=0,
        hostapi_name=hostapi_name,
        hostapi_label="WDM (WASAPI)",
        max_input_channels=input_channels,
        max_output_channels=output_channels,
        default_samplerate=48000.0,
    )


class AudioRoutingVirtualDeviceTests(_QtTestCase):
    def test_selector_round_trip_preserves_virtual_device_hostapi(self) -> None:
        selector = encode_device_selector(
            hostapi_name="Windows WASAPI",
            device_name="Voicemeeter Out B2 (VB-Audio Voicemeeter VAIO)",
        )

        hostapi_name, device_name = parse_device_selector(selector)

        self.assertEqual(hostapi_name, "Windows WASAPI")
        self.assertEqual(device_name, "Voicemeeter Out B2 (VB-Audio Voicemeeter VAIO)")

    def test_apply_config_matches_plain_device_name_to_encoded_virtual_selector(self) -> None:
        page = AudioRoutingPage(on_route_changed=lambda: None)
        inputs = [
            _device(index=0, name="Voicemeeter Out B2 (VB-Audio Voicemeeter VAIO)", output_channels=0),
            _device(index=1, name="USB Mic", output_channels=0),
        ]
        outputs = [
            _device(index=2, name="Voicemeeter Input (VB-Audio Voicemeeter VAIO)", input_channels=0),
            _device(index=3, name="Desk Speaker", input_channels=0),
        ]

        page.set_devices(inputs, outputs)
        plain_name = "Voicemeeter Out B2 (VB-Audio Voicemeeter VAIO)"
        page._apply_selector_to_route(
            page.meeting_in_hostapi_combo,
            page.meeting_in_combo,
            inputs,
            plain_name,
        )

        selected = page.selected_audio_routes().meeting_in
        self.assertEqual(
            selected,
            encode_device_selector(
                hostapi_name="Windows WASAPI",
                device_name="Voicemeeter Out B2 (VB-Audio Voicemeeter VAIO)",
            ),
        )

    def test_hostapi_filter_keeps_virtual_output_choices_grouped_under_selected_api(self) -> None:
        page = AudioRoutingPage(on_route_changed=lambda: None)
        outputs = [
            _device(
                index=0,
                name="Voicemeeter Input (VB-Audio Voicemeeter VAIO)",
                input_channels=0,
                hostapi_name="Windows WASAPI",
            ),
            _device(
                index=1,
                name="Voicemeeter Aux Input (VB-Audio Voicemeeter AUX VAIO)",
                input_channels=0,
                hostapi_name="Windows WASAPI",
            ),
            _device(
                index=2,
                name="Laptop Speaker",
                input_channels=0,
                hostapi_name="MME",
            ),
        ]

        page.set_devices([], outputs)
        index = page.speaker_out_hostapi_combo.findData("Windows WASAPI")
        self.assertGreaterEqual(index, 0)
        page.speaker_out_hostapi_combo.setCurrentIndex(index)
        page._refresh_speaker_out_combo()

        items = {
            page.speaker_out_combo.itemData(i)
            for i in range(page.speaker_out_combo.count())
        }
        self.assertIn(
            encode_device_selector(
                hostapi_name="Windows WASAPI",
                device_name="Voicemeeter Input (VB-Audio Voicemeeter VAIO)",
            ),
            items,
        )
        self.assertIn(
            encode_device_selector(
                hostapi_name="Windows WASAPI",
                device_name="Voicemeeter Aux Input (VB-Audio Voicemeeter AUX VAIO)",
            ),
            items,
        )
        self.assertNotIn(
            encode_device_selector(hostapi_name="MME", device_name="Laptop Speaker"),
            items,
        )

    @patch("app.infra.audio.device_registry.hostapi_name_by_index", return_value="Windows WASAPI")
    @patch("app.infra.audio.device_registry.list_indexed_devices")
    def test_device_manager_collapses_duplicate_registry_rows(self, mock_list_indexed_devices, _mock_hostapi) -> None:
        mock_list_indexed_devices.return_value = [
            (
                0,
                {
                    "name": "Desk Speaker",
                    "hostapi": 0,
                    "max_input_channels": 0,
                    "max_output_channels": 2,
                    "default_samplerate": 48000.0,
                },
            ),
            (
                1,
                {
                    "name": "Desk Speaker",
                    "hostapi": 0,
                    "max_input_channels": 0,
                    "max_output_channels": 2,
                    "default_samplerate": 48000.0,
                },
            ),
        ]

        devices = DeviceManager().list_output_devices()

        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0].name, "Desk Speaker")


if __name__ == "__main__":
    unittest.main()
