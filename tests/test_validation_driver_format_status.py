from app.infra.audio.virtual_devices import VirtualAudioEndpoint, VirtualAudioInstallStatus
from tools.validation import preflight_release_check
from tools.validation.wasapi_endpoint_format import WasapiEndpointFormat, WasapiFormatReport
from tools.validation.common import driver_format_status_item, inspect_wdk_environment


def _status(*, speaker_rate: float = 48000.0, mic_rate: float = 48000.0, speaker_channels: int = 2, mic_channels: int = 2) -> VirtualAudioInstallStatus:
    speaker = VirtualAudioEndpoint(1, "SyncTranslate Virtual Speaker", "Windows WASAPI", 0, speaker_channels, speaker_rate)
    mic = VirtualAudioEndpoint(2, "SyncTranslate Virtual Microphone", "Windows WASAPI", mic_channels, 0, mic_rate)
    return VirtualAudioInstallStatus(True, True, True, 1, 2, speaker.name, mic.name, (speaker,), (mic,))


def _wasapi_report(status: str, *, dtype: str = "PCM16", bits: int = 16) -> WasapiFormatReport:
    return WasapiFormatReport(
        status,
        "format status",
        {"summary": "48000Hz PCM16 2ch"},
        [
            WasapiEndpointFormat("render", "SyncTranslate Virtual Speaker", 48000, 2, bits, bits, 0xFFFE, "", dtype),
            WasapiEndpointFormat("capture", "SyncTranslate Virtual Microphone", 48000, 2, bits, bits, 0xFFFE, "", dtype),
        ],
    )


def _wasapi_report_with_float_shared_mix() -> WasapiFormatReport:
    return WasapiFormatReport(
        "PASS",
        "device format status",
        {"summary": "48000Hz PCM16 2ch"},
        [
            WasapiEndpointFormat("render", "SyncTranslate Virtual Speaker", 48000, 2, 16, 16, 0xFFFE, "", "PCM16"),
            WasapiEndpointFormat("capture", "SyncTranslate Virtual Microphone", 48000, 2, 16, 16, 0xFFFE, "", "PCM16"),
        ],
        shared_mix_formats=[
            WasapiEndpointFormat("render", "SyncTranslate Virtual Speaker", 48000, 2, 32, 32, 0xFFFE, "", "FLOAT32", "shared_mix_format"),
            WasapiEndpointFormat("capture", "SyncTranslate Virtual Microphone", 48000, 2, 32, 32, 0xFFFE, "", "FLOAT32", "shared_mix_format"),
        ],
    )


def test_driver_format_status_warns_when_bit_depth_is_unknown_but_rate_and_channels_match(monkeypatch) -> None:
    monkeypatch.setattr("tools.validation.common.query_synctranslate_endpoint_formats", lambda: WasapiFormatReport("WARN", "unknown", {}, []))
    item = driver_format_status_item(_status(), required=True)
    assert item.status == "WARN"
    assert item.details["expected"]["summary"] == "48000Hz PCM16 2ch"
    assert item.details["driver_format_status"] == "UNKNOWN"


def test_driver_format_status_passes_when_wasapi_device_format_matches(monkeypatch) -> None:
    monkeypatch.setattr("tools.validation.common.query_synctranslate_endpoint_formats", lambda: _wasapi_report("PASS"))
    item = driver_format_status_item(_status(), required=True)
    assert item.status == "PASS"
    assert item.details["driver_format_status"] == "PASS"


def test_driver_format_status_keeps_float_shared_mix_as_detail_not_failure(monkeypatch) -> None:
    monkeypatch.setattr("tools.validation.common.query_synctranslate_endpoint_formats", _wasapi_report_with_float_shared_mix)
    item = driver_format_status_item(_status(), required=True)
    assert item.status == "PASS"
    assert item.details["driver_format_status"] == "PASS"
    assert item.details["shared_mix_formats"][0]["dtype"] == "FLOAT32"


def test_driver_format_status_fails_when_wasapi_device_format_is_float(monkeypatch) -> None:
    monkeypatch.setattr("tools.validation.common.query_synctranslate_endpoint_formats", lambda: _wasapi_report("FAIL", dtype="FLOAT32", bits=32))
    item = driver_format_status_item(_status(), required=True)
    assert item.status == "FAIL"
    assert item.details["driver_format_status"] == "FAIL"


def test_driver_format_status_fails_on_dialogue_mismatch() -> None:
    item = driver_format_status_item(_status(speaker_rate=44100.0, mic_channels=1), required=True)
    assert item.status == "FAIL"
    assert "speaker_default_samplerate=44100.0" in item.details["mismatches"]
    assert "microphone_channels=1" in item.details["mismatches"]


def test_driver_format_status_fails_for_dialogue_when_ks_interfaces_exist_but_endpoints_are_hidden() -> None:
    status = VirtualAudioInstallStatus(
        installed=True,
        speaker_available=False,
        microphone_available=False,
        speaker_index=-1,
        microphone_index=-1,
        speaker_name="",
        microphone_name="",
        render_endpoints=(),
        capture_endpoints=(),
        speaker_interface_available=True,
        microphone_interface_available=True,
        speaker_interface_count=2,
        microphone_interface_count=2,
    )
    item = driver_format_status_item(status, required=True)
    assert item.status == "FAIL"
    assert item.details["driver_format_status"] == "UNKNOWN"
    assert item.details["reason"] == "ks_interfaces_present_but_user_endpoints_hidden"


def test_no_wdk_environment_inspection_is_graceful() -> None:
    result = inspect_wdk_environment()
    assert result["status"] in {"PASS", "WARN"}
    assert {"msbuild", "infverif", "stampinf", "signtool"} <= set(result["tools"])


def test_preflight_meeting_does_not_fail_because_driver_is_missing(monkeypatch) -> None:
    missing = VirtualAudioInstallStatus(False, False, False, -1, -1, "", "", (), ())
    monkeypatch.setattr(preflight_release_check, "detect_virtual_audio_install", lambda: missing)

    report = preflight_release_check.build_report(mode="meeting")
    driver_items = [item for item in report.items if item.name in {"virtual_driver", "driver_format_status"}]
    assert all(item.status != "FAIL" for item in driver_items)
    assert report.details["driver_format_expected"] == "48000Hz PCM16 2ch"


def test_preflight_dialogue_driver_format_mismatch_is_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        preflight_release_check,
        "detect_virtual_audio_install",
        lambda: _status(speaker_rate=44100.0, mic_channels=1),
    )

    report = preflight_release_check.build_report(mode="dialogue")
    assert any(item.name == "driver_format_status" and item.status == "FAIL" for item in report.items)
    assert "driver_format" in report.details["missing_driver_items"]
