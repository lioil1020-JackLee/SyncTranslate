from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8", errors="ignore")


def test_driver_build_docs_cover_wdk_test_mode_install_verify_uninstall() -> None:
    text = _read("docs/driver_build_and_signing.md")
    for token in (
        "Windows Driver Kit",
        "check_wdk_environment.ps1",
        "build_driver_package.ps1",
        "bcdedit /set testsigning on",
        "install_test_driver.ps1",
        "verify_driver_format.ps1",
        "uninstall_test_driver.ps1",
        "production driver signing",
        "meeting mode does not require",
    ):
        assert token in text


def test_protocol_v2_docs_define_pcm16_stereo_48k_boundary() -> None:
    text = _read("docs/virtual_audio_protocol_v2.md")
    for token in (
        "48000Hz",
        "PCM16",
        "2ch",
        "interleaved stereo",
        '"protocol_version": 2',
        "AudioFrame",
        "16000Hz mono float32",
    ):
        assert token in text


def test_readme_links_driver_build_and_protocol_docs() -> None:
    text = _read("README.md")
    assert "docs/driver_build_and_signing.md" in text
    assert "docs/virtual_audio_protocol_v2.md" in text
    assert "48000Hz / PCM16 / 2ch" in text
