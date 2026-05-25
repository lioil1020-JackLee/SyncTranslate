from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DRIVER_ROOT = ROOT / "drivers" / "synctranslate_virtual_audio"
RING_HEADER = DRIVER_ROOT / "overlay" / "EndpointsCommon" / "synctranslate_pcm_ring.h"
RING_SOURCE = DRIVER_ROOT / "overlay" / "EndpointsCommon" / "synctranslate_pcm_ring.cpp"
CONTROL_SOURCE = DRIVER_ROOT / "overlay" / "synctranslate_control.cpp"
FORMAT_DOC = DRIVER_ROOT / "format_contract_v2.md"
OVERLAY_SCRIPT = DRIVER_ROOT / "scripts" / "apply_sysvad_overlay.ps1"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def test_driver_v2_contract_constants_are_pcm16_stereo_48k() -> None:
    text = _read(RING_HEADER)

    assert "#define SYNCTRANSLATE_VIRTUAL_AUDIO_SAMPLE_RATE 48000u" in text
    assert "#define SYNCTRANSLATE_VIRTUAL_AUDIO_CHANNELS 2u" in text
    assert "#define SYNCTRANSLATE_VIRTUAL_AUDIO_BITS_PER_SAMPLE 16u" in text
    assert "#define SYNCTRANSLATE_VIRTUAL_AUDIO_BYTES_PER_SAMPLE 2u" in text
    assert "SYNCTRANSLATE_VIRTUAL_AUDIO_BYTES_PER_FRAME" in text
    assert "SYNCTRANSLATE_PCM_RING_DEFAULT_FRAMES (SYNCTRANSLATE_VIRTUAL_AUDIO_SAMPLE_RATE * 5u)" in text


def test_driver_ring_buffers_pcm16_stereo_frames_not_float32_mono() -> None:
    header = _read(RING_HEADER)
    source = _read(RING_SOURCE)

    assert "SyncTranslatePcmRingWritePcm16Stereo" in header
    assert "SyncTranslatePcmRingReadPcm16Stereo" in header
    assert "SyncTranslatePcmRingWriteFloat32Mono" not in header
    assert "SyncTranslatePcmRingReadFloat32Mono" not in header
    assert "SHORT* samples" in source
    assert "capacityFrames * SYNCTRANSLATE_VIRTUAL_AUDIO_BYTES_PER_FRAME" in source
    assert "sourceIndex = (SIZE_T)frame * SYNCTRANSLATE_VIRTUAL_AUDIO_CHANNELS" in source
    assert "ringIndex = (SIZE_T)g_syncTranslatePcmRing.writePos * SYNCTRANSLATE_VIRTUAL_AUDIO_CHANNELS" in source


def test_write_pcm_ioctl_requires_v2_frame_alignment() -> None:
    text = _read(CONTROL_SOURCE)

    assert "inputBytes % SYNCTRANSLATE_VIRTUAL_AUDIO_BYTES_PER_FRAME" in text
    assert "SyncTranslatePcmRingWritePcm16Stereo((const BYTE*)systemBuffer, inputBytes)" in text
    assert "written * SYNCTRANSLATE_VIRTUAL_AUDIO_BYTES_PER_FRAME" in text
    assert "SyncTranslatePcmRingWriteFloat32Mono" not in text
    assert "inputBytes / sizeof(FLOAT)" not in text


def test_driver_v2_format_doc_records_boundary_and_no_driver_meeting_mode() -> None:
    text = _read(FORMAT_DOC)

    assert "48000 Hz" in text
    assert "PCM16" in text
    assert "2 channels" in text
    assert "interleaved stereo" in text
    assert "meeting mode does not require the driver" in text
    assert "App internal audio remains float32 AudioFrame" in text


def test_sysvad_overlay_script_documents_v2_endpoint_format_macro_patch() -> None:
    text = _read(OVERLAY_SCRIPT)

    assert "Set-SyncTranslateV2FormatContract" in text
    assert "Set-SyncTranslateMicArray48kStereoOnly" in text
    assert "speakerwavtable.h" in text
    assert "micarraywavtable.h" in text
    assert "48000Hz PCM16 2ch" in text
    assert "SPEAKER_HOST_MAX_SAMPLE_RATE                48000" in text
    assert "MICARRAY_PROCESSED_CHANNELS             2" in text
    assert "MICARRAY_32_BITS_PER_SAMPLE_PCM         16" in text
    assert "C_ASSERT(SIZEOF_ARRAY(MicArrayPinDataRangesProcessedStream) == 1)" in text
    assert "KSAUDIO_SPEAKER_STEREO" in text
    assert "PKEY_AudioEngine_DeviceFormat" in text
    assert "fe,ff,02,00,80,bb,00,00,00,ee,02,00,04,00,10,00" in text
    assert 'Add-InfSectionLine -Path $componentInf -Section "SYSVAD.I.WaveMicArray1.AddReg"' in text
    assert "[SYSVAD_SA.NT.Interfaces]" in text
    assert "KSNODETYPE_ANY" in text
    assert "KSNODETYPE_SPEAKER" in text
    assert "KSNODETYPE_MICROPHONE_ARRAY" in text
    assert "Needs=KS.Registration, WDMAUDIO.Registration" in text
    assert "HKR,FX" not in text
    assert "Update-InfSectionText -Path $componentInf -Section \"SYSVAD.I.TopologySpeaker.AddReg\"" not in text
    assert 'Set-InfStringDefinition -Path $extensionInf -Name "ExtendedFriendlyName" -Value "SyncTranslate Virtual Audio Device"' in text
    assert "ComponentizedApoSample.inx" in text
    assert "g_RenderEndpoints" in text
    assert "g_CaptureEndpoints" in text
    assert "Keep the stock SysVAD [SYSVAD_SA.NT.Interfaces] AddInterface section intact" in text


@pytest.mark.integration
@pytest.mark.skip(reason="Requires WDK, signed driver package, installed SyncTranslate virtual devices, and VM audio smoke validation.")
def test_driver_v2_pcm16_stereo_integration_gate() -> None:
    """Marker-only gate for real driver validation outside unit tests."""
