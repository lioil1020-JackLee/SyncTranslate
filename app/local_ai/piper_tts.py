from __future__ import annotations

import io
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.local_ai.runtime_paths import resolve_runtime_path


@dataclass(slots=True)
class PiperTtsEngine:
    executable_path: str = ".\\tools\\piper\\piper.exe"
    model_path: str = ".\\models\\tts\\zh-TW-medium.onnx"
    config_path: str = ".\\models\\tts\\zh-TW-medium.onnx.json"
    speaker_id: int = 0
    length_scale: float = 1.0
    noise_scale: float = 0.667
    noise_w: float = 0.8
    sample_rate: int = 22050

    def synthesize(self, text: str) -> np.ndarray:
        if not text.strip():
            return np.zeros((0, 1), dtype=np.float32)
        exe = resolve_runtime_path(self.executable_path)
        model = resolve_runtime_path(self.model_path)
        conf = resolve_runtime_path(self.config_path)

        if not exe.exists() or not model.exists():
            # Keep app usable even before local toolchain is installed.
            return self._fallback_tone(text=text, sample_rate=self.sample_rate)

        with tempfile.NamedTemporaryFile(prefix="synctranslate_tts_", suffix=".wav", delete=False) as out_file:
            output_path = Path(out_file.name)
        try:
            cmd = [
                str(exe),
                "--model",
                str(model),
                "--output_file",
                str(output_path),
                "--speaker",
                str(max(0, int(self.speaker_id))),
                "--length_scale",
                str(max(0.1, float(self.length_scale))),
                "--noise_scale",
                str(max(0.0, float(self.noise_scale))),
                "--noise_w",
                str(max(0.0, float(self.noise_w))),
            ]
            if conf.exists():
                cmd.extend(["--config", str(conf)])
            completed = subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if completed.returncode != 0:
                stderr = completed.stderr.decode("utf-8", errors="replace").strip()
                raise ValueError(f"piper synthesis failed: {stderr or completed.returncode}")
            return _load_wav(output_path.read_bytes())
        finally:
            try:
                output_path.unlink(missing_ok=True)
            except Exception:
                pass

    @staticmethod
    def _fallback_tone(*, text: str, sample_rate: int) -> np.ndarray:
        duration = min(1.8, 0.25 + len(text) * 0.018)
        count = int(sample_rate * duration)
        t = np.linspace(0.0, duration, count, endpoint=False)
        tone = 0.14 * np.sin(2 * np.pi * 330.0 * t)
        return tone.reshape(-1, 1).astype(np.float32)


def _load_wav(wav_bytes: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    if width != 2:
        raise ValueError(f"unsupported sample width: {width}")
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels)[:, :1]
    else:
        samples = samples.reshape(-1, 1)
    return samples
