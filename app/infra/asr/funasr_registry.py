from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from pathlib import Path
from threading import Condition
from threading import Lock
from typing import Callable


@dataclass(slots=True)
class FunASRModelHandle:
    kind: str
    key: str
    model: object
    device_effective: str
    model_path: str
    init_mode: str
    init_failure: str = ""
    postprocess: Callable[[str], str] | None = None
    invoke_lock: Lock | None = None

    def runtime_info(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "key": self.key,
            "model_path": self.model_path,
            "device_effective": self.device_effective,
            "model_init_mode": self.init_mode,
            "init_failure": self.init_failure,
            "warm": self.init_mode == "warm",
        }


@dataclass(slots=True)
class _RegistryState:
    kind: str
    key: str
    requested_device: str
    model_path: str
    handle: FunASRModelHandle | None = None
    loading: bool = False
    init_failure: str = ""

    def snapshot(self) -> dict[str, object]:
        if self.handle is not None:
            return self.handle.runtime_info()
        return {
            "kind": self.kind,
            "key": self.key,
            "model_path": self.model_path,
            "device_effective": _normalize_funasr_device(self.requested_device),
            "model_init_mode": "lazy",
            "init_failure": self.init_failure,
            "warm": False,
        }


class FunASRModelRegistry:
    def __init__(self) -> None:
        self._condition = Condition()
        self._states: dict[str, _RegistryState] = {}

    def get_asr(self, *, model_name: str, requested_device: str) -> FunASRModelHandle:
        key = f"asr::{model_name}::{requested_device or 'auto'}"
        return self._get_or_create(
            kind="asr",
            key=key,
            model_path=_resolve_asr_model_path(model_name),
            requested_device=requested_device,
        )

    def get_vad(self, *, requested_device: str) -> FunASRModelHandle:
        key = f"vad::fsmn-vad::{requested_device or 'auto'}"
        return self._get_or_create(
            kind="vad",
            key=key,
            model_path=_resolve_vad_model_path(),
            requested_device=requested_device,
        )

    def snapshot_asr(self, *, model_name: str, requested_device: str) -> dict[str, object]:
        key = f"asr::{model_name}::{requested_device or 'auto'}"
        return self._snapshot(
            kind="asr",
            key=key,
            model_path=_resolve_asr_model_path(model_name),
            requested_device=requested_device,
        )

    def snapshot_vad(self, *, requested_device: str) -> dict[str, object]:
        key = f"vad::fsmn-vad::{requested_device or 'auto'}"
        return self._snapshot(
            kind="vad",
            key=key,
            model_path=_resolve_vad_model_path(),
            requested_device=requested_device,
        )

    def _snapshot(
        self,
        *,
        kind: str,
        key: str,
        model_path: str,
        requested_device: str,
    ) -> dict[str, object]:
        with self._condition:
            state = self._states.get(key)
            if state is None:
                state = _RegistryState(
                    kind=kind,
                    key=key,
                    requested_device=requested_device,
                    model_path=model_path,
                )
                self._states[key] = state
            return state.snapshot()

    def _get_or_create(
        self,
        *,
        kind: str,
        key: str,
        model_path: str,
        requested_device: str,
    ) -> FunASRModelHandle:
        with self._condition:
            state = self._states.get(key)
            if state is None:
                state = _RegistryState(
                    kind=kind,
                    key=key,
                    requested_device=requested_device,
                    model_path=model_path,
                )
                self._states[key] = state
            while state.loading:
                self._condition.wait()
            if state.handle is not None:
                return self._mark_warm(state.handle)
            if state.init_failure:
                raise RuntimeError(state.init_failure)
            state.loading = True

        try:
            handle = self._load_handle(
                kind=kind,
                key=key,
                model_path=model_path,
                requested_device=requested_device,
            )
        except Exception as exc:
            with self._condition:
                state.loading = False
                state.init_failure = str(exc)
                self._condition.notify_all()
            raise

        with self._condition:
            state.loading = False
            state.init_failure = ""
            state.handle = handle
            self._condition.notify_all()
            return handle

    def _load_handle(
        self,
        *,
        kind: str,
        key: str,
        model_path: str,
        requested_device: str,
    ) -> FunASRModelHandle:
        device_effective = _normalize_funasr_device(requested_device)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from funasr import AutoModel  # type: ignore

            postprocess = None
            if kind == "asr":
                try:
                    from funasr.utils.postprocess_utils import rich_transcription_postprocess  # type: ignore
                except Exception:
                    rich_transcription_postprocess = None
                postprocess = rich_transcription_postprocess
            model = AutoModel(
                model=model_path,
                device=device_effective,
                disable_update=True,
            )
        return FunASRModelHandle(
            kind=kind,
            key=key,
            model=model,
            device_effective=device_effective,
            model_path=model_path,
            init_mode="lazy",
            init_failure="",
            postprocess=postprocess,
            invoke_lock=Lock(),
        )

    @staticmethod
    def _mark_warm(handle: FunASRModelHandle) -> FunASRModelHandle:
        if handle.init_mode != "warm":
            handle.init_mode = "warm"
        return handle


def _normalize_funasr_device(device: str) -> str:
    normalized = str(device or "cpu").strip().lower()
    if normalized.startswith("cuda"):
        try:
            import torch  # type: ignore

            if bool(torch.cuda.is_available()):
                return "cuda:0"
        except Exception:
            pass
    return "cpu"


def _resolve_asr_model_path(model_name: str) -> str:
    candidate = Path.home() / ".cache" / "modelscope" / "hub" / "models" / "iic" / "SenseVoiceSmall"
    if candidate.exists():
        return str(candidate)
    text = str(model_name or "").strip()
    if text:
        return text
    return "iic/SenseVoiceSmall"


def _resolve_vad_model_path() -> str:
    candidate = Path.home() / ".cache" / "modelscope" / "hub" / "models" / "iic" / "speech_fsmn_vad_zh-cn-16k-common-pytorch"
    if candidate.exists():
        return str(candidate)
    return "fsmn-vad"


_REGISTRY = FunASRModelRegistry()


def get_funasr_registry() -> FunASRModelRegistry:
    return _REGISTRY


__all__ = ["FunASRModelHandle", "FunASRModelRegistry", "get_funasr_registry"]
