from __future__ import annotations

from typing import Callable

from app.app_bootstrap import PipelineBundle
from app.schemas import AppConfig


class RuntimeFacade:
    def __init__(self, builder: Callable[[AppConfig, int], PipelineBundle]) -> None:
        self._builder = builder
        self._bundle: PipelineBundle | None = None
        self._dirty = True
        self._revision = 0

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def bundle(self) -> PipelineBundle | None:
        return self._bundle

    def mark_dirty(self) -> None:
        self._dirty = True

    def rebuild(self, config: AppConfig) -> PipelineBundle:
        self._revision += 1
        self._bundle = self._builder(config, self._revision)
        self._dirty = False
        return self._bundle

    def ensure_ready(self, config: AppConfig) -> PipelineBundle:
        if self._bundle is None or self._dirty:
            return self.rebuild(config)
        return self._bundle
