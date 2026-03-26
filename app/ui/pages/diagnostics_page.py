from __future__ import annotations

from typing import Callable

from PySide6.QtWidgets import QGroupBox, QHBoxLayout, QPushButton, QTextEdit, QVBoxLayout, QWidget, QSizePolicy


class DiagnosticsPage(QWidget):
    def __init__(
        self,
        on_health_check: Callable[[], None],
        on_export_diagnostics: Callable[[], None],
        on_save_config: Callable[[], None],
        on_reload_config: Callable[[], None],
    ) -> None:
        super().__init__()
        self._on_health_check = on_health_check
        self._on_export_diagnostics = on_export_diagnostics
        self._on_save_config = on_save_config
        self._on_reload_config = on_reload_config
        self._asr_text = "-"
        self._llm_text = "-"
        self._tts_text = "-"

        self.diagnostics_details = QTextEdit()
        self.diagnostics_details.setReadOnly(True)
        self.diagnostics_details.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.extra_panel: QWidget | None = None

        buttons = QHBoxLayout()
        save_btn = QPushButton("儲存設定")
        reload_btn = QPushButton("重新載入")
        check_btn = QPushButton("系統檢查")
        export_btn = QPushButton("匯出診斷資訊")
        save_btn.clicked.connect(self._on_save_config)
        reload_btn.clicked.connect(self._on_reload_config)
        check_btn.clicked.connect(lambda *_: self._on_health_check())
        export_btn.clicked.connect(self._on_export_diagnostics)
        buttons.addWidget(save_btn)
        buttons.addWidget(reload_btn)
        buttons.addWidget(check_btn)
        buttons.addWidget(export_btn)

        diagnostics_group = QGroupBox("診斷視窗")
        diagnostics_group_layout = QVBoxLayout(diagnostics_group)
        diagnostics_group_layout.setContentsMargins(8, 8, 8, 8)
        diagnostics_group_layout.addWidget(self.diagnostics_details, 1)

        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        content_layout.addWidget(diagnostics_group, 1)

        layout = QVBoxLayout(self)
        layout.addLayout(buttons)
        self.extra_container = QVBoxLayout()
        self.extra_container.setContentsMargins(0, 0, 0, 0)
        self.extra_container.setSpacing(0)
        content_layout.addLayout(self.extra_container, 2)
        layout.addLayout(content_layout, 1)
        self._refresh_diagnostics_text()

    def _refresh_diagnostics_text(self) -> None:
        self.diagnostics_details.setPlainText(
            "\n".join(
                [
                    f"ASR：{self._asr_text}",
                    f"LLM：{self._llm_text}",
                    f"TTS：{self._tts_text}",
                ]
            )
        )

    def set_extra_panel(self, widget: QWidget) -> None:
        if self.extra_panel is widget:
            return
        if self.extra_panel is not None:
            self.extra_container.removeWidget(self.extra_panel)
            self.extra_panel.setParent(None)
        self.extra_panel = widget
        self.extra_container.addWidget(widget, 1)

    def set_asr_details(self, text: str) -> None:
        self._asr_text = text.strip() or "-"
        self._refresh_diagnostics_text()

    def set_llm_details(self, text: str) -> None:
        self._llm_text = text.strip() or "-"
        self._refresh_diagnostics_text()

    def set_tts_details(self, text: str) -> None:
        self._tts_text = text.strip() or "-"
        self._refresh_diagnostics_text()

    def set_health_details(self, text: str) -> None:
        clean = text.strip() or "-"
        self._asr_text = clean
        self._llm_text = clean
        self._tts_text = clean
        self._refresh_diagnostics_text()
