from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QFrame, QPlainTextEdit, QSizePolicy, QVBoxLayout, QWidget


class DiagnosticsPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._asr_text = "-"
        self._llm_text = "-"
        self._tts_text = "-"
        self._asr_runtime_text = ""
        self._llm_runtime_text = ""
        self._tts_runtime_text = ""

        self.diagnostics_details = QPlainTextEdit()
        self.diagnostics_details.setReadOnly(True)
        self.diagnostics_details.setFrameShape(QFrame.Shape.NoFrame)
        self.diagnostics_details.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.diagnostics_details.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.diagnostics_details.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.diagnostics_details.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        metrics = QFontMetrics(self.diagnostics_details.font())
        self.diagnostics_details.setFixedHeight((metrics.lineSpacing() * 3) + 14)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.diagnostics_details, 0)
        self._refresh_diagnostics_text()

    def _refresh_diagnostics_text(self) -> None:
        self.diagnostics_details.setPlainText(
            "\n".join(
                [
                    f"ASR: {self._compose_line(self._asr_text, self._asr_runtime_text)}",
                    f"LLM: {self._compose_line(self._llm_text, self._llm_runtime_text)}",
                    f"TTS: {self._compose_line(self._tts_text, self._tts_runtime_text)}",
                ]
            )
        )

    @staticmethod
    def _compose_line(primary: str, runtime: str) -> str:
        clean_primary = primary.strip() or "-"
        clean_runtime = runtime.strip()
        if not clean_runtime:
            return clean_primary
        if clean_primary == "-":
            return clean_runtime
        return f"{clean_primary} | {clean_runtime}"

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

    def set_asr_runtime_details(self, text: str) -> None:
        self._asr_runtime_text = text.strip()
        self._refresh_diagnostics_text()

    def set_llm_runtime_details(self, text: str) -> None:
        self._llm_runtime_text = text.strip()
        self._refresh_diagnostics_text()

    def set_tts_runtime_details(self, text: str) -> None:
        self._tts_runtime_text = text.strip()
        self._refresh_diagnostics_text()
