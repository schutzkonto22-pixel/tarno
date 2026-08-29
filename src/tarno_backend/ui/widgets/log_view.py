"""Log view widget for the control window."""

from __future__ import annotations

from PySide6.QtWidgets import QPlainTextEdit, QSizePolicy, QWidget


class LogView(QPlainTextEdit):
    """Displays a rolling log tail."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(500)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(
            "background-color: #111114; color: #94a3b8; font-family: Consolas, 'Courier New', monospace; font-size: 12px; padding: 8px;"
        )

    def append_log(self, text: str) -> None:
        self.appendPlainText(text.rstrip("\n"))
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())
