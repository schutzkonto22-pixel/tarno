"""Status bar widget for the control window."""

from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QWidget


class StatusBar(QWidget):
    """Displays the current TARNO state."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("statusBar")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 6, 16, 6)
        layout.setSpacing(8)

        self._dot = QLabel("●")
        self._dot.setObjectName("statusDot")
        self._dot.setStyleSheet("color: #2dd4bf;")
        layout.addWidget(self._dot)

        self._label = QLabel("Bereit")
        self._label.setObjectName("statusText")
        layout.addWidget(self._label)

        layout.addStretch()

    def set_state(self, state: str) -> None:
        self._label.setText(state)
        color = self._color_for_state(state)
        self._dot.setStyleSheet(f"color: {color};")

    def _color_for_state(self, state: str) -> str:
        state = state.lower()
        if state in ("bereit", "ready"):
            return "#2dd4bf"
        if state in ("hört zu", "hört zu...", "listening"):
            return "#f59e0b"
        if state in ("denkt nach", "denkt nach...", "thinking"):
            return "#60a5fa"
        if state in ("spricht", "spricht...", "speaking"):
            return "#a78bfa"
        if state in ("stumm", "muted"):
            return "#ef4444"
        if state in ("fehler", "error"):
            return "#f87171"
        return "#94a3b8"
