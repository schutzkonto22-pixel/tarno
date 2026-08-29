"""Dashboard page — BuildMC AI Desktop clone for TARNO."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from tarno_backend.gui.theme import COLORS


class _StatCard(QFrame):
    """BuildMC-style stat card with label and value."""

    def __init__(self, icon: str, label: str, value: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("card")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(4)

        header = QHBoxLayout()
        header.setSpacing(8)
        icon_label = QLabel(icon)
        icon_label.setStyleSheet(f"background: transparent; font-size: 16px; color: {COLORS['brand_400']};")
        header.addWidget(icon_label)
        label_label = QLabel(label)
        label_label.setStyleSheet(f"background: transparent; color: {COLORS['text_secondary']}; font-size: 12px;")
        header.addWidget(label_label)
        header.addStretch()
        layout.addLayout(header)

        value_label = QLabel(value)
        value_label.setObjectName("cardValue")
        value_label.setStyleSheet("background: transparent;")
        layout.addWidget(value_label)


class _ActionCard(QFrame):
    """BuildMC-style action card with icon, title and description."""

    clicked = Signal()

    def __init__(self, icon: str, title: str, description: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(4)

        icon_label = QLabel(icon)
        icon_label.setStyleSheet(f"background: transparent; font-size: 24px; color: {COLORS['brand_400']};")
        layout.addWidget(icon_label)

        title_label = QLabel(title)
        title_label.setObjectName("cardTitle")
        title_label.setStyleSheet("background: transparent;")
        layout.addWidget(title_label)

        desc = QLabel(description)
        desc.setObjectName("cardSubtitle")
        desc.setStyleSheet("background: transparent;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class DashboardPage(QWidget):
    """TARNO dashboard styled after BuildMC AI."""

    chat_requested = Signal()
    voice_requested = Signal()
    plugins_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Page header
        header = QFrame()
        header.setObjectName("pageHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 24, 24, 16)

        title_block = QVBoxLayout()
        title_block.setSpacing(2)
        title = QLabel("Dashboard")
        title.setObjectName("pageTitle")
        title.setStyleSheet("background: transparent;")
        title_block.addWidget(title)
        subtitle = QLabel("Übersicht über TARNO-Status, System und Aktivitäten.")
        subtitle.setObjectName("pageSubtitle")
        subtitle.setStyleSheet("background: transparent;")
        title_block.addWidget(subtitle)
        header_layout.addLayout(title_block)
        header_layout.addStretch()

        status_layout = QHBoxLayout()
        status_layout.setSpacing(8)
        self._status_dot = QLabel("●")
        self._status_dot.setStyleSheet(f"color: {COLORS['success']}; background: transparent; font-size: 10px;")
        status_layout.addWidget(self._status_dot)
        self._status_text = QLabel("Online")
        self._status_text.setStyleSheet("background: transparent;")
        status_layout.addWidget(self._status_text)
        header_layout.addLayout(status_layout)
        layout.addWidget(header)

        # Scrollable content
        scroll = QScrollArea()
        scroll.setObjectName("chatScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(24, 0, 24, 24)
        content_layout.setSpacing(16)

        # Stat cards
        stats_grid = QGridLayout()
        stats_grid.setSpacing(16)
        self._stat_cards = {
            "provider": _StatCard("⚡", "Provider", "mistral"),
            "tools": _StatCard("🛠", "Tools", "11"),
            "status": _StatCard("🎯", "Status", "Bereit"),
            "memory": _StatCard("🧠", "Memory", "0"),
        }
        for i, card in enumerate(self._stat_cards.values()):
            stats_grid.addWidget(card, i // 4, i % 4)
        content_layout.addLayout(stats_grid)

        # Actions card
        actions_card = QFrame()
        actions_card.setObjectName("card")
        actions_layout = QVBoxLayout(actions_card)
        actions_layout.setContentsMargins(16, 16, 16, 16)
        actions_title = QLabel("Schnellaktionen")
        actions_title.setObjectName("cardTitle")
        actions_title.setStyleSheet("background: transparent;")
        actions_layout.addWidget(actions_title)

        actions_grid = QGridLayout()
        actions_grid.setSpacing(16)
        actions = [
            ("💬", "Chat starten", "Text-Konversation mit TARNO beginnen."),
            ("🎙", "Voice starten", "Sprachsteuerung aktivieren."),
            ("🔌", "Plugins", "Tools und Erweiterungen verwalten."),
        ]
        action_cards = [_ActionCard(icon, title, desc) for icon, title, desc in actions]
        for i, card in enumerate(action_cards):
            actions_grid.addWidget(card, i // 2, i % 2)
        actions_layout.addLayout(actions_grid)

        action_cards[0].clicked.connect(self.chat_requested)
        action_cards[1].clicked.connect(self.voice_requested)
        action_cards[2].clicked.connect(self.plugins_requested)
        content_layout.addWidget(actions_card)

        # System + Logs row
        bottom_grid = QGridLayout()
        bottom_grid.setSpacing(16)

        system_card = QFrame()
        system_card.setObjectName("card")
        system_layout = QVBoxLayout(system_card)
        system_layout.setContentsMargins(16, 16, 16, 16)
        system_title = QLabel("System")
        system_title.setObjectName("cardTitle")
        system_title.setStyleSheet("background: transparent;")
        system_layout.addWidget(system_title)

        system_info = QVBoxLayout()
        system_info.setSpacing(4)
        self._sys_value_labels: dict[str, QLabel] = {}
        for key, label in [("cpu", "CPU"), ("ram", "RAM"), ("gpu", "GPU")]:
            row = QHBoxLayout()
            row.setSpacing(8)
            key_label = QLabel(label)
            key_label.setStyleSheet(f"background: transparent; color: {COLORS['text_secondary']};")
            row.addWidget(key_label)
            row.addStretch()
            v = QLabel("Wird ermittelt...")
            v.setStyleSheet("background: transparent;")
            row.addWidget(v)
            self._sys_value_labels[key] = v
            system_info.addLayout(row)
        system_layout.addLayout(system_info)
        system_layout.addStretch()
        bottom_grid.addWidget(system_card, 0, 0)

        logs_card = QFrame()
        logs_card.setObjectName("card")
        logs_layout = QVBoxLayout(logs_card)
        logs_layout.setContentsMargins(16, 16, 16, 16)
        logs_title = QLabel("Recent Logs")
        logs_title.setObjectName("cardTitle")
        logs_title.setStyleSheet("background: transparent;")
        logs_layout.addWidget(logs_title)
        self._logs = QLabel("No logs yet")
        self._logs.setStyleSheet(
            f"background: {COLORS['dark_800']}; color: {COLORS['text_secondary']}; "
            "padding: 12px; border-radius: 8px; font-family: 'JetBrains Mono', monospace; font-size: 11px;"
        )
        self._logs.setWordWrap(True)
        self._logs.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._logs.setMinimumHeight(160)
        logs_layout.addWidget(self._logs)
        bottom_grid.addWidget(logs_card, 0, 1)
        content_layout.addLayout(bottom_grid)

        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

    def set_provider(self, name: str) -> None:
        self._stat_cards["provider"].findChildren(QLabel)[-1].setText(name)

    def set_status(self, state: str) -> None:
        self._stat_cards["status"].findChildren(QLabel)[-1].setText(state)
        is_ready = state in ("Bereit", "Online")
        color = COLORS["success"] if is_ready else COLORS["warning"]
        self._status_dot.setStyleSheet(f"color: {color}; background: transparent; font-size: 10px;")
        self._status_text.setText("Online" if is_ready else "Busy")

    def set_cpu_ram(self, cpu: str, ram: str) -> None:
        self._sys_value_labels["cpu"].setText(cpu)
        self._sys_value_labels["ram"].setText(ram)

    def set_gpu(self, gpu: str) -> None:
        self._sys_value_labels["gpu"].setText(gpu)

    def add_log(self, line: str) -> None:
        current = self._logs.text()
        if current == "No logs yet":
            current = ""
        lines = (current + "\n" + line).strip().split("\n")
        if len(lines) > 20:
            lines = lines[-20:]
        self._logs.setText("\n".join(lines))
