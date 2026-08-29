"""First-start setup wizard for TARNO."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication, QWizard

from tarno_backend.first_start.config_initializer import ConfigInitializer
from tarno_backend.first_start.pages import (
    AudioPage,
    HardwarePage,
    ModelDownloadPage,
    PrivacyPage,
    ProviderPage,
    WakeWordPage,
    WelcomePage,
)


def _hologram_style() -> str:
    return """
    QWizard {
        background-color: #0a0a0f;
        color: #e0f7ff;
        font-family: 'Segoe UI', sans-serif;
        font-size: 14px;
    }
    QLabel { color: #e0f7ff; }
    QPushButton {
        background-color: #00d4ff;
        color: #0a0a0f;
        border: none;
        padding: 8px 20px;
        border-radius: 4px;
    }
    QLineEdit, QComboBox {
        background-color: #111118;
        color: #e0f7ff;
        border: 1px solid #005f73;
        padding: 6px;
    }
    QProgressBar {
        background-color: #111118;
        border: 1px solid #005f73;
    }
    QProgressBar::chunk { background-color: #00d4ff; }
    """


class FirstStartWizard(QWizard):
    """Branded PySide6 wizard for initial TARNO setup."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("TARNO Erst-Einrichtung")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setStyleSheet(_hologram_style())
        self.setMinimumSize(760, 520)

        initializer = ConfigInitializer()
        self.addPage(WelcomePage())
        self.addPage(HardwarePage())
        self.addPage(AudioPage())
        self.addPage(WakeWordPage())
        self.addPage(PrivacyPage())
        self.addPage(ProviderPage())
        self.addPage(ModelDownloadPage(initializer))


def run_first_start(argv: list[str] | None = None) -> int:
    app = QApplication(argv or sys.argv)
    app.setStyle("Fusion")
    wizard = FirstStartWizard()
    wizard.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(run_first_start())
