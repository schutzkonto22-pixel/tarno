"""Pages for the TARNO first-start setup wizard."""

from __future__ import annotations

from PySide6.QtCore import QThread, QUrl, Signal, Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QLabel,
    QListWidget,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QWizardPage,
)

from tarno_backend.first_start.config_initializer import ConfigInitializer
from tarno_backend.first_start.hardware_detection import detect_hardware
from tarno_backend.model_manager import DownloadProgress, ModelManager


class WelcomePage(QWizardPage):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setTitle("Willkommen bei TARNO")
        layout = QVBoxLayout(self)
        label = QLabel("Lass uns TARNO für dich einrichten.")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addStretch()
        layout.addWidget(label)
        layout.addStretch()


class HardwarePage(QWizardPage):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setTitle("Hardware-Erkennung")
        self._info = QLabel("Erkenne System...")
        layout = QVBoxLayout(self)
        layout.addWidget(self._info)
        btn = QPushButton("Erneut erkennen")
        btn.clicked.connect(self._detect)
        layout.addWidget(btn)
        layout.addStretch()

    def initializePage(self) -> None:
        self._detect()

    def _detect(self) -> None:
        profile = detect_hardware()
        lines = [
            f"Betriebssystem: {profile.os_name}",
            f"CPU-Kerne: {profile.cpu_count}",
            f"Arbeitsspeicher: {profile.memory_mb} MB",
            f"Audio-Eingänge: {len(profile.audio_input_devices)}",
            f"Audio-Ausgänge: {len(profile.audio_output_devices)}",
        ]
        self._info.setText("\n".join(lines))


class AudioPage(QWizardPage):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setTitle("Audio-Konfiguration")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Whisper-Modell:"))
        self._whisper = QComboBox()
        self._whisper.addItems(["tiny", "base", "small", "medium"])
        self._whisper.setCurrentText("small")
        layout.addWidget(self._whisper)
        layout.addStretch()
        self.registerField("whisperModel", self._whisper, "currentText")


class WakeWordPage(QWizardPage):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setTitle("Wake-Word")
        layout = QVBoxLayout(self)
        self._enabled = QCheckBox("Wake-Word aktiviert")
        self._enabled.setChecked(True)
        layout.addWidget(self._enabled)

        layout.addWidget(QLabel("Backend:"))
        self._backend = QComboBox()
        self._backend.addItems(["openwakeword", "porcupine"])
        self._backend.setCurrentText("openwakeword")
        layout.addWidget(self._backend)

        layout.addWidget(QLabel("Wake-Word-Modell:"))
        self._model = QComboBox()
        self._model.addItems(["hey_jarvis", "hey_mycroft", "alexa", "timer", "weather"])
        self._model.setCurrentText("hey_jarvis")
        layout.addWidget(self._model)

        info = QLabel(
            "openWakeWord ist die empfohlene Standardeinstellung und benötigt keinen API-Key.\n"
            "Picovine benötigt einen Access Key in der Umgebungsvariable PICOVOICE_API_KEY."
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        layout.addStretch()
        self.registerField("wakeWordEnabled", self._enabled)
        self.registerField("wakeWordBackend", self._backend, "currentText")
        self.registerField("wakeWordModel", self._model, "currentText")


class PrivacyPage(QWizardPage):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setTitle("Datenschutz")
        layout = QVBoxLayout(self)
        self._pii = QCheckBox("PII-Redaktion aktivieren")
        self._pii.setChecked(True)
        self._crash = QCheckBox("Crash-Reporting erlauben (opt-in)")
        self._crash.setChecked(False)
        layout.addWidget(self._pii)
        layout.addWidget(self._crash)
        layout.addStretch()
        self.registerField("piiRedaction", self._pii)
        self.registerField("crashReporting", self._crash)


class ProviderPage(QWizardPage):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setTitle("KI-Provider")
        layout = QGridLayout(self)
        layout.addWidget(QLabel("Provider:"), 0, 0)
        self._provider = QComboBox()
        self._provider.addItems(["mistral", "claude", "groq", "gemini", "huggingface", "ollama"])
        layout.addWidget(self._provider, 0, 1)

        info = QLabel(
            "TARNO braucht API-Keys für die gewählten Dienste.\n"
            "Die Keys werden nicht hier eingegeben, sondern als Windows-"
            "Umgebungsvariablen gesetzt.\n"
            "Klicke auf \"Anleitung öffnen\", um zu erfahren, wie das geht."
        )
        info.setWordWrap(True)
        layout.addWidget(info, 1, 0, 1, 2)

        help_btn = QPushButton("Anleitung öffnen")
        help_btn.clicked.connect(self._open_help)
        layout.addWidget(help_btn, 2, 0)

        test_btn = QPushButton("API-Key prüfen")
        test_btn.clicked.connect(self._test_key)
        layout.addWidget(test_btn, 2, 1)

        self._status = QLabel("Noch nicht geprüft")
        layout.addWidget(self._status, 3, 0, 1, 2)
        layout.setRowStretch(4, 1)

        self.registerField("llmProvider", self._provider, "currentText")

    def _open_help(self) -> None:
        # Dev-only doc, not bundled into a PyInstaller build - fine to use
        # the .git-marker-based resolver here since this path is only ever
        # meaningful during a source run anyway.
        from tarno_backend.utils.paths import DOCS_DIR

        doc = DOCS_DIR / "api-keys.md"
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(doc)))

    def _test_key(self) -> None:
        import os

        provider = self.field("llmProvider")
        key_var = {
            "mistral": "MISTRAL_API_KEY",
            "claude": "ANTHROPIC_API_KEY",
            "groq": "GROQ_API_KEY",
            "gemini": "GEMINI_API_KEY",
        }.get(provider)

        if key_var is None:
            self._status.setText(f"✅ Für Provider '{provider}' ist kein API-Key erforderlich")
            return

        if os.environ.get(key_var):
            self._status.setText(f"✅ {key_var} ist gesetzt")
        else:
            self._status.setText(
                f"❌ {key_var} ist nicht gesetzt.\n"
                "Lege die Umgebungsvariable an und starte den Wizard neu."
            )


class ModelDownloadWorker(QThread):
    """Download configured models in a background thread."""

    progress = Signal(DownloadProgress)
    finished = Signal(dict)

    def __init__(self, models: list[dict], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._models = models

    def run(self) -> None:
        manager = ModelManager(models_config=self._models)
        results = manager.download_all(progress_callback=self.progress.emit)
        self.finished.emit(results)


class ModelDownloadPage(QWizardPage):
    def __init__(self, initializer: ConfigInitializer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setTitle("Modelle herunterladen")
        self._initializer = initializer
        self._status = QLabel("Bereite Download vor...")
        self._progress = QProgressBar()
        layout = QVBoxLayout(self)
        layout.addWidget(self._status)
        layout.addWidget(self._progress)
        self._log = QListWidget()
        layout.addWidget(self._log)
        self._worker: ModelDownloadWorker | None = None

    def initializePage(self) -> None:
        self._apply_choices()
        self._download_models()

    def _apply_choices(self) -> None:
        self._initializer.set_provider(
            self.field("llmProvider"),
            "",
        )
        self._initializer.set_privacy(
            self.field("piiRedaction"),
            self.field("crashReporting"),
        )
        self._initializer.set_audio(None, self.field("whisperModel"))
        self._initializer.set_wakeword(
            self.field("wakeWordEnabled"),
            self.field("wakeWordModel"),
            self.field("wakeWordBackend"),
        )
        self._initializer.save()

    def _download_models(self) -> None:
        from tarno_backend.first_start.config_initializer import required_models_for_config

        models = required_models_for_config(self._initializer.config)
        if not models:
            self._status.setText("Alle benötigten Modelle sind bereits enthalten.")
            self._progress.setValue(100)
            return

        self._status.setText("Starte Download...")
        self._worker = ModelDownloadWorker(models, parent=self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_download_finished)
        self._worker.start()

    def _on_download_finished(self, results: dict) -> None:
        failed = [name for name, ok in results.items() if not ok]
        if failed:
            self._status.setText(f"Fehler bei: {', '.join(failed)}")
        else:
            self._status.setText("Alle Modelle bereit")
            self._progress.setValue(100)

    def _on_progress(self, progress: DownloadProgress) -> None:
        self._status.setText(f"Lade {progress.model}...")
        if progress.total:
            self._progress.setValue(int(progress.downloaded / progress.total * 100))
