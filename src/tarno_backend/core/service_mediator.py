"""Service mediator — bridges the GUI to all backend workers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot

from tarno_backend.ai.conversation import ConversationManager
from tarno_backend.ai.factory import create_provider_with_fallback as _create_provider
from tarno_backend.ai.tool_registry import ToolDefinition, ToolRegistry
from tarno_backend.browser.web_control import web_search
from tarno_backend.desktop.app_control import open_application
from tarno_backend.desktop.file_manager import (
    copy_file, create_directory, delete_file, list_directory, move_file, open_file, read_file, search_files, write_file,
)
from tarno_backend.core.command_tool import CommandTool
from tarno_backend.desktop.screenshot import take_screenshot
from tarno_backend.desktop.system_info import get_system_info
from tarno_backend.extensions.coordinator import ExtensionCoordinator
from tarno_backend.telemetry.logging import new_correlation_id, set_correlation_id
from tarno_backend.memory.store import MemoryStore
from tarno_backend.voice.audio_manager import AudioManager
from tarno_backend.voice.echo_protection import EchoProtection
from tarno_backend.voice.synthesizer import SpeechSynthesizer
from tarno_backend.workers.llm_worker import LLMWorker
from tarno_backend.workers.tts_worker import TTSWorker
from tarno_backend.workers.voice_worker import VoiceWorker

if TYPE_CHECKING:
    from tarno_backend.core.config import TarnoConfig

log = logging.getLogger(__name__)


def register_default_tools(registry: ToolRegistry) -> None:
    """Register all built-in desktop tools."""
    tools = [
        ToolDefinition(
            name="open_application",
            description="Öffnet eine Anwendung auf dem Computer (z.B. Notepad, Chrome, Calculator)",
            input_schema={"type": "object", "properties": {"app_name": {"type": "string", "description": "Name der Anwendung"}}, "required": ["app_name"]},
            handler=open_application,
        ),
        ToolDefinition(
            name="open_file",
            description="Öffnet eine Datei mit der Standard-Anwendung",
            input_schema={"type": "object", "properties": {"filepath": {"type": "string", "description": "Vollständiger Pfad zur Datei"}}, "required": ["filepath"]},
            handler=open_file,
        ),
        ToolDefinition(
            name="web_search",
            description="Öffnet den Browser mit einer URL oder führt eine Google-Suche durch",
            input_schema={"type": "object", "properties": {"query": {"type": "string", "description": "URL oder Suchbegriff"}}, "required": ["query"]},
            handler=web_search,
        ),
        ToolDefinition(
            name="get_system_info",
            description="Gibt Systeminformationen zurück (OS, CPU, RAM, Festplatte)",
            input_schema={"type": "object", "properties": {}},
            handler=lambda: get_system_info(),
        ),
        ToolDefinition(
            name="take_screenshot",
            description="Erstellt einen Screenshot des Bildschirms",
            input_schema={"type": "object", "properties": {}},
            handler=lambda: take_screenshot(),
        ),
        ToolDefinition(
            name="search_files",
            description="Sucht Dateien in einem Verzeichnis nach Muster (z.B. '*.pdf', '*.docx')",
            input_schema={"type": "object", "properties": {"directory": {"type": "string", "description": "Verzeichnis zum Suchen"}, "pattern": {"type": "string", "description": "Suchmuster (z.B. '*.pdf')"}}, "required": ["directory", "pattern"]},
            handler=search_files,
        ),
        ToolDefinition(
            name="read_file",
            description="Liest den Inhalt einer Textdatei",
            input_schema={"type": "object", "properties": {"filepath": {"type": "string", "description": "Pfad zur Datei"}}, "required": ["filepath"]},
            handler=read_file,
        ),
        ToolDefinition(
            name="write_file",
            description="Schreibt einen Text in eine Datei (nur unter dem Benutzer-Home-Verzeichnis).",
            input_schema={"type": "object", "properties": {"filepath": {"type": "string", "description": "Zielpfad zur Datei, z.B. ~/Desktop/anleitung.txt"}, "content": {"type": "string", "description": "Inhalt der Datei"}}, "required": ["filepath", "content"]},
            handler=write_file,
        ),
        ToolDefinition(
            name="list_directory",
            description="Listet Dateien und Verzeichnisse in einem Verzeichnis auf (nur unter dem Benutzer-Home-Verzeichnis).",
            input_schema={"type": "object", "properties": {"path": {"type": "string", "description": "Pfad zum Verzeichnis, z.B. ~/Desktop"}}, "required": ["path"]},
            handler=list_directory,
        ),
        ToolDefinition(
            name="create_directory",
            description="Erstellt ein neues Verzeichnis (inkl. Unterverzeichnisse)",
            input_schema={"type": "object", "properties": {"path": {"type": "string", "description": "Pfad des zu erstellenden Verzeichnisses"}}, "required": ["path"]},
            handler=create_directory,
        ),
        ToolDefinition(
            name="copy_file",
            description="Kopiert eine Datei an einen neuen Ort",
            input_schema={"type": "object", "properties": {"source": {"type": "string", "description": "Quellpfad"}, "destination": {"type": "string", "description": "Zielpfad"}}, "required": ["source", "destination"]},
            handler=copy_file,
        ),
        ToolDefinition(
            name="move_file",
            description="Verschiebt eine Datei an einen neuen Ort",
            input_schema={"type": "object", "properties": {"source": {"type": "string", "description": "Quellpfad"}, "destination": {"type": "string", "description": "Zielpfad"}}, "required": ["source", "destination"]},
            handler=move_file,
        ),
        ToolDefinition(
            name="delete_file",
            description="Löscht eine Datei (Vorsicht — unwiderruflich)",
            input_schema={"type": "object", "properties": {"filepath": {"type": "string", "description": "Pfad der zu löschenden Datei"}}, "required": ["filepath"]},
            handler=delete_file,
        ),
    ]
    for t in tools:
        registry.register(t)
    log.info("%d Tools registriert", len(tools))


class ServiceMediator(QObject):
    """Central coordinator between GUI and backend workers."""

    message_received = Signal(str, str)   # role, text
    status_changed = Signal(str)
    mic_state_changed = Signal(bool)
    error_occurred = Signal(str)
    task_approval_requested = Signal(str, str)  # task_id, description
    task_approved = Signal(str)
    task_rejected = Signal(str)

    def __init__(self, config: "TarnoConfig") -> None:
        super().__init__()
        set_correlation_id(new_correlation_id())
        self._config = config
        self._pending_voice_turn = False

        # Core components
        self._provider = _create_provider(config)
        memory_store: MemoryStore | None = None
        if config.memory.enabled:
            memory_path = Path(config.memory.storage_dir).expanduser() / "memory.db"
            memory_store = MemoryStore(memory_path)
        self._conversation = ConversationManager(
            config.llm.max_history,
            memory=config.memory,
            memory_store=memory_store,
            tts_config=config.audio.tts_prompt,
        )
        self._tools = ToolRegistry()
        self._synthesizer = SpeechSynthesizer(config.audio)
        self._audio_manager = AudioManager(config.audio_manager)
        self._echo_protection = EchoProtection()
        register_default_tools(self._tools)

        command_tool = CommandTool(config)
        self._tools.register(ToolDefinition(
            name="execute_command",
            description=(
                "Führt einen validierten Shell-Befehl aus. Nicht für direkte "
                "Nutzung gedacht — das LLM sollte vorher den Intent beschreiben."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "intent": {"type": "string", "description": "Was der Nutzer will, z.B. software_remove"},
                    "shell": {"type": "string", "enum": ["powershell", "cmd", "bash"], "default": "powershell"},
                    "command": {"type": "string", "description": "Der auszuführende Befehl"},
                    "params": {"type": "object"},
                    "explanation": {"type": "string"},
                    "user_query": {"type": "string"},
                },
                "required": ["intent", "command"],
            },
            handler=command_tool,
        ))

        def _request_task_approval(task) -> bool | None:
            self.task_approval_requested.emit(task.id, task.description)
            return None

        self._extensions = ExtensionCoordinator(
            tool_executor=self._tools.execute,
            reminder_callback=lambda msg: self._tts_worker.speak.emit(msg),
            approval_callback=_request_task_approval,
        )
        if config.briefing.enabled:
            self._extensions.set_briefing_times(
                config.briefing.times,
                self._run_briefing,
            )

        # LLM worker thread
        self._llm_thread = QThread()
        self._llm_worker = LLMWorker(self._provider, self._conversation, self._tools)
        self._llm_worker.moveToThread(self._llm_thread)

        self._llm_worker.thinking_started.connect(self._on_thinking)
        self._llm_worker.response_ready.connect(self._on_response)
        self._llm_worker.tool_executed.connect(self._on_tool_executed)
        self._llm_worker.error_occurred.connect(self._on_error)
        self._llm_thread.start()

        # TTS worker thread
        self._tts_thread = QThread()
        self._tts_worker = TTSWorker(
            self._synthesizer,
            on_start=self._on_tts_started,
            on_finish=self._on_tts_finished,
        )
        self._tts_worker.moveToThread(self._tts_thread)

        self._tts_worker.speaking_started.connect(lambda: self.status_changed.emit("Spricht..."))
        self._tts_worker.speaking_finished.connect(self._on_tts_finished)
        self._tts_thread.start()

        # Voice worker thread
        self._voice_thread = QThread()
        self._voice_worker = VoiceWorker(config)
        self._voice_worker.moveToThread(self._voice_thread)

        self._voice_worker.wake_word_detected.connect(self._on_wake_word)
        self._voice_worker.play_confirmation.connect(
            self._tts_worker.play_sound,
            Qt.ConnectionType.BlockingQueuedConnection,
        )
        self._voice_worker.speech_recognized.connect(self._on_speech)
        self._voice_worker.listening_started.connect(lambda: (
            self.mic_state_changed.emit(True),
            self.status_changed.emit("Hört zu..."),
        ))
        self._voice_worker.listening_stopped.connect(lambda: self.mic_state_changed.emit(False))
        self._voice_worker.error_occurred.connect(self._on_error)
        self._voice_thread.start()

        self._extensions.start()
        log.info("ServiceMediator initialisiert (provider=%s)", self._provider.name)

    @property
    def provider_name(self) -> str:
        return self._provider.name

    @Slot(str)
    def send_text(self, text: str) -> None:
        self.message_received.emit("user", text)

        routine_result = self._extensions.try_routine(text)
        if routine_result is not None:
            self._on_response(routine_result.message)
            return

        self._conversation.add_user_message(text)
        self._conversation.inject_relevant_memory(text)
        from PySide6.QtCore import QMetaObject, Qt, Q_ARG
        QMetaObject.invokeMethod(
            self._llm_worker, "process",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(str, text),
        )

    @Slot(bool)
    def toggle_voice(self, active: bool) -> None:
        if active:
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(
                self._voice_worker, "start_listening",
                Qt.ConnectionType.QueuedConnection,
            )
        else:
            self._voice_worker.stop_listening()

    @Slot(str)
    def approve_task(self, task_id: str) -> None:
        task = self._extensions.task_planner.approve(task_id)
        self.task_approved.emit(task_id)
        self.message_received.emit("assistant", f"Aufgabe genehmigt: {task.description}")

    @Slot(str)
    def reject_task(self, task_id: str) -> None:
        task = self._extensions.task_planner.reject(task_id)
        self.task_rejected.emit(task_id)
        self.message_received.emit("assistant", f"Aufgabe abgelehnt: {task.description}")

    def shutdown(self) -> None:
        self._extensions.stop()
        self._voice_worker.stop_listening()

        self._llm_thread.quit()
        self._llm_thread.wait(3000)

        self._tts_thread.quit()
        self._tts_thread.wait(3000)

        self._voice_thread.quit()
        self._voice_thread.wait(3000)

        log.info("ServiceMediator heruntergefahren")

    def _on_thinking(self) -> None:
        self.status_changed.emit("Denkt nach...")

    def _on_response(self, text: str) -> None:
        self.message_received.emit("assistant", text)
        self.status_changed.emit("Bereit")
        from PySide6.QtCore import QMetaObject, Qt, Q_ARG
        QMetaObject.invokeMethod(
            self._tts_worker, "speak",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(str, text),
        )

    def _run_briefing(self) -> None:
        try:
            response = self._provider.send(
                messages=[{"role": "user", "content": self._config.briefing.prompt}],
                system=self._conversation.system_prompt,
                tools=self._tools.get_tool_schemas() if self._provider.supports_tools else None,
            )
            self._on_response(response.text)
        except Exception:
            log.exception("Briefing fehlgeschlagen")

    def _on_tool_executed(self, tool_name: str, result: str) -> None:
        self.status_changed.emit(f"Führe {tool_name} aus...")

    def _on_tts_started(self) -> None:
        self._audio_manager.enter_tts()
        self._echo_protection.on_tts_started()

    def _on_tts_finished(self) -> None:
        self._audio_manager.leave_tts()
        self._echo_protection.on_tts_finished()
        self.status_changed.emit("Bereit")
        if self._pending_voice_turn:
            self._pending_voice_turn = False
            self._voice_worker.signal_turn_ready()

    def _on_wake_word(self) -> None:
        self.status_changed.emit("Hört zu...")

    def _on_speech(self, text: str) -> None:
        self._pending_voice_turn = True
        self.send_text(text)

    def _on_error(self, message: str) -> None:
        self.error_occurred.emit(message)
        self.status_changed.emit(f"Fehler: {message}")
