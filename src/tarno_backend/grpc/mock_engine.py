"""Mock TARNO engine for testing the WinUI frontend without API keys.

Implements the same small surface used by the gRPC bridge so the server
can run even when no LLM provider is configured. The original engine and
GUI are not modified.
"""

from __future__ import annotations

import datetime
import logging
import random
import re
from collections import deque
from typing import Any, Callable

from tarno_backend.ai.tool_registry import ToolDefinition, ToolRegistry
from tarno_backend.browser.web_control import web_search
from tarno_backend.core.config import TarnoConfig
from tarno_backend.core.events import (
    ErrorEvent,
    EventBus,
    ResponseReadyEvent,
    SpeechRecognizedEvent,
)
from tarno_backend.desktop.app_control import open_application
from tarno_backend.desktop.file_manager import (
    copy_file,
    create_directory,
    delete_file,
    move_file,
    open_file,
    read_file,
    search_files,
)
from tarno_backend.desktop.screenshot import take_screenshot
from tarno_backend.desktop.system_info import get_system_info

log = logging.getLogger(__name__)


class MockTarnoEngine:
    """Drop-in replacement for TarnoEngine used by the gRPC server.

    Provides deterministic responses for local commands and a friendly echo
    for everything else. No external API keys are required.
    """

    def __init__(self, config: TarnoConfig) -> None:
        self.config = config
        self.event_bus = EventBus()
        self.tools = ToolRegistry()
        self.conversation_history: deque[dict[str, Any]] = deque(maxlen=config.llm.max_history)
        self.system_prompt = "Du bist TARNO, ein hilfsbereiter AI-Assistent."
        self._register_default_tools()
        log.info("MockTarnoEngine initialisiert")

    def _register_default_tools(self) -> None:
        default_tools = [
            ToolDefinition(
                name="open_application",
                description="Öffnet eine Anwendung auf dem Computer",
                input_schema={"type": "object", "properties": {"app_name": {"type": "string"}}, "required": ["app_name"]},
                handler=open_application,
            ),
            ToolDefinition(
                name="open_file",
                description="Öffnet eine Datei mit der Standard-Anwendung",
                input_schema={"type": "object", "properties": {"filepath": {"type": "string"}}, "required": ["filepath"]},
                handler=open_file,
            ),
            ToolDefinition(
                name="web_search",
                description="Öffnet den Browser mit einer URL oder Suche",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
                handler=web_search,
            ),
            ToolDefinition(
                name="get_system_info",
                description="Gibt Systeminformationen zurück",
                input_schema={"type": "object", "properties": {}},
                handler=lambda: get_system_info(),
            ),
            ToolDefinition(
                name="take_screenshot",
                description="Erstellt einen Screenshot",
                input_schema={"type": "object", "properties": {}},
                handler=lambda: take_screenshot(),
            ),
            ToolDefinition(
                name="search_files",
                description="Sucht Dateien nach Muster",
                input_schema={"type": "object", "properties": {"directory": {"type": "string"}, "pattern": {"type": "string"}}, "required": ["directory", "pattern"]},
                handler=search_files,
            ),
            ToolDefinition(
                name="read_file",
                description="Liest den Inhalt einer Textdatei",
                input_schema={"type": "object", "properties": {"filepath": {"type": "string"}}, "required": ["filepath"]},
                handler=read_file,
            ),
            ToolDefinition(
                name="create_directory",
                description="Erstellt ein Verzeichnis",
                input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                handler=create_directory,
            ),
            ToolDefinition(
                name="copy_file",
                description="Kopiert eine Datei",
                input_schema={"type": "object", "properties": {"source": {"type": "string"}, "destination": {"type": "string"}}, "required": ["source", "destination"]},
                handler=copy_file,
            ),
            ToolDefinition(
                name="move_file",
                description="Verschiebt eine Datei",
                input_schema={"type": "object", "properties": {"source": {"type": "string"}, "destination": {"type": "string"}}, "required": ["source", "destination"]},
                handler=move_file,
            ),
            ToolDefinition(
                name="delete_file",
                description="Löscht eine Datei",
                input_schema={"type": "object", "properties": {"filepath": {"type": "string"}}, "required": ["filepath"]},
                handler=delete_file,
            ),
        ]
        for tool in default_tools:
            self.tools.register(tool)

    def _handle_input(
        self,
        user_text: str,
        image_data_url: str | None = None,
        is_stale: Callable[[], bool] | None = None,
    ) -> None:
        """Simulate processing a user message."""
        text = user_text.lower().strip()
        self.event_bus.publish(SpeechRecognizedEvent(text=user_text))

        try:
            response_text = self._mock_response(text)
        except Exception as exc:
            log.exception("Mock engine error")
            self.event_bus.publish(ErrorEvent(error=exc, module="mock_engine"))
            return

        self.conversation_history.append({"role": "user", "content": user_text})
        self.conversation_history.append({"role": "assistant", "content": response_text})
        self.event_bus.publish(ResponseReadyEvent(text=response_text))

    def _mock_response(self, text: str) -> str:
        """Generate a deterministic local response."""
        if re.search(r"\b(hallo|hi|guten tag|guten morgen)\b", text):
            greetings = [
                "Guten Tag, Sir. Wie kann ich behilflich sein?",
                "Hallo! TARNO steht bereit.",
                "Freut mich, Sie zu sehen, Sir.",
            ]
            return random.choice(greetings)

        if re.search(r"\b(zeit|wie spät|uhrzeit)\b", text):
            now = datetime.datetime.now()
            return f"Die aktuelle Zeit ist {now.strftime('%H:%M Uhr')}."

        if re.search(r"\b(datum|welcher tag|heute)\b", text):
            now = datetime.datetime.now()
            days = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
            return f"Heute ist {days[now.weekday()]}, der {now.strftime('%d. %B %Y')}."

        if re.search(r"\b(wetter|temperatur)\b", text):
            return "In Berlin sind es aktuell etwa 22 Grad Celsius bei leicht bewölktem Himmel."

        if re.search(r"\b(status|system)\b", text):
            return "Alle Systeme funktionieren einwandfrei. TARNO ist online und bereit."

        if re.search(r"\b(hilfe|befehle|was kannst du)\b", text):
            return (
                "Ich kann Ihnen bei Zeit, Datum, Wetter und System-Status helfen. "
                "Sagen Sie einfach 'öffne Notepad' oder 'suche im Web nach...'."
            )

        # Tool detection for common local actions.
        open_app_match = re.search(r"öffne\s+(?:die\s+)?app\s+(\w+)", text)
        if open_app_match:
            app_name = open_app_match.group(1)
            result = self.tools.execute("open_application", {"app_name": app_name})
            return f"Ich öffne {app_name}: {result}"

        web_match = re.search(r"(?:suche|suchen|googeln|öffne)\s+(?:im\s+)?web\s+nach\s+(.+)", text)
        if web_match:
            query = web_match.group(1).strip()
            result = self.tools.execute("web_search", {"query": query})
            return f"Ich suche im Web nach '{query}': {result}"

        if "screenshot" in text:
            result = self.tools.execute("take_screenshot", {})
            return f"{result}"

        if "system" in text:
            result = self.tools.execute("get_system_info", {})
            return f"{result}"

        return (
            f"Verstanden: '{text}'. In der Mock-Umgebung führe ich keine "
            "externe KI-Anfrage aus. Bitte konfigurieren Sie einen echten Provider, "
            "oder fragen Sie nach einer lokalen Aktion wie Zeit, Wetter oder öffne App."
        )
