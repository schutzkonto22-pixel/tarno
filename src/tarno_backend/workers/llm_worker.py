"""Background thread for LLM provider calls."""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal, Slot

from tarno_backend.ai.conversation import ConversationManager
from tarno_backend.ai.provider import LLMProvider, LLMResponse
from tarno_backend.ai.response_guard import guard_response
from tarno_backend.ai.tool_registry import ToolRegistry

log = logging.getLogger(__name__)


class LLMWorker(QObject):
    """Runs LLM inference on a dedicated thread."""

    thinking_started = Signal()
    response_ready = Signal(str)
    tool_executed = Signal(str, str)  # tool_name, result
    error_occurred = Signal(str)
    finished = Signal()

    def __init__(
        self,
        provider: LLMProvider,
        conversation: ConversationManager,
        tools: ToolRegistry,
    ) -> None:
        super().__init__()
        self._provider = provider
        self._conversation = conversation
        self._tools = tools

    def set_provider(self, provider: LLMProvider) -> None:
        self._provider = provider

    @Slot(str)
    def process(self, user_text: str) -> None:
        self.thinking_started.emit()
        self._conversation.add_user_message(user_text)

        tool_schemas = self._tools.get_tool_schemas() if self._provider.supports_tools else None

        try:
            response = self._provider.send(
                messages=self._conversation.get_messages(),
                system=self._conversation.system_prompt,
                tools=tool_schemas,
            )
        except Exception as exc:
            log.exception("LLM Fehler")
            self.error_occurred.emit(f"Kommunikationsfehler: {exc}")
            self.finished.emit()
            return

        spoken = self._process_response(response, user_text)
        if spoken:
            self.response_ready.emit(spoken)
        self.finished.emit()

    def _process_response(self, response: LLMResponse, user_text: str = "") -> str:
        if not response.tool_calls:
            self._conversation.add_assistant_response(response.text)
            return guard_response(response.text, user_text)

        spoken_parts: list[str] = []
        if response.text:
            spoken_parts.append(response.text)

        tc = response.tool_calls[0]
        result = self._tools.execute(tc.name, tc.input)
        self.tool_executed.emit(tc.name, result.message)

        self._conversation.add_assistant_tool_use(response)
        self._conversation.add_tool_result(tc.id, result.message, tool_name=tc.name)

        try:
            tool_schemas = self._tools.get_tool_schemas() if self._provider.supports_tools else None
            followup = self._provider.send_tool_result(
                messages=self._conversation.get_messages(),
                system=self._conversation.system_prompt,
                tools=tool_schemas,
            )
            if followup.text:
                spoken_parts.append(followup.text)
            self._conversation.add_assistant_response(followup.text)
        except Exception:
            log.exception("LLM Followup-Fehler")
            spoken_parts.append(f"Aktion ausgeführt: {result.message}")

        if not result.success:
            log.warning("Tool %s meldet Fehler: %s", tc.name, result.error_code)

        return guard_response(" ".join(spoken_parts), user_text)
