"""gRPC server exposing TARNO to the WinUI 3 frontend.

This bridge reuses the existing TarnoEngine and its EventBus without
modifying the original GUI or console entry points. The gRPC server runs
in a separate asyncio process and translates engine events into
ServerMessage streams.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, AsyncIterator

import grpc

from tarno_backend.core.config import TarnoConfig, _USER_CONFIG_NAME
from tarno_backend.core.engine import TarnoEngine
from tarno_backend.core.events import (
    AgentTraceEvent,
    CodingOutputEvent,
    ContextUsageEvent,
    ErrorEvent,
    EventBus,
    ProcessingStageEvent,
    ResponseReadyEvent,
    SpeechRecognizedEvent,
    TtsFinishedEvent,
    TtsStartedEvent,
    VisionAttachmentEvent,
    VoiceErrorEvent,
    WakeWordDetectedEvent,
)
from tarno_backend.ai.model_catalog import PROVIDER_MODEL_CATALOG
from tarno_backend.ai.ollama_client import list_ollama_models
from tarno_backend.ai.pool.edit_apply import apply_merged_instruction
from tarno_backend.ai.pool.exceptions import InvalidPoolConfigError
from tarno_backend.ai.pool.models import PoolAgentSpec, PoolConfig, PoolMessage
from tarno_backend.ai.pool.orchestrator import PoolOrchestrator
from tarno_backend.core.exceptions import PermissionDeniedError
from tarno_backend.desktop.system_info import get_system_info
from tarno_backend.security.content_filter import is_input_safe, is_output_safe, is_prompt_injection_safe
from tarno_backend.grpc import tarno_pb2, tarno_pb2_grpc
from tarno_backend.grpc.tarno_pb2 import VoiceState
from tarno_backend.grpc.mock_engine import MockTarnoEngine

log = logging.getLogger(__name__)


class _ClientConnection:
    """Holds the asyncio queue for one connected gRPC client."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue[tarno_pb2.ServerMessage] = asyncio.Queue()
        self.id = uuid.uuid4().hex[:8]


class TarnoGrpcServicer(tarno_pb2_grpc.TarnoServicer):
    """Implements the Tarno gRPC service.

    Keeps the original PySide6 GUI untouched; the WinUI frontend is just
    another client of the same backend components.
    """

    def __init__(self, bridge: "TarnoGrpcBridge") -> None:
        self.bridge = bridge

    async def Stream(
        self,
        request_iterator: AsyncIterator[tarno_pb2.ClientMessage],
        context: grpc.ServicerContext,
    ) -> AsyncIterator[tarno_pb2.ServerMessage]:
        """Bidirectional streaming: one client per call."""
        connection = self.bridge.add_client()
        log.info("WinUI client connected: %s", connection.id)

        try:
            # Send a welcome message so the UI shows TARNO is online.
            yield tarno_pb2.ServerMessage(
                status=tarno_pb2.StatusUpdate(state="Bereit")
            )

            # Consume client requests in a background task while yielding events.
            request_task = asyncio.create_task(
                self._consume_requests(request_iterator, connection)
            )

            while True:
                message = await connection.queue.get()
                if message is None:
                    break
                yield message

            request_task.cancel()
            try:
                await request_task
            except asyncio.CancelledError:
                pass
        finally:
            self.bridge.remove_client(connection)
            log.info("WinUI client disconnected: %s", connection.id)

    async def _consume_requests(
        self,
        request_iterator: AsyncIterator[tarno_pb2.ClientMessage],
        connection: _ClientConnection,
    ) -> None:
        async for request in request_iterator:
            log.debug("Received client message: %s", request)
            try:
                if request.HasField("chat_input"):
                    self.bridge.handle_chat_input(
                        request.chat_input.text,
                        attachment_data=request.chat_input.attachment_data,
                        attachment_mime_type=request.chat_input.attachment_mime_type,
                        chat_id=request.chat_input.chat_id,
                        workspace=request.chat_input.workspace,
                        chat_mode=request.chat_input.chat_mode,
                    )
                elif request.HasField("voice_command"):
                    self.bridge.set_voice_active(request.voice_command.active)
                elif request.HasField("command"):
                    self.bridge.handle_command_request(request.command)
                elif request.HasField("permission_response"):
                    self.bridge.handle_permission_response(request.permission_response)
                elif request.HasField("cancel_chat"):
                    self.bridge.cancel_current_chat()
                elif request.HasField("set_autonomy_mode"):
                    self.bridge.handle_set_autonomy_mode(request.set_autonomy_mode)
                elif request.HasField("set_vision_enabled"):
                    self.bridge.handle_set_vision_enabled(request.set_vision_enabled)
                elif request.HasField("set_reasoning_mode"):
                    self.bridge.handle_set_reasoning_mode(request.set_reasoning_mode)
                elif request.HasField("set_provider"):
                    self.bridge.handle_set_provider(request.set_provider)
                elif request.HasField("set_high_tier_model"):
                    self.bridge.handle_set_high_tier_model(request.set_high_tier_model)
                elif request.HasField("set_wakeword_model_size"):
                    self.bridge.handle_set_wakeword_model_size(request.set_wakeword_model_size)
                elif request.HasField("set_microphone_device"):
                    self.bridge.handle_set_microphone_device(request.set_microphone_device)
                elif request.HasField("set_speaker_device"):
                    self.bridge.handle_set_speaker_device(request.set_speaker_device)
                elif request.HasField("set_language"):
                    self.bridge.handle_set_language(request.set_language)
                elif request.HasField("coding_task"):
                    self.bridge.handle_coding_task(request.coding_task)
                elif request.HasField("set_active_chat"):
                    self.bridge.handle_set_active_chat(request.set_active_chat)
                elif request.HasField("set_speak_responses"):
                    self.bridge.handle_set_speak_responses(request.set_speak_responses)
                elif request.HasField("set_context_efficiency"):
                    self.bridge.handle_set_context_efficiency(request.set_context_efficiency)
                elif request.HasField("start_pool_task"):
                    self.bridge.handle_start_pool_task(request.start_pool_task)
            except Exception as exc:
                # A single bad/erroring request must not kill this client's
                # entire request loop - without this, every later message
                # from the same WinUI session would silently be ignored
                # until the client reconnects.
                log.exception("Fehler bei der Verarbeitung einer Client-Nachricht")
                self.bridge._broadcast(tarno_pb2.ServerMessage(
                    log=tarno_pb2.LogEntry(
                        level="ERROR",
                        message=f"Anfrage konnte nicht verarbeitet werden: {exc}",
                        module="tarno_backend.grpc.server",
                        timestamp=int(time.time() * 1000),
                    )
                ))

    async def GetSystemInfo(
        self, request: tarno_pb2.Empty, context: grpc.ServicerContext
    ) -> tarno_pb2.SystemInfo:
        return self.bridge.get_system_info()

    async def SetApiKey(
        self, request: tarno_pb2.ApiKeyRequest, context: grpc.ServicerContext
    ) -> tarno_pb2.ApiKeyResponse:
        return self.bridge.set_api_key(request.provider, request.api_key)

    async def GetApiKeyStatus(
        self, request: tarno_pb2.Empty, context: grpc.ServicerContext
    ) -> tarno_pb2.ApiKeyStatusResponse:
        return self.bridge.get_api_key_status()

    async def GetMemorySummary(
        self, request: tarno_pb2.MemoryQuery, context: grpc.ServicerContext
    ) -> tarno_pb2.MemorySummary:
        return self.bridge.get_memory_summary(request.search)

    async def ImportMemory(
        self, request: tarno_pb2.ImportMemoryRequest, context: grpc.ServicerContext
    ) -> tarno_pb2.ImportMemoryResponse:
        return self.bridge.import_memory(request.json_text)

    async def Dictate(
        self, request: tarno_pb2.DictateRequest, context: grpc.ServicerContext
    ) -> tarno_pb2.DictateResponse:
        success, text, error = await self.bridge.dictate_once(request.eagerness)
        return tarno_pb2.DictateResponse(success=success, text=text, error=error)

    async def GetModelCatalog(
        self, request: tarno_pb2.Empty, context: grpc.ServicerContext
    ) -> tarno_pb2.ModelCatalog:
        return self.bridge.get_model_catalog()

    async def GetMicrophoneDevices(
        self, request: tarno_pb2.Empty, context: grpc.ServicerContext
    ) -> tarno_pb2.MicrophoneDeviceList:
        return self.bridge.get_microphone_devices()

    async def GetSpeakerDevices(
        self, request: tarno_pb2.Empty, context: grpc.ServicerContext
    ) -> tarno_pb2.SpeakerDeviceList:
        return self.bridge.get_speaker_devices()


class TarnoGrpcBridge:
    """Owns the engine and forwards events to connected gRPC clients."""

    # How long request_permission_sync waits for a PermissionResponse before
    # denying by default. Class attribute (not a magic number in the method)
    # so tests can shrink it instead of waiting out the real timeout.
    _PERMISSION_TIMEOUT_SECONDS = 120.0

    # Watchdog fuer eine einzelne Chat-Verarbeitung (LLM-Call + Tool-Ausfuehrung
    # + TTS-Wiedergabe) auf dem einzigen Executor-Worker-Thread. Grosszuegig
    # ueber allen bekannten inneren Timeouts (Mistral-HTTP-Timeout 60s inkl.
    # Retries, TTS-get_busy()-Sicherheits-Timeout) - greift NUR, wenn der
    # Worker-Thread tatsaechlich haengt (z.B. ein natives pygame/pyaudio-
    # Deadlock ausserhalb jeder Python-Timeout-Kontrolle). Ohne dieses Netz
    # blockiert ein einziger haengender Turn ALLE folgenden Nachrichten fuer
    # immer, da der ThreadPoolExecutor(max_workers=1) nie einen zweiten
    # Worker bekommt und ein gehaengter Thread sich nicht zurueckholen laesst -
    # genau das vom Nutzer gemeldete "Stuck-Zustand, nur Neustart hilft".
    _TURN_WATCHDOG_TIMEOUT_SECONDS = 600.0

    def __init__(self, engine: TarnoEngine) -> None:
        self.engine = engine
        self.config = engine.config
        self.event_bus: EventBus = engine.event_bus
        self.clients: list[_ClientConnection] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tarno_engine")
        self._coding_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tarno_coding")
        # Agent-Pool: eigener Executor, getrennt von _coding_executor, damit
        # Einzel-Agent-Coding-Tasks und Pool-Tasks sich nie gegenseitig
        # blockieren. Pool-Laeufe selbst bleiben serialisiert (max_workers=1) -
        # die Parallelitaet ZWISCHEN den 2-4 Pool-Agenten kommt aus
        # PoolOrchestrator's eigenem asyncio.gather, nicht aus mehr Slots hier.
        self._pool_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tarno_pool_runner")
        self._is_mock = isinstance(engine, MockTarnoEngine)
        # Phase B (In-Chat-Diktat): serialisiert gleichzeitige Dictate-Aufrufe
        # ueber mehrere Clients - verhindert zwei parallele Mikrofonzugriffe,
        # nicht dass hier praktisch je zwei Clients gleichzeitig diktieren.
        self._dictate_lock = asyncio.Lock()
        self._voice_controller = None
        self._pending_voice_turn = False
        # Which chat the UI currently has open - used to attribute
        # voice-recognized speech (VoiceController.on_speech has no client
        # request to read a chat_id from, unlike a typed ChatInput) to the
        # right chat instead of the hardcoded "default" one. Updated via
        # SetActiveChat whenever the user switches/creates a chat.
        self._active_chat_id: str = "default"
        self._pending_permissions: dict[str, Future] = {}
        self._generation = 0
        self._active_generation: int | None = None
        self._cancelled_generations: set[int] = set()
        self._current_turn_id: str = ""
        self._subscribe()
        self._attach_log_handler()
        self._wire_remote_permissions()
        self._save_audit_baseline()
        self._start_background_loops()

    def _start_background_loops(self) -> None:
        """Starts ExtensionCoordinator (reminders/routines/briefings/tasks,
        Block 3's TarnoScheduler) and ProactiveEngine (Block 3/7's autonomous
        trigger loop, including the vision observer).

        Found via live testing: TarnoEngine.run()/run_text_mode() (voice and
        console launch modes) already call ._extensions.start() and
        ._proactive_engine.start(), but the gRPC/WinUI backend path - the
        ONLY path actually used in practice - drove the engine directly via
        _handle_input() and never called either. Concretely, this meant
        add_reminder-scheduled reminders were computed and stored correctly
        but never actually checked/spoken (ExtensionCoordinator's minute-
        interval scheduler never started), and the Block 7 vision observer's
        camera would open on toggle but never actually get ticked to detect
        motion (ProactiveEngine's background thread never started). Mirrors
        the same class of gap _save_audit_baseline() above already fixed for
        the audit log. No-op for the mock engine (no _extensions/
        _proactive_engine there).
        """
        extensions = getattr(self.engine, "_extensions", None)
        if extensions is not None:
            extensions.start()
        proactive_engine = getattr(self.engine, "_proactive_engine", None)
        if proactive_engine is not None:
            proactive_engine.start()

    def _stop_background_loops(self) -> None:
        extensions = getattr(self.engine, "_extensions", None)
        if extensions is not None:
            extensions.stop()
        proactive_engine = getattr(self.engine, "_proactive_engine", None)
        if proactive_engine is not None:
            proactive_engine.stop()
        vision_observer = getattr(self.engine, "_vision_observer", None)
        if vision_observer is not None:
            vision_observer.close()

    def _save_audit_baseline(self) -> None:
        """Snapshot audit-log hashes so tampering can be detected at shutdown.

        TarnoEngine.run()/run_text_mode() already do this for the voice/console
        launch modes, but the gRPC backend calls neither - it drives the engine
        directly via _handle_input - so without this call here, AuditManager's
        integrity check was silently never armed for the WinUI launch path.
        No-op for the mock engine (no _audit_manager there).
        """
        audit_manager = getattr(self.engine, "_audit_manager", None)
        if audit_manager is not None:
            audit_manager.save_state()

    def shutdown(self) -> None:
        """Stop background loops, verify audit-log integrity, rotate old logs.
        Call on server stop."""
        self._stop_background_loops()
        audit_manager = getattr(self.engine, "_audit_manager", None)
        if audit_manager is None:
            return
        ok, violations = audit_manager.verify_integrity()
        if not ok:
            log.warning("Audit-Integritätsverletzungen: %s", violations)
        audit_manager.rotate_old_logs(retention_days=365)

    def _wire_remote_permissions(self) -> None:
        """Route CommandTool's confirmation dialogs through this bridge.

        The real engine's "execute_command" tool defaults to a Qt-or-console
        confirmation chain. This process is a headless asyncio gRPC server
        with neither a Qt event loop nor an attached console, so that chain
        would hang indefinitely on a MEDIUM/HIGH-risk command. Replace it
        with one that round-trips the confirmation to the connected WinUI
        client instead. No-op for the mock engine (no CommandTool there).
        """
        command_tool = getattr(self.engine, "command_tool", None)
        if command_tool is None or not hasattr(command_tool, "configure_remote_permissions"):
            return
        command_tool.configure_remote_permissions(
            self.request_permission_sync,
            safety_phrase_confirm=lambda: True,  # WinUI dialog already enforced the phrase
        )

    def _attach_log_handler(self) -> None:
        """Stream INFO+ log records to all connected clients as LogEntry."""
        bridge = self

        class _BridgeLogHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                # Skip our own grpc module logs to avoid feedback loops.
                if record.name.startswith("tarno_backend.grpc"):
                    return
                try:
                    msg = record.getMessage()
                    bridge._broadcast(tarno_pb2.ServerMessage(
                        log=tarno_pb2.LogEntry(
                            level=record.levelname,
                            message=msg,
                            module=record.name,
                            timestamp=int(record.created * 1000),
                        )
                    ))
                except Exception:
                    # A logging Handler must never raise (e.g. a closed/stale
                    # event loop from a torn-down bridge) - that would turn an
                    # unrelated log.warning()/log.error() call anywhere in the
                    # app into an uncaught crash. Drop the broadcast silently.
                    pass

        handler = _BridgeLogHandler(level=logging.INFO)
        logging.getLogger().addHandler(handler)

    def _subscribe(self) -> None:
        self.event_bus.subscribe(SpeechRecognizedEvent, self._on_speech)
        self.event_bus.subscribe(ResponseReadyEvent, self._on_response)
        self.event_bus.subscribe(ErrorEvent, self._on_error)
        self.event_bus.subscribe(WakeWordDetectedEvent, self._on_wake_word)
        self.event_bus.subscribe(TtsStartedEvent, self._on_tts_started)
        self.event_bus.subscribe(TtsFinishedEvent, self._on_tts_finished)
        self.event_bus.subscribe(ContextUsageEvent, self._on_context_usage)
        self.event_bus.subscribe(VoiceErrorEvent, self._on_voice_error)
        self.event_bus.subscribe(ProcessingStageEvent, self._on_processing_stage)
        self.event_bus.subscribe(VisionAttachmentEvent, self._on_vision_image)
        self.event_bus.subscribe(AgentTraceEvent, self._on_agent_trace)
        self.event_bus.subscribe(CodingOutputEvent, self._on_coding_output)

    def add_client(self) -> _ClientConnection:
        conn = _ClientConnection()
        self.clients.append(conn)
        return conn

    def remove_client(self, connection: _ClientConnection) -> None:
        if connection in self.clients:
            self.clients.remove(connection)

    def _broadcast(self, message: tarno_pb2.ServerMessage) -> None:
        if self._loop is None:
            return
        for conn in list(self.clients):
            self._loop.call_soon_threadsafe(conn.queue.put_nowait, message)

    def _make_chat_message(
        self,
        role: str,
        text: str,
        chat_id: str = "",
        turn_id: str = "",
    ) -> tarno_pb2.ServerMessage:
        return tarno_pb2.ServerMessage(
            chat_message=tarno_pb2.ChatMessage(
                id=turn_id or uuid.uuid4().hex,
                role=role,
                text=text,
                timestamp=int(time.time() * 1000),
                chat_id=chat_id,
            )
        )

    def _make_status(self, state: str) -> tarno_pb2.ServerMessage:
        return tarno_pb2.ServerMessage(
            status=tarno_pb2.StatusUpdate(state=state)
        )

    def _make_voice_state(self, state: VoiceState) -> tarno_pb2.ServerMessage:
        return tarno_pb2.ServerMessage(
            voice_state=tarno_pb2.VoiceStateUpdate(state=state)
        )

    def _make_thinking(self, active: bool, stage: str = "", reasoning: str = "", stage_key: str = "") -> tarno_pb2.ServerMessage:
        return tarno_pb2.ServerMessage(
            thinking=tarno_pb2.ThinkingUpdate(active=active, stage=stage, reasoning=reasoning, stage_key=stage_key)
        )

    def _make_agent_trace(self, event: AgentTraceEvent) -> tarno_pb2.ServerMessage:
        ts = int(time.time() * 1000)
        agent_trace = tarno_pb2.AgentTrace(
            trace_id=event.trace_id or uuid.uuid4().hex,
            parent_id=event.parent_id,
            timestamp=ts,
            turn_id=event.turn_id or self._current_turn_id,
            chat_id=event.chat_id,
        )
        payload = event.payload
        if event.trace_type == "thought_delta":
            agent_trace.thought_delta.title = payload.get("title", "")
            agent_trace.thought_delta.delta = payload.get("delta", "")
            agent_trace.thought_delta.collapsed = payload.get("collapsed", False)
            agent_trace.thought_delta.finished = payload.get("finished", False)
        elif event.trace_type == "tool_start":
            agent_trace.tool_start.tool = payload.get("tool", "")
            agent_trace.tool_start.target = payload.get("target", "")
            agent_trace.tool_start.description = payload.get("description", "")
        elif event.trace_type == "tool_end":
            agent_trace.tool_end.tool = payload.get("tool", "")
            agent_trace.tool_end.target = payload.get("target", "")
            agent_trace.tool_end.success = payload.get("success", True)
            agent_trace.tool_end.result_summary = payload.get("result_summary", "")
        elif event.trace_type == "file_diff":
            agent_trace.file_diff.path = payload.get("path", "")
            agent_trace.file_diff.is_new = payload.get("is_new", False)
            agent_trace.file_diff.language = payload.get("language", "")
            agent_trace.file_diff.content = payload.get("content", "")
        return tarno_pb2.ServerMessage(agent_trace=agent_trace)

    def _on_agent_trace(self, event: AgentTraceEvent) -> None:
        self._broadcast(self._make_agent_trace(event))

    def _make_coding_output(self, event: CodingOutputEvent) -> tarno_pb2.ServerMessage:
        return tarno_pb2.ServerMessage(
            coding_output=tarno_pb2.CodingOutputMessage(
                kind=event.kind,
                text=event.text,
                timestamp_ms=event.timestamp_ms,
                chat_id=event.chat_id,
            )
        )

    def _on_coding_output(self, event: CodingOutputEvent) -> None:
        self._broadcast(self._make_coding_output(event))

    def _on_processing_stage(self, event: ProcessingStageEvent) -> None:
        """Broadcasts a real, in-progress pipeline stage and a matching
        AgentTrace event for the ThoughtStream UI. The thinking update is
        sent last so that callers looking at the most recent broadcast see
        the current stage and its metadata (reasoning/stage_key) correctly."""
        trace_id = uuid.uuid4().hex
        if event.stage == "reasoning" and event.reasoning:
            self._broadcast(self._make_agent_trace(AgentTraceEvent(
                turn_id=self._current_turn_id,
                trace_id=trace_id,
                trace_type="thought_delta",
                payload={"title": "Thought", "delta": event.reasoning, "finished": True},
            )))
        elif event.stage == "tool_exec":
            self._broadcast(self._make_agent_trace(AgentTraceEvent(
                turn_id=self._current_turn_id,
                trace_id=trace_id,
                trace_type="tool_start",
                payload={"tool": "tool", "target": event.detail, "description": event.detail},
            )))
            self._broadcast(self._make_agent_trace(AgentTraceEvent(
                turn_id=self._current_turn_id,
                trace_id=trace_id,
                trace_type="tool_end",
                payload={"tool": "tool", "target": event.detail, "success": True, "result_summary": "done"},
            )))

        self._broadcast(self._make_thinking(True, stage=event.detail, reasoning=event.reasoning, stage_key=event.stage))

    def _on_speech(self, event: SpeechRecognizedEvent) -> None:
        self._broadcast(self._make_chat_message("user", event.text))
        self._broadcast(self._make_voice_state(VoiceState.VOICE_PROCESSING))

    def _is_active_generation_cancelled(self) -> bool:
        return self._active_generation is not None and self._active_generation in self._cancelled_generations

    def _on_response(self, event: ResponseReadyEvent) -> None:
        # Response text is ready; TTS will start immediately after this event,
        # which triggers _on_tts_started -> VOICE_SPEAKING.
        if self._is_active_generation_cancelled():
            log.debug("Antwort für abgebrochene Generation %s verworfen", self._active_generation)
            return

        # Output-Gate (Verteidigungsschicht 2): verwirft LLM-Antworten mit
        # gefährlichem Code, bevor sie angezeigt/gesprochen werden.
        output_safe, output_reason = is_output_safe(event.text)
        if not output_safe:
            log.warning("LLM-Antwort blockiert (Sicherheitsfilter): %s", output_reason)
            self._broadcast(self._make_chat_message(
                "assistant",
                "Payload blockiert: Generierter Code verstößt gegen die Sicherheitsregeln.",
                chat_id=event.chat_id,
                turn_id=event.turn_id,
            ))
            return

        self._broadcast(self._make_chat_message(
            "assistant",
            event.text,
            chat_id=event.chat_id,
            turn_id=event.turn_id,
        ))

    def _on_context_usage(self, event: ContextUsageEvent) -> None:
        if self._is_active_generation_cancelled():
            return
        self._broadcast(tarno_pb2.ServerMessage(
            context_usage=tarno_pb2.ContextUsage(
                used=event.tokens_used,
                max=event.context_window,
            )
        ))

    def _on_voice_error(self, event: VoiceErrorEvent) -> None:
        # LLM turn came back as a synthetic error placeholder (rate limit,
        # 403, network) rather than a real answer - flash the Voice-Orb to
        # "error". The client-side orb.html itself holds this state visible
        # for a minimum duration before the next state can overwrite it
        # (see orb.html's MIN_ERROR_HOLD_MS) - no server-side delay needed.
        if self._is_active_generation_cancelled():
            return
        self._broadcast(self._make_voice_state(VoiceState.VOICE_ERROR))

    def _on_tts_started(self, event: TtsStartedEvent) -> None:
        # Suppress wake-word listening for as long as TARNO is speaking,
        # regardless of what triggered this TTS (voice OR a typed chat
        # message) - without this, a typed-chat-triggered response left the
        # wake-word scanner listening the whole time, able to hear and
        # self-trigger on TARNO's own voice.
        if self._voice_controller is not None:
            self._voice_controller.on_tts_started()
        # Echte Amplituden-Huellkurve VOR dem Zustandswechsel senden, damit
        # der Client seine Resonanz-Animation bereit hat, sobald der Orb in
        # "speaking" wechselt (kein Nachlaufen/keine leere erste Sekunde).
        log.info("TTS gestartet, envelope=%d levels, duration=%.3fs", len(event.envelope), event.duration_seconds)
        if event.envelope:
            self._broadcast(tarno_pb2.ServerMessage(
                speech_envelope=tarno_pb2.SpeechEnvelopeUpdate(
                    levels=event.envelope,
                    duration_seconds=event.duration_seconds,
                )
            ))
            log.info("SpeechEnvelopeUpdate gesendet: %d levels, %.3fs", len(event.envelope), event.duration_seconds)
        else:
            log.warning("TTS gestartet, aber envelope leer - kein SpeechEnvelopeUpdate gesendet")
        self._broadcast(self._make_voice_state(VoiceState.VOICE_SPEAKING))
        self._broadcast(self._make_status("Spricht..."))

    def _on_vision_image(self, event: VisionAttachmentEvent) -> None:
        """Sendet das tatsaechlich analysierte Kamera-Frame an das UI, damit
        der Nutzer sieht, was das Vision-Modell gesehen hat."""
        if not event.image_bytes:
            return
        log.info("VisionImageAttachment gesendet: %d Bytes (Quelle=%s)", len(event.image_bytes), event.source)
        caption = "Vom Vision-Modell analysiertes Kamerabild" if event.source == "camera" else f"Analysiertes Bild (Quelle: {event.source})"
        self._broadcast(tarno_pb2.ServerMessage(
            vision_image=tarno_pb2.VisionImageAttachment(
                image_bytes=event.image_bytes,
                caption=caption,
                timestamp=int(time.time() * 1000),
            )
        ))

    def _on_tts_finished(self, event: TtsFinishedEvent) -> None:
        if self._voice_controller is not None:
            self._voice_controller.on_tts_finished()
        self._broadcast(self._make_voice_state(VoiceState.VOICE_IDLE))
        self._broadcast(self._make_status("Bereit"))

    def _on_error(self, event: ErrorEvent) -> None:
        self._broadcast(
            tarno_pb2.ServerMessage(
                log=tarno_pb2.LogEntry(
                    level="ERROR",
                    message=str(event.error),
                    module=event.module,
                    timestamp=int(time.time() * 1000),
                )
            )
        )
        self._broadcast(self._make_voice_state(VoiceState.VOICE_ERROR))
        self._broadcast(self._make_status("Fehler"))

    def _on_wake_word(self, event: WakeWordDetectedEvent) -> None:
        self._broadcast(self._make_voice_state(VoiceState.VOICE_LISTENING))
        self._broadcast(self._make_status("Hört zu..."))

    def handle_chat_input(
        self,
        text: str,
        attachment_data: bytes = b"",
        attachment_mime_type: str = "",
        chat_id: str = "",
        workspace=None,
        chat_mode: str = "chat",
    ) -> None:
        """Called from the gRPC thread when the UI sends a chat message."""
        log.debug("handle_chat_input: %r (chat_id=%s)", text, chat_id)
        if not text.strip():
            return

        self._current_turn_id = uuid.uuid4().hex
        chat_id = chat_id or "default"
        # Keep the voice-attribution target (see on_speech/_active_chat_id)
        # in sync with the chat the user is actually typing in too, not just
        # explicit SetActiveChat navigation events.
        self._active_chat_id = chat_id

        workspace_dict = None
        if workspace is not None and workspace.id:
            folders = [
                {"id": f.id, "name": f.name, "path": f.path}
                for f in workspace.folders
            ]
            if not folders and workspace.root_path:
                # Backward compatibility for older clients that only set root_path.
                folders = [{"id": "", "name": "", "path": workspace.root_path}]
            workspace_dict = {
                "id": workspace.id,
                "name": workspace.name,
                "root_path": workspace.root_path,
                "folders": folders,
                "root_paths": [f["path"] for f in folders],
                "include_patterns": list(workspace.include_patterns),
                "exclude_patterns": list(workspace.exclude_patterns),
            }

        image_data_url: str | None = None
        if attachment_data:
            if attachment_mime_type.startswith("image/"):
                try:
                    from tarno_backend.vision.preprocessing import uploaded_bytes_to_data_url
                    image_data_url = uploaded_bytes_to_data_url(bytes(attachment_data))
                except Exception:
                    log.warning("Hochgeladenes Bild konnte nicht verarbeitet werden", exc_info=True)
                    self._broadcast(self._make_chat_message(
                        "assistant",
                        "Das hochgeladene Bild konnte nicht verarbeitet werden.",
                    ))
            else:
                self._broadcast(self._make_chat_message(
                    "assistant",
                    "Nur Bilder werden aktuell unterstützt.",
                ))

        # Input-Gate (Verteidigungsschicht 2): blockiert gefährliche Eingaben,
        # BEVOR eine Verbindung zum LLM aufgebaut wird. Kein LLM-Aufruf bei Block.
        input_safe, input_reason = is_input_safe(text)
        if input_safe:
            input_safe, input_reason = is_prompt_injection_safe(text)
        if not input_safe:
            log.warning("Chat-Eingabe blockiert (Sicherheitsfilter): %s", input_reason)
            self._broadcast(self._make_chat_message(
                "assistant",
                "Zugriff verweigert: Sicherheitsrichtlinie verletzt.",
            ))
            self._broadcast(self._make_status("Bereit"))
            self._finish_voice_turn()
            return

        self._generation += 1
        my_generation = self._generation

        # Ueberlebt selbst einen Watchdog-Timeout: der verwaiste Worker-Thread
        # kann is_stale() noch weit nach dieser Coroutine abfragen (siehe
        # _cleanup_orphaned_generation unten, die den Discard aus dem
        # finally-Block dieser Funktion herausgezogen hat).
        def is_stale() -> bool:
            return my_generation in self._cancelled_generations

        async def _process() -> None:
            log.debug("Starting engine processing for: %r (Generation %d)", text, my_generation)
            discard_now = True
            try:
                async with self._lock:
                    self._active_generation = my_generation
                    # NOT asyncio.wait_for(loop.run_in_executor(...)): once the
                    # worker thread has actually started a blocking call (e.g.
                    # a stuck socket read that ignores its own timeout), that
                    # wrapped future is RUNNING, not merely pending - fut.cancel()
                    # on a running executor future is a no-op (Python cannot
                    # forcibly interrupt a running OS thread), so wait_for's
                    # cancel-then-await-completion machinery ends up silently
                    # waiting for the real call to finish anyway, completely
                    # defeating the timeout (confirmed live: a hung urlopen()
                    # call outlasted this watchdog by 6+ minutes with zero
                    # effect). A raw concurrent.futures.Future.result(timeout=)
                    # does not have this problem - it always raises after the
                    # deadline via a plain condition-variable wait, regardless
                    # of whether the submitted work ever completes. Waiting on
                    # it from a throwaway helper thread (loop.run_in_executor
                    # with the default executor) keeps the event loop itself
                    # unblocked.
                    raw_future = self._executor.submit(
                        self.engine._handle_input,
                        text,
                        image_data_url,
                        chat_id,
                        self._current_turn_id,
                        is_stale,
                        workspace_dict,
                        chat_mode,
                    )
                    await asyncio.get_running_loop().run_in_executor(
                        None, raw_future.result, self._TURN_WATCHDOG_TIMEOUT_SECONDS
                    )
            except TimeoutError:
                log.error(
                    "Watchdog: Verarbeitung (Generation %d) nach %.0fs nicht beendet - "
                    "Worker-Thread gilt als gehaengt, wird durch einen frischen ersetzt",
                    my_generation, self._TURN_WATCHDOG_TIMEOUT_SECONDS,
                )
                if my_generation not in self._cancelled_generations:
                    self._broadcast(self._make_chat_message(
                        "assistant",
                        "Entschuldigung, die Verarbeitung hat zu lange gedauert und wurde abgebrochen.",
                    ))
                    self._broadcast(self._make_status("Fehler"))
                self._replace_wedged_executor()
                # Der urspruengliche Aufruf laeuft im verwaisten alten
                # Executor-Thread weiter (siehe _replace_wedged_executor) -
                # die Generation-Markierung bleibt bestehen, bis er
                # tatsaechlich fertig ist, statt sofort im finally-Block
                # unten freigegeben zu werden. Sonst wuerde is_stale() fuer
                # den verwaisten Thread faelschlich wieder False liefern,
                # sobald DIESE Coroutine hier fertig ist - obwohl der
                # eigentliche Aufruf noch Minuten weiterlaufen kann.
                discard_now = False
                asyncio.create_task(self._cleanup_orphaned_generation(my_generation, raw_future))
            except Exception as exc:
                log.exception("Verarbeitung der Nachricht fehlgeschlagen: %r", text)
                if my_generation not in self._cancelled_generations:
                    self._broadcast(tarno_pb2.ServerMessage(
                        log=tarno_pb2.LogEntry(
                            level="ERROR",
                            message=f"Verarbeitung fehlgeschlagen: {exc}",
                            module="tarno_backend.grpc.server",
                            timestamp=int(time.time() * 1000),
                        )
                    ))
                    self._broadcast(self._make_chat_message(
                        "assistant",
                        f"Entschuldigung, bei der Verarbeitung ist ein Fehler aufgetreten: {exc}",
                    ))
                    self._broadcast(self._make_status("Fehler"))
            finally:
                log.debug("Engine processing finished for: %r", text)
                if discard_now:
                    self._cancelled_generations.discard(my_generation)
                self._active_generation = None
                self._finish_voice_turn()

        asyncio.create_task(_process())

    async def _cleanup_orphaned_generation(self, generation: int, raw_future: Future) -> None:
        """Wartet OHNE eigenes Timeout auf das tatsaechliche Ende eines nach
        einem Watchdog-Timeout verwaisten Worker-Thread-Aufrufs (siehe
        _replace_wedged_executor) und gibt die Generation-Markierung erst
        DANN frei. Bis dahin liefert is_stale() (siehe handle_chat_input)
        fuer diese Generation weiterhin korrekt True, falls der Nutzer
        zwischenzeitlich abgebrochen hatte - TarnoEngine._handle_input
        unterdrueckt dann sein finales speak()/ResponseReadyEvent selbst."""
        try:
            await asyncio.get_running_loop().run_in_executor(None, raw_future.result)
        except Exception:
            pass
        finally:
            self._cancelled_generations.discard(generation)

    def _replace_wedged_executor(self) -> None:
        """Swaps in a fresh single-worker executor after the watchdog fires.
        The old executor's worker thread is NOT forcibly stopped (Python
        cannot kill a running thread) - it's simply abandoned via a
        non-blocking shutdown(wait=False), so future turns get a clean
        worker instead of queueing behind a thread that may never return."""
        old_executor = self._executor
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tarno_engine")
        old_executor.shutdown(wait=False)

    def cancel_current_chat(self) -> None:
        """Called when the user hits "Stop" on an in-flight chat request.

        The blocking HTTP call to the LLM provider already in flight cannot
        actually be aborted (see CancelChat proto comment), but the eventual
        response is discarded instead of surprising the user after they've
        already moved on.
        """
        self._cancelled_generations.add(self._generation)
        log.info("Chat-Anfrage (Generation %d) abgebrochen", self._generation)
        self._broadcast(self._make_status("Bereit"))

    def _finish_voice_turn(self) -> None:
        """After the engine spoke its answer, let the voice pipeline continue
        listening for a follow-up (conversation window)."""
        if self._pending_voice_turn and self._voice_controller is not None:
            self._pending_voice_turn = False
            self._voice_controller.signal_turn_ready()

    def set_voice_active(self, active: bool) -> None:
        if self._is_mock:
            # No real audio pipeline in mock mode — just mirror the UI state.
            state = VoiceState.VOICE_LISTENING if active else VoiceState.VOICE_IDLE
            self._broadcast(self._make_voice_state(state))
            self._broadcast(self._make_status("Hört zu..." if active else "Bereit"))
            return

        if active:
            self._start_voice_controller()
        else:
            self._stop_voice_controller()

    def _start_voice_controller(self) -> None:
        from tarno_backend.core.voice_controller import VoiceController

        if self._voice_controller is not None and self._voice_controller.active:
            return

        loop = self._loop

        def on_wake_word() -> None:
            self._broadcast(self._make_voice_state(VoiceState.VOICE_LISTENING))
            self._broadcast(self._make_status("Hört zu..."))
            # Play the confirmation sound synchronously on the voice thread so the
            # acknowledgment finishes before speech capture starts (no self-listening).
            if self.config.wakeword.confirm_wake_word:
                try:
                    self.engine.synthesizer.play_sound()
                except Exception:
                    log.exception("Wake-Word-Bestätigung fehlgeschlagen")

        def on_speech(text: str) -> None:
            self._pending_voice_turn = True
            if loop is not None:
                # Attribute to whichever chat the UI currently has open
                # (tracked via SetActiveChat) - previously always "default"
                # regardless of the selected chat, since handle_chat_input's
                # chat_id parameter was never passed here at all.
                loop.call_soon_threadsafe(
                    lambda: self.handle_chat_input(text, chat_id=self._active_chat_id)
                )

        def on_state(state: str) -> None:
            mapping = {
                "idle": (VoiceState.VOICE_IDLE, "Bereit"),
                "listening": (VoiceState.VOICE_LISTENING, "Hört zu..."),
                "processing": (VoiceState.VOICE_PROCESSING, "Denkt nach..."),
                "speaking": (VoiceState.VOICE_SPEAKING, "Spricht..."),
                "error": (VoiceState.VOICE_ERROR, "Voice-Fehler"),
            }
            voice_state, status = mapping.get(state, (VoiceState.VOICE_IDLE, "Bereit"))
            self._broadcast(self._make_voice_state(voice_state))
            self._broadcast(self._make_status(status))

        def on_error(message: str) -> None:
            self._broadcast(self._make_voice_state(VoiceState.VOICE_ERROR))
            self._broadcast(self._make_status("Fehler"))
            details = message
            exc_info = sys.exc_info()
            if exc_info[0] is not None:
                details += "\n\n" + "".join(traceback.format_exception(*exc_info))
            self._broadcast(tarno_pb2.ServerMessage(
                log=tarno_pb2.LogEntry(
                    level="ERROR", message=details, module="voice_controller",
                    timestamp=int(time.time() * 1000),
                )
            ))

        self._voice_controller = VoiceController(
            self.config,
            on_wake_word=on_wake_word,
            on_speech=on_speech,
            on_state=on_state,
            on_error=on_error,
        )
        self._voice_controller.start()
        log.info("Voice-Pipeline gestartet (gRPC)")

    def _stop_voice_controller(self) -> None:
        if self._voice_controller is not None:
            self._voice_controller.stop()
            self._voice_controller = None
            log.info("Voice-Pipeline gestoppt (gRPC)")
        self._broadcast(self._make_voice_state(VoiceState.VOICE_IDLE))
        self._broadcast(self._make_status("Bereit"))

    def handle_command_request(self, command: tarno_pb2.CommandRequest) -> None:
        """Execute a local tool command directly (e.g. open_app, web_search)."""
        # Route through the engine's tool registry to reuse existing handlers.
        try:
            result = self.engine.tools.execute(command.name, dict(command.params))
        except Exception as exc:
            log.exception("Tool-Ausführung fehlgeschlagen: %s", command.name)
            self._broadcast(tarno_pb2.ServerMessage(
                log=tarno_pb2.LogEntry(
                    level="ERROR",
                    message=f"Befehl '{command.name}' fehlgeschlagen: {exc}",
                    module="tarno_backend.grpc.server",
                    timestamp=int(time.time() * 1000),
                )
            ))
            self._broadcast(self._make_chat_message(
                "assistant", f"Der Befehl '{command.name}' konnte nicht ausgeführt werden: {exc}"
            ))
            return
        self._broadcast(self._make_chat_message("assistant", f"{result}"))

    def handle_coding_task(self, request: tarno_pb2.CodingTaskRequest) -> None:
        """Start a coding task on the dedicated executor so chat stays responsive."""
        workspace_dict = {
            "id": request.workspace.id,
            "name": request.workspace.name,
            "root_path": request.workspace.root_path,
            "include_patterns": list(request.workspace.include_patterns),
            "exclude_patterns": list(request.workspace.exclude_patterns),
            "folders": [
                {"id": f.id, "name": f.name, "path": f.path}
                for f in request.workspace.folders
            ],
        }
        self._coding_executor.submit(
            self._run_coding_task,
            request.prompt,
            workspace_dict,
            list(request.file_paths),
            request.read_only,
            request.chat_id,
        )

    def _run_coding_task(
        self,
        prompt: str,
        workspace: dict[str, Any],
        file_paths: list[str],
        read_only: bool,
        chat_id: str,
    ) -> None:
        try:
            self.engine._coding_agent.run(
                prompt,
                workspace=workspace,
                fnames=file_paths,
                read_only=read_only,
                chat_id=chat_id,
            )
        except Exception as exc:
            log.exception("Coding-Task fehlgeschlagen")
            self._broadcast(self._make_chat_message(
                "assistant",
                f"Coding-Task fehlgeschlagen: {exc}",
                chat_id=chat_id,
            ))

    def handle_start_pool_task(self, request: tarno_pb2.StartPoolTaskRequest) -> None:
        """Startet einen Agent-Pool-Lauf (2-4 Agenten) auf dem dedizierten
        Pool-Executor - getrennt vom Einzel-Agent-_coding_executor, damit
        sich beide nie blockieren. No-op fuer die Mock-Engine (kein
        _coding_agent dort)."""
        if not hasattr(self.engine, "_coding_agent"):
            log.warning("Pool-Task ignoriert: kein _coding_agent verfügbar (Mock-Engine?)")
            self._broadcast(tarno_pb2.ServerMessage(
                pool_task_response=tarno_pb2.PoolTaskResponse(
                    success=False, error="Agent-Pool im Mock-Modus nicht verfügbar.",
                    chat_id=request.chat_id,
                )
            ))
            return

        try:
            agent_specs = [
                PoolAgentSpec(
                    provider=a.provider, model=a.model,
                    role="lead" if a.is_lead else "worker",
                )
                for a in request.agents
            ]
            pool_config = PoolConfig.from_specs(agent_specs)
        except InvalidPoolConfigError as exc:
            self._broadcast(tarno_pb2.ServerMessage(
                pool_task_response=tarno_pb2.PoolTaskResponse(
                    success=False, error=str(exc), chat_id=request.chat_id,
                )
            ))
            return

        workspace_dict = {
            "id": request.workspace.id,
            "name": request.workspace.name,
            "root_path": request.workspace.root_path,
            "include_patterns": list(request.workspace.include_patterns),
            "exclude_patterns": list(request.workspace.exclude_patterns),
            "folders": [
                {"id": f.id, "name": f.name, "path": f.path}
                for f in request.workspace.folders
            ],
        }
        self._pool_executor.submit(
            self._run_pool_task,
            request.prompt,
            pool_config,
            workspace_dict,
            list(request.file_paths),
            request.read_only,
            request.chat_id,
        )

    def _run_pool_task(
        self,
        prompt: str,
        pool_config: PoolConfig,
        workspace: dict[str, Any],
        file_paths: list[str],
        read_only: bool,
        chat_id: str,
    ) -> None:
        def _on_message(message: PoolMessage) -> None:
            self._broadcast(tarno_pb2.ServerMessage(
                pool_message=tarno_pb2.PoolMessageProto(
                    from_agent=message.from_agent,
                    to_agent=message.to_agent,
                    kind=message.kind,
                    text=message.text,
                    timestamp_ms=message.timestamp_ms,
                    chat_id=chat_id,
                    sub_task_id=message.sub_task_id or "",
                )
            ))

        orchestrator = PoolOrchestrator(self.engine.config, pool_config)
        summary = ""
        success = True
        error = ""
        try:
            final_message = asyncio.run(
                orchestrator.run(prompt, file_paths, on_message=_on_message, workspace=workspace)
            )
            summary = final_message.text

            # Kontext-Effizienz-Plan Phase 5 (Sicherheit): genau EIN
            # Bestaetigungsdialog pro Pool-Lauf, am Merge-Schritt - zeigt die
            # zusammengefuehrte Lead-Anweisung, nicht N Einzelvorschlaege.
            permission_service = getattr(self.engine._coding_agent, "_permission_service", None)
            if not read_only and permission_service is not None:
                permission_service.request_coding_edit(
                    summary, auto_allow=self.engine.config.coding.auto_allow,
                )

            for output in apply_merged_instruction(
                pool_config.lead, self.engine.config, summary, workspace, file_paths, read_only,
            ):
                if output.kind == "error":
                    success = False
                    error = output.text
        except PermissionDeniedError as exc:
            success = False
            error = str(exc)
            log.info("Pool-Task-Anwendung abgelehnt: %s", exc)
        except Exception as exc:
            log.exception("Pool-Task fehlgeschlagen")
            success = False
            error = str(exc)
        finally:
            orchestrator.close()

        self._broadcast(tarno_pb2.ServerMessage(
            pool_task_response=tarno_pb2.PoolTaskResponse(
                success=success, summary=summary, error=error, chat_id=chat_id,
            )
        ))

    def handle_set_autonomy_mode(self, request: tarno_pb2.SetAutonomyMode) -> None:
        """Apply a new autonomy mode to CommandTool's PermissionService and
        broadcast the change to every connected client. No-op for the mock
        engine (no CommandTool/execute_command tool registered there)."""
        from tarno_backend.core.permission_service import AutonomyMode

        command_tool = getattr(self.engine, "command_tool", None)
        if command_tool is None:
            log.warning("Autonomie-Modus-Änderung ignoriert: kein CommandTool verfügbar (Mock-Engine?)")
            return

        try:
            mode = AutonomyMode(request.mode)
        except ValueError:
            log.warning("Unbekannter Autonomie-Modus vom Client: %s", request.mode)
            return

        command_tool.set_autonomy_mode(mode)
        self._broadcast(tarno_pb2.ServerMessage(
            autonomy_mode_update=tarno_pb2.AutonomyModeUpdate(mode=mode.value)
        ))

    def handle_set_vision_enabled(self, request: tarno_pb2.SetVisionEnabled) -> None:
        """Block 7, Phase 69 (Privacy-Layer): live toggle for the vision
        observer's recording opt-out. No-op for the mock engine (no
        set_vision_enabled method there)."""
        if not hasattr(self.engine, "set_vision_enabled"):
            log.warning("Vision-Umschaltung ignoriert: kein Vision-Observer verfügbar (Mock-Engine?)")
            return

        applied = self.engine.set_vision_enabled(request.enabled)
        self._broadcast(tarno_pb2.ServerMessage(
            vision_status_update=tarno_pb2.VisionStatusUpdate(
                camera_active=self.engine.vision_camera_active if applied else False,
                vision_available=applied and self.engine.vision_available,
            )
        ))

    def handle_set_reasoning_mode(self, request: tarno_pb2.SetReasoningMode) -> None:
        """Phase C: toggles the active reasoning_effort. No-op for the mock
        engine (no set_reasoning_mode method there)."""
        if not hasattr(self.engine, "set_reasoning_mode"):
            log.warning("Denk-Modus-Umschaltung ignoriert: kein set_reasoning_mode verfügbar (Mock-Engine?)")
            return

        effort = request.effort or "medium"
        self.engine.set_reasoning_mode(request.enabled, effort)
        self._broadcast(tarno_pb2.ServerMessage(
            reasoning_mode_update=tarno_pb2.ReasoningModeUpdate(
                enabled=request.enabled, effort=effort,
            )
        ))

    def handle_set_high_tier_model(self, request: tarno_pb2.SetHighTierModel) -> None:
        """Manueller "Starkes Modell"-Schalter (ersetzt die vormals
        automatische, fehleranfaellige Klassifizierung). No-op fuer die
        Mock-Engine (kein set_high_tier_model dort)."""
        if not hasattr(self.engine, "set_high_tier_model"):
            log.warning("Starkes-Modell-Umschaltung ignoriert: kein set_high_tier_model verfügbar (Mock-Engine?)")
            return

        self.engine.set_high_tier_model(request.enabled)
        self._broadcast(tarno_pb2.ServerMessage(
            high_tier_model_update=tarno_pb2.HighTierModelUpdate(enabled=request.enabled)
        ))

    def handle_set_speak_responses(self, request: tarno_pb2.SetSpeakResponses) -> None:
        """Chat-Umschalter "Vorlesen". No-op fuer die Mock-Engine (kein
        set_speak_responses dort)."""
        if not hasattr(self.engine, "set_speak_responses"):
            log.warning("Vorlesen-Umschaltung ignoriert: kein set_speak_responses verfügbar (Mock-Engine?)")
            return

        self.engine.set_speak_responses(request.enabled)
        self._broadcast(tarno_pb2.ServerMessage(
            speak_responses_update=tarno_pb2.SpeakResponsesUpdate(enabled=request.enabled)
        ))

    def handle_set_context_efficiency(self, request: tarno_pb2.SetContextEfficiency) -> None:
        """Chat-Umschalter "Kontext-Effizienz" (Beta). No-op fuer die
        Mock-Engine (kein set_context_efficiency dort)."""
        if not hasattr(self.engine, "set_context_efficiency"):
            log.warning("Kontext-Effizienz-Umschaltung ignoriert: kein set_context_efficiency verfügbar (Mock-Engine?)")
            return

        applied = self.engine.set_context_efficiency(request.enabled)
        self._broadcast(tarno_pb2.ServerMessage(
            context_efficiency_update=tarno_pb2.ContextEfficiencyUpdate(enabled=applied)
        ))

    def handle_set_wakeword_model_size(self, request: tarno_pb2.SetWakeWordModelSize) -> None:
        """Settings small/large toggle for the Vosk wake-word model. The
        actual download/load happens on the VoiceController's own background
        thread (see VoiceController.request_vosk_model_size) - this just
        queues the request and broadcasts a "loading" status immediately so
        the UI doesn't look frozen for however long an uncached large-model
        download takes, then broadcasts the final result once it lands."""
        size = request.size
        if self._voice_controller is None:
            log.warning("Wake-Word-Modellwechsel ignoriert: Sprach-Pipeline ist nicht aktiv")
            self._broadcast(tarno_pb2.ServerMessage(
                wakeword_model_size_update=tarno_pb2.WakeWordModelSizeUpdate(
                    active_size=self.config.wakeword.vosk_model_size,
                    loading=False,
                    status="Sprach-Pipeline ist nicht aktiv.",
                )
            ))
            return

        label = "großes" if size == "large" else "kleines"
        self._broadcast(tarno_pb2.ServerMessage(
            wakeword_model_size_update=tarno_pb2.WakeWordModelSizeUpdate(
                active_size=size, loading=True, status=f"Lade {label} Vosk-Modell...",
            )
        ))

        def _on_result(success: bool, applied_size: str) -> None:
            active = applied_size if success else (self._voice_controller.vosk_model_size or applied_size)
            self._broadcast(tarno_pb2.ServerMessage(
                wakeword_model_size_update=tarno_pb2.WakeWordModelSizeUpdate(
                    active_size=active,
                    loading=False,
                    status="Aktiv" if success else "Modellwechsel fehlgeschlagen",
                )
            ))

        self._voice_controller.request_vosk_model_size(size, on_result=_on_result)

    def handle_set_microphone_device(self, request: tarno_pb2.SetMicrophoneDevice) -> None:
        """Settings microphone picker - same hot-swap-and-broadcast shape as
        handle_set_wakeword_model_size (see VoiceController.
        request_microphone_device)."""
        device = request.device
        if self._voice_controller is None:
            log.warning("Mikrofon-Wechsel ignoriert: Sprach-Pipeline ist nicht aktiv")
            self._broadcast(tarno_pb2.ServerMessage(
                microphone_device_update=tarno_pb2.MicrophoneDeviceUpdate(
                    active_device=self.config.audio.microphone_device,
                    loading=False,
                    status="Sprach-Pipeline ist nicht aktiv.",
                )
            ))
            return

        self._broadcast(tarno_pb2.ServerMessage(
            microphone_device_update=tarno_pb2.MicrophoneDeviceUpdate(
                active_device=device, loading=True, status="Wechsle Mikrofon...",
            )
        ))

        def _on_result(success: bool, applied_device: str) -> None:
            active = applied_device if success else self.config.audio.microphone_device
            self._broadcast(tarno_pb2.ServerMessage(
                microphone_device_update=tarno_pb2.MicrophoneDeviceUpdate(
                    active_device=active,
                    loading=False,
                    status="Aktiv" if success else "Mikrofon-Wechsel fehlgeschlagen",
                )
            ))

        self._voice_controller.request_microphone_device(device, on_result=_on_result)

    def handle_set_speaker_device(self, request: tarno_pb2.SetSpeakerDevice) -> None:
        """Settings speaker picker. Unlike the microphone, TTS playback isn't
        a continuously-polled stream, so this applies immediately (guarded by
        SpeechSynthesizer's own lock, see set_output_device's docstring) -
        no VoiceController hot-swap queue needed."""
        device = request.device
        synthesizer = getattr(self.engine, "synthesizer", None)
        if synthesizer is None:
            log.warning("Lautsprecher-Wechsel ignoriert: keine Synthesizer-Instanz (Mock-Engine?)")
            self._broadcast(tarno_pb2.ServerMessage(
                speaker_device_update=tarno_pb2.SpeakerDeviceUpdate(
                    active_device="", loading=False, status="Sprachausgabe ist nicht aktiv.",
                )
            ))
            return
        success = synthesizer.set_output_device(device or None)
        self._broadcast(tarno_pb2.ServerMessage(
            speaker_device_update=tarno_pb2.SpeakerDeviceUpdate(
                active_device=device if success else "",
                loading=False,
                status="Aktiv" if success else "Lautsprecher-Wechsel fehlgeschlagen",
            )
        ))

    def handle_set_language(self, request: tarno_pb2.SetLanguage) -> None:
        """Global language switch (de | en). Updates config, persists to disk
        and broadcasts the result. TTS/STT will pick up the new language for
        the next turn without touching wake-word thresholds or microphone."""
        language = request.language
        log.info("Sprachwechsel angefordert: %s", language)
        try:
            changed = self.config.set_language(language)
            user_path = Path.home() / ".tarno" / "config" / _USER_CONFIG_NAME
            self.config.save(user_path)
            status = "Sprache umgestellt" if changed else "Sprache bereits aktiv"
            log.info("%s auf %s", status, self.config.language)
            self._broadcast(tarno_pb2.ServerMessage(
                language_changed=tarno_pb2.LanguageChanged(
                    language=self.config.language,
                    success=True,
                    message=status,
                )
            ))
        except Exception:
            log.exception("Sprachwechsel fehlgeschlagen")
            self._broadcast(tarno_pb2.ServerMessage(
                language_changed=tarno_pb2.LanguageChanged(
                    language=getattr(self.config, "language", "de"),
                    success=False,
                    message="Sprachwechsel fehlgeschlagen",
                )
            ))

    def handle_set_active_chat(self, request: tarno_pb2.SetActiveChat) -> None:
        """Tracks which chat the UI has open, so on_speech (below) knows
        which chat to attribute voice-recognized text to instead of always
        using the hardcoded "default" chat_id."""
        self._active_chat_id = request.chat_id or "default"

    def get_microphone_devices(self) -> tarno_pb2.MicrophoneDeviceList:
        from tarno_backend.voice.audio_stream import list_input_devices
        return tarno_pb2.MicrophoneDeviceList(devices=[
            tarno_pb2.MicrophoneDeviceInfo(
                name=d["name"], index=d["index"], is_default=d["is_default"],
                host_api=d["host_api"], is_builtin=d["is_builtin"],
            )
            for d in list_input_devices()
        ])

    def get_speaker_devices(self) -> tarno_pb2.SpeakerDeviceList:
        from tarno_backend.voice.synthesizer import list_output_devices
        return tarno_pb2.SpeakerDeviceList(devices=[
            tarno_pb2.SpeakerDeviceInfo(
                name=d["name"], index=d["index"], is_default=d["is_default"],
                is_builtin=d["is_builtin"],
            )
            for d in list_output_devices()
        ])

    def handle_set_provider(self, request: tarno_pb2.SetProvider) -> None:
        """Phase D: applies an explicit provider switch and broadcasts the
        resulting state. On failure, broadcasts the STILL-active (old)
        provider name, never the failed wish-name, so the UI can't end up
        showing a provider that isn't actually running. No-op for the mock
        engine (no set_provider method there)."""
        if not hasattr(self.engine, "set_provider"):
            log.warning("Provider-Wechsel ignoriert: kein set_provider verfügbar (Mock-Engine?)")
            return

        success, message = self.engine.set_provider(request.provider, request.model or None)
        # set_provider only reassigns self.engine.provider/active_model on full
        # success, so these are always the STILL-active values here, regardless
        # of whether this particular request succeeded.
        active_provider = getattr(self.engine, "provider", None)
        provider_name = active_provider.name if active_provider is not None else ""
        self._broadcast(tarno_pb2.ServerMessage(
            provider_update=tarno_pb2.ProviderUpdate(
                provider=provider_name, success=success, message=message,
                model=getattr(self.engine, "active_model", None) or "",
            )
        ))

    def get_model_catalog(self) -> tarno_pb2.ModelCatalog:
        """Phase 1-6: builds the curated per-provider model catalog for the
        WinUI picker. Static providers come from tarno.ai.model_catalog;
        Ollama is queried live since it's whatever the user has pulled
        locally, not a fixed list."""
        providers: dict[str, tarno_pb2.ProviderModelList] = {}
        for provider, models in PROVIDER_MODEL_CATALOG.items():
            providers[provider] = tarno_pb2.ProviderModelList(
                models=[
                    tarno_pb2.ModelInfo(
                        id=m.id,
                        label=m.label,
                        rate_limit_note=m.rate_limit_note,
                        capabilities=list(m.capabilities),
                    )
                    for m in models
                ]
            )

        ollama_base_url = getattr(self.engine.config.llm.ollama, "base_url", None)
        ollama_names = list_ollama_models(ollama_base_url) if ollama_base_url else list_ollama_models()
        providers["ollama"] = tarno_pb2.ProviderModelList(
            models=[
                tarno_pb2.ModelInfo(
                    id=name, label=name, rate_limit_note="lokal, keine Cloud-Limits",
                    capabilities=["lokal"],
                )
                for name in ollama_names
            ]
        )

        return tarno_pb2.ModelCatalog(providers=providers)

    def request_permission_sync(self, permission, target: str, risk_level, duration) -> tuple[bool, bool]:
        """Block the calling thread until the WinUI client answers a PermissionRequest.

        Signature matches PermissionService's ConfirmationDialog callable, so this
        can be passed straight into CommandTool.configure_remote_permissions().
        Always called from the engine's single-worker executor thread (see
        handle_chat_input's run_in_executor), never from the asyncio event loop
        thread - blocking here is safe and does not stall the gRPC server.
        """
        from tarno_backend.core.permission_service import PermissionService

        request_id = uuid.uuid4().hex
        future: Future = Future()
        self._pending_permissions[request_id] = future

        safety_phrase = PermissionService._SAFETY_PHRASE if risk_level.value == "high" else ""
        self._broadcast(tarno_pb2.ServerMessage(
            permission_request=tarno_pb2.PermissionRequest(
                request_id=request_id,
                permission=permission.value,
                target=target,
                risk_level=risk_level.value,
                safety_phrase=safety_phrase,
            )
        ))

        try:
            approved, persistent = future.result(timeout=self._PERMISSION_TIMEOUT_SECONDS)
        except TimeoutError:
            log.warning("Permission-Anfrage %s: keine Antwort vom Frontend (Timeout)", request_id)
            approved, persistent = False, False
        finally:
            self._pending_permissions.pop(request_id, None)
        return approved, persistent

    def handle_permission_response(self, response: tarno_pb2.PermissionResponse) -> None:
        """Resolve the pending future for a PermissionRequest, if still open."""
        future = self._pending_permissions.get(response.request_id)
        if future is None:
            log.warning(
                "Permission-Antwort für unbekannte/abgelaufene Anfrage %s ignoriert",
                response.request_id,
            )
            return
        if not future.done():
            future.set_result((response.approved, response.persistent))

    def set_api_key(self, provider: str, api_key: str) -> tarno_pb2.ApiKeyResponse:
        from tarno_backend.ai.factory import _PROVIDER_SECRET_NAMES
        from tarno_backend.security.secrets import SecretsVault

        secret_name = _PROVIDER_SECRET_NAMES.get(provider)
        if secret_name is None:
            return tarno_pb2.ApiKeyResponse(
                success=False, message=f"Unbekannter Provider: {provider}"
            )
        if not api_key.strip():
            return tarno_pb2.ApiKeyResponse(success=False, message="API-Key ist leer.")
        try:
            vault = SecretsVault(backend=self.config.security.secrets_backend)
            vault.set(secret_name, api_key.strip())
        except Exception as exc:
            log.exception("Konnte API-Key für %s nicht speichern", provider)
            return tarno_pb2.ApiKeyResponse(success=False, message=str(exc))
        return tarno_pb2.ApiKeyResponse(success=True, message="Gespeichert.")

    def get_api_key_status(self) -> tarno_pb2.ApiKeyStatusResponse:
        from tarno_backend.ai.factory import _PROVIDER_SECRET_NAMES
        from tarno_backend.security.secrets import SecretsVault

        vault = SecretsVault(backend=self.config.security.secrets_backend)
        configured = {
            provider: bool(vault.get(secret_name))
            for provider, secret_name in _PROVIDER_SECRET_NAMES.items()
        }
        return tarno_pb2.ApiKeyStatusResponse(configured=configured)

    _MEMORY_SUMMARY_LIMIT = 200

    def get_memory_summary(self, search: str) -> tarno_pb2.MemorySummary:
        """Block 6, Phase 54: feeds the MemoryPage sidebar with real facts/
        episodes from the Block 4 memory layer. Returns memory_enabled=False
        (empty lists) gracefully if memory is disabled or this is the mock
        engine (no memory_store attribute at all)."""
        memory_store = getattr(self.engine, "memory_store", None)
        if memory_store is None:
            return tarno_pb2.MemorySummary(memory_enabled=False)

        search = search.strip()
        try:
            facts = memory_store.search_facts(search, limit=self._MEMORY_SUMMARY_LIMIT)
            episodes = memory_store.get_recent_episodes(limit=self._MEMORY_SUMMARY_LIMIT)
            if search:
                needle = search.lower()
                episodes = [e for e in episodes if needle in e.summary.lower()]
        except Exception:
            log.exception("Memory-Summary konnte nicht geladen werden")
            return tarno_pb2.MemorySummary(memory_enabled=True)

        return tarno_pb2.MemorySummary(
            memory_enabled=True,
            facts=[
                tarno_pb2.MemoryFactEntry(
                    key=f.key, value=f.value, source=f.source,
                    updated_at=int(f.updated_at.timestamp() * 1000),
                )
                for f in facts
            ],
            episodes=[
                tarno_pb2.MemoryEpisodeEntry(
                    summary=e.summary, tags=e.tags,
                    created_at=int(e.created_at.timestamp() * 1000),
                )
                for e in episodes
            ],
        )

    def import_memory(self, json_text: str) -> tarno_pb2.ImportMemoryResponse:
        memory_store = getattr(self.engine, "memory_store", None)
        if memory_store is None:
            return tarno_pb2.ImportMemoryResponse(
                success=False, imported_count=0,
                message="Gedächtnis ist deaktiviert oder nicht verfügbar.",
            )
        try:
            count = memory_store.import_from_json_text(json_text)
            return tarno_pb2.ImportMemoryResponse(
                success=True, imported_count=count,
                message=f"{count} Einträge importiert.",
            )
        except Exception as exc:
            log.exception("Memory-Import fehlgeschlagen")
            return tarno_pb2.ImportMemoryResponse(
                success=False, imported_count=0, message=f"Import fehlgeschlagen: {exc}",
            )

    async def dictate_once(self, eagerness: str) -> tuple[bool, str, str]:
        """Phase B: capture one dictated utterance for the chat text box.

        Blocks the calling gRPC handler coroutine (via run_in_executor) but
        never the asyncio event loop itself. Refuses immediately - without
        ever touching the microphone - if the normal wake-word/voice loop is
        currently active, since both would otherwise fight over the same
        audio device."""
        if self._voice_controller is not None and self._voice_controller.active:
            return False, "", "Mikrofon ist durch den Sprachmodus belegt — bitte zuerst deaktivieren."

        loop = asyncio.get_running_loop()
        async with self._dictate_lock:
            return await loop.run_in_executor(self._executor, self._blocking_dictate, eagerness)

    def _blocking_dictate(self, eagerness: str) -> tuple[bool, str, str]:
        from tarno_backend.voice.audio_stream import AudioStream, AudioStreamSource
        from tarno_backend.voice.faster_whisper_recognizer import FasterWhisperRecognizer

        try:
            recognizer = FasterWhisperRecognizer(self.config.audio, agc_config=self.config.agc)
        except Exception as exc:
            log.exception("Diktat: Spracherkennung konnte nicht geladen werden")
            return False, "", f"Spracherkennung konnte nicht geladen werden: {exc}"

        stream = AudioStream(self.config.audio)
        try:
            stream.start()
        except Exception as exc:
            log.exception("Diktat: Mikrofon konnte nicht geöffnet werden")
            return False, "", f"Mikrofon konnte nicht geöffnet werden: {exc}"

        try:
            source = AudioStreamSource(stream)
            recognizer.calibrate(source)
            text = recognizer.listen_and_recognize(source)
        except Exception as exc:
            log.exception("Diktat fehlgeschlagen")
            return False, "", f"Diktat fehlgeschlagen: {exc}"
        finally:
            stream.stop()

        if not text:
            return False, "", "Keine Sprache erkannt."
        return True, text, ""

    def get_system_info(self) -> tarno_pb2.SystemInfo:
        try:
            import platform
            import psutil

            info = get_system_info()
            return tarno_pb2.SystemInfo(
                os=f"{platform.system()} {platform.release()}",
                cpu=platform.processor(),
                memory=info.split("\n")[2] if "\n" in info else "",
                cpu_percent=psutil.cpu_percent(interval=0.1),
                memory_percent=psutil.virtual_memory().percent,
            )
        except Exception as exc:
            log.warning("System info error: %s", exc)
            return tarno_pb2.SystemInfo(os="unknown", cpu="unknown", memory="")

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop


def _free_port(port: int) -> bool:
    """If something is listening on *port*, kill the process if it is
    one of ours (tarno.exe / python).  This avoids "Address already in
    use" after a previous backend crash or lingering process."""
    try:
        netstat = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True, check=False, timeout=5
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False

    pids: set[int] = set()
    for line in netstat.stdout.splitlines():
        if f":{port}" in line:
            parts = line.strip().split()
            if parts and parts[-1].isdigit():
                pids.add(int(parts[-1]))

    killed = False
    for pid in pids:
        try:
            tasklist = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
                capture_output=True, text=True, check=False, timeout=5,
            )
            name_line = tasklist.stdout.strip().splitlines()[0]
            name = name_line.split(",")[0].strip().strip('"')
        except Exception:
            continue

        lower = name.lower()
        if lower.startswith(("tarno", "python")):
            try:
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(pid)],
                    capture_output=True, check=False, timeout=5,
                )
                log.warning("Port %d freigegeben: %s (PID %d) beendet", port, name, pid)
                killed = True
            except Exception:
                pass

    if killed:
        time.sleep(0.5)
    return killed


class TarnoGrpcServer:
    """Wrapper that starts/stops the asyncio gRPC server."""

    def __init__(self, bridge: TarnoGrpcBridge, port: int = 50051) -> None:
        self.bridge = bridge
        self.port = port
        self._server: grpc.aio.Server | None = None

    def _make_server(self) -> grpc.aio.Server:
        server = grpc.aio.server(options=[
            # 16MB statt Default-4MB - Bild-Uploads (Phase A) koennen die
            # 4MB-Standardgrenze ueberschreiten.
            ("grpc.max_receive_message_length", 16 * 1024 * 1024),
            ("grpc.max_send_message_length", 16 * 1024 * 1024),
        ])
        tarno_pb2_grpc.add_TarnoServicer_to_server(
            TarnoGrpcServicer(self.bridge), server
        )
        return server

    async def _bind_and_start(self, server: grpc.aio.Server) -> None:
        # Loopback-only: the gRPC service has no authentication (it can write
        # API keys into the SecretsVault via SetApiKey), so binding to the
        # wildcard address ([::]) would expose it to the whole local network.
        # The WinUI client always connects via "localhost" (GrpcClientService),
        # so loopback binding does not change normal operation.
        server.add_insecure_port(f"127.0.0.1:{self.port}")
        server.add_insecure_port(f"[::1]:{self.port}")
        await server.start()

    async def start(self) -> None:
        self.bridge.set_event_loop(asyncio.get_running_loop())
        self._server = self._make_server()
        try:
            await self._bind_and_start(self._server)
        except Exception:
            # Meistens ein Port-Konflikt (z.B. eine Backend-Leiche von einem
            # vorherigen Absturz haelt den Port noch). Versuchen wir, den
            # Port freizugeben und binden neu.
            log.warning(
                "gRPC-Server konnte Port %d nicht binden; versuche Freigabe...",
                self.port,
            )
            _free_port(self.port)
            self._server = self._make_server()
            try:
                await self._bind_and_start(self._server)
            except Exception:
                log.exception(
                    "gRPC-Server konnte Port %d nicht binden (evtl. bereits belegt)",
                    self.port,
                )
                raise
        log.info("TARNO gRPC server listening on 127.0.0.1/::1 port %d", self.port)

    async def stop(self) -> None:
        if self._server:
            await self._server.stop(5)
            log.info("TARNO gRPC server stopped")


def create_bridge(config: TarnoConfig | None = None, mock: bool = False) -> TarnoGrpcBridge:
    """Create a bridge using either the real engine or the mock engine."""
    config = config or TarnoConfig.load()
    if mock:
        engine: TarnoEngine = MockTarnoEngine(config)
    else:
        engine = TarnoEngine(config, text_mode=True)
    return TarnoGrpcBridge(engine)


async def run_grpc_server(config: TarnoConfig | None = None, port: int = 50051, mock: bool = False) -> None:
    """Entry point used by `python -m tarno.grpc.server`."""
    logging.basicConfig(level=logging.INFO)
    bridge = create_bridge(config, mock=mock)
    server = TarnoGrpcServer(bridge, port)
    await server.start()
    try:
        await asyncio.Event().wait()
    finally:
        await server.stop()
        bridge.shutdown()
