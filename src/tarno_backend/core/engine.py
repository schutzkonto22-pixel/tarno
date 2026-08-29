"""Main orchestrator — wires all subsystems together and runs the voice loop."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from tarno_backend.ai.context.output_compressor import compress_tool_output_detailed
from tarno_backend.ai.context.summarizer import HistorySummarizer
from tarno_backend.ai.context.usage_tracker import ContextUsageTracker
from tarno_backend.ai.conversation import ConversationManager
from tarno_backend.ai.persona_guard import PersonaGuard
from tarno_backend.ai.factory import create_provider_with_fallback as _create_provider
from tarno_backend.ai.coding.agent import CodingAgent
from tarno_backend.ai.coding.tool import build_coding_task_tool
from tarno_backend.ai.provider import LLMResponse, ToolCall
from tarno_backend.ai.response_guard import guard_response
from tarno_backend.core.action_result import ActionResult
from tarno_backend.ai.prompts.code_system import CONTINUE_REMINDER
from tarno_backend.ai.prompts.proactive_system import PROACTIVE_SYSTEM_PROMPT
from tarno_backend.ai.tool_registry import ToolDefinition, ToolRegistry
from tarno_backend.plugins.manager import PluginManager
from tarno_backend.browser.browser_automation import (
    browser_click,
    browser_close,
    browser_navigate,
    browser_read,
    browser_type,
)
from tarno_backend.browser.web_control import fetch_webpage, open_browser, web_search
from tarno_backend.core.calendar_service import CalendarService
from tarno_backend.core.permission_service import AutonomyMode, PermissionService
from tarno_backend.core.command_tool import CommandTool
from tarno_backend.ui.confirmation_dialog import create_default_confirmation
from tarno_backend.core.events import (
    AgentTraceEvent,
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
from tarno_backend.core.proactive_engine import (
    IdleCuriosityObserver,
    ProactiveDraft,
    ProactiveEngine,
    SystemObserver,
    TimeCalendarObserver,
    UserBehaviorObserver,
)
from tarno_backend.integrations.mesh.observer import CloudMeshObserver, MeshObserver
from tarno_backend.memory.fact_extractor import FactExtractor
from tarno_backend.memory.knowledge_base import KnowledgeBase
from tarno_backend.memory.store import MemoryStore
from tarno_backend.telemetry.logging import new_correlation_id, set_correlation_id
from tarno_backend.desktop.app_control import open_application
from tarno_backend.desktop.file_manager import (
    copy_file,
    create_directory,
    delete_file,
    list_directory,
    move_file,
    open_file,
    read_file,
    search_files,
    write_file,
)
from tarno_backend.desktop.screenshot import take_screenshot
from tarno_backend.desktop.system_info import get_system_info
from tarno_backend.desktop.window_manager import close_window, focus_window
from tarno_backend.extensions.coordinator import ExtensionCoordinator
from tarno_backend.security.audit import AuditManager
from tarno_backend.ui.console import ConsoleUI
from tarno_backend.utils.text import is_exit
from tarno_backend.voice.audio_stream import AudioStream, AudioStreamSource
from tarno_backend.voice.echo_protection import EchoProtection
from tarno_backend.voice.synthesizer import SpeechSynthesizer

if TYPE_CHECKING:
    from tarno_backend.core.config import TarnoConfig
    from tarno_backend.vision.vision_observer import VisionObserver
    from tarno_backend.voice.recognizer import SpeechRecognizer

log = logging.getLogger(__name__)


class TarnoEngine:
    """Owns the lifecycle of all subsystems and runs the main loop."""

    def __init__(self, config: TarnoConfig, text_mode: bool = False) -> None:
        set_correlation_id(new_correlation_id())
        self.config = config
        self.event_bus = EventBus()
        self._running = False

        try:
            autonomy_mode = AutonomyMode(self.config.command_execution.autonomy_mode)
        except ValueError:
            log.warning(
                "Unbekannter autonomy_mode '%s', verwende 'balanced'",
                self.config.command_execution.autonomy_mode,
            )
            autonomy_mode = AutonomyMode.BALANCED
        self._permission_service = PermissionService(
            dialog_factory=create_default_confirmation(prefer_qt=True),
            mode=autonomy_mode,
        )

        self.provider = _create_provider(config)
        # Phase C: "Denk-Modus"-Umschalter - fordert aktiv strukturierten
        # Denk-Inhalt an (kostet mehr Tokens/Latenz), im Unterschied zum
        # kostenlosen, passiven <think>-Capture (extract_reasoning). None =
        # aus; Provider ohne reasoning_effort-Unterstuetzung ignorieren den
        # Wert ohnehin still.
        self.reasoning_effort: str | None = None
        # Manueller "Starkes Modell"-Schalter (ersetzt das frueher automatisch
        # per Klassifizierer getriggerte High-Tier, das faelschlich auf
        # Wortlaenge/Schluesselwoerter reagierte - siehe MistralProvider._classify_difficulty).
        self.use_high_tier_model: bool = False
        # Chat-Umschalter "Vorlesen" - schaltet TTS fuer normale Antworten
        # ab, damit der naechste Chat-Turn nicht auf die laufende
        # Sprachwiedergabe warten muss (siehe SetSpeakResponses im Proto).
        # Default True = bisheriges Verhalten unveraendert.
        self.speak_responses_enabled: bool = True
        # Kontext-Effizienz (Phase 3-5): pro-Chat Kontext-Nutzungs-Tracker
        # (mirrort self._conversations, ein Tracker je chat_id) + ein
        # gemeinsamer, lazy gebauter HistorySummarizer (haelt selbst keinen
        # Zustand - nimmt Nachrichten/vorherige Zusammenfassung als Parameter).
        # Default aus - der echte Nutzer-Schalter kommt in Phase 6, siehe
        # set_context_efficiency().
        self._usage_trackers: dict[str, ContextUsageTracker] = {}
        self._history_summarizer: HistorySummarizer | None = None
        self._context_efficiency_enabled: bool = config.context_efficiency.enabled
        # Phase 1-6: aktuell aktives Modell, falls per set_provider() explizit
        # gewaehlt (None = Provider-Default aus der Config).
        self.active_model: str | None = None
        self.memory_store: MemoryStore | None = None
        self.knowledge_base: KnowledgeBase | None = None
        if config.memory.enabled:
            storage_dir = Path(config.memory.storage_dir).expanduser()
            storage_dir.mkdir(parents=True, exist_ok=True)
            self.memory_store = MemoryStore(storage_dir / "memory.db")
            self.knowledge_base = KnowledgeBase(storage_dir / "kb.db")
        self._conversations: dict[str, ConversationManager] = {}
        self._current_chat_id = "default"
        self._current_turn_id = ""
        self._active_workspace: dict[str, Any] | None = None
        self._ensure_conversation("default")
        self.tools = ToolRegistry()
        self._fact_extractor = FactExtractor()
        self._persona_guard = PersonaGuard()
        self._echo_protection = EchoProtection()
        self.synthesizer = SpeechSynthesizer(
            config.audio,
            on_tts_started=self._on_tts_started,
            on_tts_finished=self._on_tts_finished,
        )

        self.ui = ConsoleUI(self.event_bus, text_mode=text_mode)

        self._coding_agent = CodingAgent(
            self.config,
            event_bus=self.event_bus,
            get_workspace=lambda: self._active_workspace,
            permission_service=self._permission_service,
        )

        self._register_default_tools()
        self._load_plugins()

        self._extensions = ExtensionCoordinator(
            tool_executor=self.tools.execute,
            reminder_callback=self._on_reminder_due,
        )
        if self.config.briefing.enabled:
            self._extensions.set_briefing_times(
                self.config.briefing.times,
                self._run_briefing,
            )

        self._audit_manager = AuditManager(self.config.command_execution.log_dir)

        self._last_interaction_at = datetime.now()
        self._vision_observer: "VisionObserver | None" = None
        self._proactive_engine = self._build_proactive_engine()

        log.info(
            "TARNO Engine initialisiert (provider=%s)",
            self.provider.name,
        )

    @property
    def conversation(self) -> ConversationManager:
        """Aktive Conversation fuer das gerade bearbeitete chat_id."""
        return self._ensure_conversation(self._current_chat_id)

    @property
    def _workspace_root(self) -> str | None:
        """Primary root path of the active workspace, if any."""
        if self._active_workspace is None:
            return None
        root_paths = self._active_workspace.get("root_paths", [])
        if root_paths:
            return root_paths[0]
        return self._active_workspace.get("root_path")

    def _ensure_conversation(self, chat_id: str) -> ConversationManager:
        if chat_id not in self._conversations:
            self._conversations[chat_id] = ConversationManager(
                self.config.llm.max_history,
                memory=self.config.memory,
                memory_store=self.memory_store,
                knowledge_base=self.knowledge_base,
                tts_config=self.config.audio.tts_prompt,
            )
        return self._conversations[chat_id]

    def _build_proactive_engine(self) -> ProactiveEngine:
        """Block 3 (Phasen 21-30): autonome Trigger-Engine.

        Always constructed and started, even with zero initial observers -
        this gives Block 7's live camera opt-in (set_vision_enabled) a
        running tick loop to attach to at any time, not just when
        config.vision.enabled was already true at startup. The privacy-
        relevant "no unsolicited autonomous nudges by default" guarantee is
        preserved by simply not adding the calendar/system/behavior/idle
        observers unless config.proactive.enabled is true - an engine with
        an empty observer list does nothing but sleep every tick.
        """
        cfg = self.config.proactive
        observers: list = []

        if cfg.enabled:
            calendar = CalendarService(self.config.briefing.calendar_file)
            observers = [
                TimeCalendarObserver(calendar, lookahead_minutes=cfg.calendar_lookahead_minutes),
                SystemObserver(
                    cpu_threshold=cfg.cpu_threshold_percent,
                    ram_threshold=cfg.ram_threshold_percent,
                    disk_threshold=cfg.disk_threshold_percent,
                ),
                UserBehaviorObserver(
                    distraction_apps=cfg.distraction_apps,
                    calendar_service=calendar,
                    minutes_before_action=cfg.distraction_minutes_before_action,
                    calendar_lookahead_minutes=cfg.calendar_lookahead_minutes,
                    close_distraction_window=cfg.close_distraction_window,
                ),
                IdleCuriosityObserver(
                    task_planner=self._extensions.task_planner,
                    get_last_interaction=lambda: self._last_interaction_at,
                    idle_minutes=cfg.idle_minutes_for_curiosity,
                ),
            ]

            # Mesh-Sprachreaktionen (lokaler Hub-Failover UND Cloud-Geraete-
            # Online/Offline) - eigenes Opt-in ueber mesh.enabled, damit
            # "proactive.enabled" allein noch keine Mesh-Kommentare ausloest,
            # falls der Nutzer das Mesh-Feature separat deaktiviert hat.
            # MeshObserver war bereits vorher fertig implementiert (siehe
            # integrations/mesh/observer.py), aber nie hier verdrahtet -
            # CloudMeshObserver ist die neue Ergaenzung fuer die Tarno-Server-
            # Geraetetracking-Daten (siehe integrations/mesh/cloud_client.py).
            if self.config.mesh.enabled:
                observers.append(MeshObserver(self.event_bus))
                observers.append(CloudMeshObserver(self.config.mesh.cloud_server_url))

        if self.config.vision.enabled:
            self._vision_observer = self._build_vision_observer()
            if self._vision_observer is not None:
                observers.append(self._vision_observer)

        return ProactiveEngine(
            observers=observers,
            on_trigger=self._on_proactive_trigger,
            tick_seconds=cfg.tick_seconds,
            score_threshold=cfg.score_threshold,
            source_cooldown_seconds=cfg.source_cooldown_seconds,
        )

    def _build_vision_observer(self, force: bool = False) -> "VisionObserver | None":
        """Block 7 (Phasen 61-70): vierter Observer, backed by einer Webcam.
        Deaktiviert per Default (config.vision.enabled=False) - Kamera-
        Dauererfassung + Cloud-Bildanalyse sind privacy-sensibel genug für
        einen expliziten Opt-in (siehe VisionConfig).

        *force* bypasses the config.vision.enabled check - used by
        set_vision_enabled() for the live opt-in toggle (Phase 69), where
        the user explicitly requests the camera turn on right now,
        independent of whatever the static startup config said."""
        cfg = self.config.vision
        if not cfg.enabled and not force:
            return None

        from tarno_backend.security.secrets import SecretsVault
        from tarno_backend.vision.camera_capture import CameraCapture
        from tarno_backend.vision.vision_observer import VisionObserver
        from tarno_backend.vision.vision_provider import MistralVisionProvider

        vault = SecretsVault(backend=self.config.security.secrets_backend)
        api_key = vault.get("MISTRAL_API_KEY") or ""

        camera = CameraCapture(device_index=cfg.device_index, device_name=cfg.device_name)
        vision_provider = MistralVisionProvider(api_key=api_key, model=cfg.vision_model)
        if not vision_provider.available:
            log.warning(
                "vision.enabled=true, aber kein Mistral-API-Key konfiguriert - "
                "Vision-Observer bleibt inaktiv."
            )
            return None

        return VisionObserver(
            camera=camera,
            vision_provider=vision_provider,
            motion_threshold=cfg.motion_threshold,
            candidate_frames=cfg.candidate_frames,
            downscale_max_edge=cfg.downscale_max_edge,
            jpeg_quality=cfg.jpeg_quality,
        )

    def set_vision_enabled(self, enabled: bool) -> bool:
        """Block 7, Phase 69: live opt-in/opt-out toggle for the gRPC bridge.

        Enabling builds the vision observer on demand (camera + vision
        model) if it doesn't exist yet - config.vision.enabled only governs
        whether it's already active when TARNO launches, it's no longer a
        hard requirement for turning the camera on later via the UI toggle.
        Returns False only if enabling was requested but no camera/API key
        is actually available.
        """
        if enabled:
            if self._vision_observer is None:
                self._vision_observer = self._build_vision_observer(force=True)
                if self._vision_observer is None:
                    return False  # no camera or no API key - genuinely can't enable
                self._proactive_engine.add_observer(self._vision_observer)
            self._vision_observer.set_enabled(True)
            return True

        if self._vision_observer is not None:
            self._vision_observer.set_enabled(False)
        return True

    def set_provider(self, provider_name: str, model: str | None = None) -> tuple[bool, str]:
        """Phase D/1-6: switches the active LLM provider (and optionally a
        specific model from the curated catalog) at runtime. Uses
        create_provider (NOT create_provider_with_fallback) - an explicit
        user-requested switch should activate exactly the chosen provider,
        not silently land on a fallback chain. self.provider is only
        reassigned on full success, so a failed switch never has a
        side effect (old provider stays active, matches the
        AutonomyMode/Vision-toggle "no unnoticed drift" pattern)."""
        from tarno_backend.ai.factory import create_provider, provider_has_api_key
        from tarno_backend.security.secrets import SecretsVault

        name = provider_name.lower()
        vault = SecretsVault(backend=self.config.security.secrets_backend)
        if not provider_has_api_key(vault, name):
            return False, f"Kein API-Key für '{name}' hinterlegt — bitte zuerst in den Einstellungen eintragen."

        try:
            new_provider = create_provider(self.config, name, model=model or None)
        except Exception as exc:
            log.warning("Provider-Wechsel zu '%s' fehlgeschlagen: %s", name, exc)
            return False, f"Provider '{name}' konnte nicht aktiviert werden: {exc}"

        self.provider = new_provider
        self.active_model = model or None
        log.info("Provider gewechselt auf '%s' (Modell: %s)", name, model or "Standard")
        return True, f"Provider auf '{name}' gewechselt."

    def set_reasoning_mode(self, enabled: bool, effort: str = "medium") -> None:
        """Phase C: toggles the active reasoning_effort for future turns.
        Providers/models without support silently ignore the value (see
        MistralProvider._REASONING_CAPABLE_MODELS), so this is safe to call
        regardless of the currently active provider."""
        self.reasoning_effort = effort if enabled else None

    def set_high_tier_model(self, enabled: bool) -> None:
        """Manueller Schalter fuer Mistral's High-Tier-Modell (mistral-medium-2508) -
        umgeht die automatische Klassifizierung komplett. Bei anderen
        Providern wirkungslos (ProviderX.send() ignoriert use_high_tier still)."""
        self.use_high_tier_model = enabled

    def set_speak_responses(self, enabled: bool) -> None:
        """Chat-Umschalter "Vorlesen": bei False wird speak() fuer normale
        Antworten uebersprungen, siehe speak_responses_enabled."""
        self.speak_responses_enabled = enabled

    def set_context_efficiency(self, enabled: bool) -> bool:
        """Chat-Umschalter "Kontext-Effizienz" (Beta) - schaltet Tool-Output-
        Kompression (Phase 2) und rekursive Verlauf-Zusammenfassung
        (Phase 5) fuer den normalen Chat-Modus ein/aus. Default aus, aendert
        also das bestehende Chat-Verhalten nur nach explizitem Opt-in.
        Gibt den tatsaechlich gesetzten Wert zurueck (fuer das Server-Echo)."""
        self._context_efficiency_enabled = enabled
        log.info("[CtxEff] Kontext-Effizienz %s", "aktiviert" if enabled else "deaktiviert")
        return self._context_efficiency_enabled

    @property
    def vision_camera_active(self) -> bool:
        """True only while the camera hardware is actually capturing."""
        return self._vision_observer is not None and self._vision_observer.is_capturing

    @property
    def vision_available(self) -> bool:
        return self._vision_observer is not None

    def _embellish_proactive_message(self, draft: ProactiveDraft) -> str:
        """Rephrases a factual ProactiveDraft.message with a bit of JARVIS-style
        personality (see prompts/proactive_system.py) - the underlying fact
        stays whatever the observer already determined, only the phrasing/
        an optional suggestion is LLM-generated. Best-effort: any failure
        (provider down, timeout, no tool support needed here but a genuine
        API error) falls back to the observer's own plain-factual message
        rather than ever blocking or skipping a proactive notice because the
        embellishment step itself broke."""
        try:
            response = self.provider.send(
                messages=[{
                    "role": "user",
                    "content": f"Beobachtung von '{draft.source}': {draft.message}",
                }],
                system=PROACTIVE_SYSTEM_PROMPT,
            )
            text = (response.text or "").strip()
            return text if text and not response.is_error else draft.message
        except Exception:
            log.exception("Proaktive Umformulierung fehlgeschlagen - nutze Rohtext")
            return draft.message

    def _on_proactive_trigger(self, draft: ProactiveDraft) -> None:
        """Speaks a triggered proactive draft the same way a normal turn would
        (chat bubble + TTS via the existing ResponseReadyEvent pipeline), and
        executes its optional follow-up action (currently only close_window,
        Phase 28) through the gated CommandTool path so it still respects the
        current AutonomyMode rather than bypassing the risk router."""
        message = self._embellish_proactive_message(draft)
        self.conversation.add_assistant_response(message)
        self.event_bus.publish(ResponseReadyEvent(text=message))
        if self.speak_responses_enabled:
            self.synthesizer.speak(message)
        self._remember_proactive_episode(draft)

        if draft.action == "close_window":
            result = self.command_tool.close_window_gated(
                draft.action_params.get("title_substring", ""),
                user_query=f"[proaktiv: {draft.source}]",
            )
            log.info("Proaktive Aktion 'close_window': %s", result.message)

    def _on_tts_started(self, text: str, envelope: list[float], duration: float) -> None:
        self._echo_protection.on_tts_started()
        self.event_bus.publish(TtsStartedEvent(text=text, envelope=envelope, duration_seconds=duration))

    def _on_tts_finished(self) -> None:
        self._echo_protection.on_tts_finished()
        self.event_bus.publish(TtsFinishedEvent())

    def run(self) -> None:
        """Full voice loop: wake word -> listen -> process -> speak."""
        from tarno_backend.voice.recognizer import SpeechRecognizer

        self._audit_manager.save_state()
        self._extensions.start()
        if self._proactive_engine is not None:
            self._proactive_engine.start()
        self.ui.show_banner()
        self._running = True

        recognizer = SpeechRecognizer(self.config.audio, vad_config=self.config.vad, agc_config=self.config.agc)

        if self.config.wakeword.enabled:
            self._run_with_wakeword(recognizer)
        else:
            self._run_push_to_talk(recognizer)

    def _run_with_wakeword(self, recognizer: "SpeechRecognizer") -> None:
        from tarno_backend.voice.wakeword import WakeWordDetector

        wakeword = WakeWordDetector(self.config.wakeword, agc_config=self.config.agc)

        self.synthesizer.speak(
            "TARNO ist nun vollständig online, sir. Die Protokolle sind kalibriert. "
            "Ich würde vorschlagen, dass Sie versuchen, meine Aufmerksamkeit nicht unmittelbar mit logischen Inkonsistenzen zu beleidigen."
        )

        try:
            with AudioStream(self.config.audio) as stream:
                log.info("Wake-Word-Erkennung aktiv — warte auf 'Hey Tarno'...")
                recognizer.calibrate(AudioStreamSource(stream))

                while self._running:
                    chunk = stream.read_chunk()
                    if self._echo_protection.is_suppressed():
                        # TARNO is currently speaking (or just finished, within
                        # the cooldown window) — skip wake-word processing so it
                        # cannot hear and re-trigger on its own TTS output.
                        continue
                    if not wakeword.process_frame(chunk):
                        continue

                    wakeword.reset()
                    self.event_bus.publish(WakeWordDetectedEvent())

                    if self.config.wakeword.confirm_wake_word:
                        self.synthesizer.play_sound()

                    self._conversation_window(recognizer, stream, wakeword)
                    stream.flush_input_buffer()
                    wakeword.reset()

        except KeyboardInterrupt:
            self._shutdown()

    def _conversation_window(
        self,
        recognizer: "SpeechRecognizer",
        stream: "Any",
        wakeword: "Any",
    ) -> None:
        """Keep listening for follow-up commands until silence timeout."""
        in_conversation = True

        while in_conversation and self._running:
            user_text = recognizer.listen_and_recognize(AudioStreamSource(stream))

            if not user_text:
                log.info("Kein Follow-up erkannt — zurück zum Wake-Word-Modus")
                print("  [Zurück zum Wake-Word-Modus — sagen Sie 'Hey Tarno']")
                wakeword.reset()
                return

            if is_exit(user_text):
                self._shutdown()
                return

            self._handle_input(user_text)

            print("  [Ich höre weiter zu... oder sagen Sie nichts für Wake-Word-Modus]")

    def _run_push_to_talk(self, recognizer: "SpeechRecognizer") -> None:
        self.synthesizer.speak(
            "Ich höre zu, sir. Es wäre erfrischend, wenn Ihre Eingabe zur Abwechslung eine klare Struktur aufweisen würde."
        )

        try:
            while self._running:
                user_text = recognizer.listen_and_recognize()
                if not user_text:
                    continue

                if is_exit(user_text):
                    self._shutdown()
                    break

                self._handle_input(user_text)

        except KeyboardInterrupt:
            self._shutdown()

    def run_text_mode(self) -> None:
        """Interactive text loop — no microphone needed."""
        self._audit_manager.save_state()
        self._extensions.start()
        if self._proactive_engine is not None:
            self._proactive_engine.start()
        self.ui.show_banner()
        print(f"  [Text-Modus · Provider: {self.provider.name}]\n")
        self._running = True

        try:
            while self._running:
                try:
                    user_input = input("  Sie: ").strip()
                except EOFError:
                    break

                if not user_input:
                    continue

                if is_exit(user_input):
                    self._shutdown()
                    break

                self._handle_input(user_input)

        except KeyboardInterrupt:
            self._shutdown()

    def _speak_if_not_stale(self, text: str, is_stale: Callable[[], bool] | None) -> None:
        """Finalizes a turn's spoken output - unless the caller (the gRPC
        bridge) has since marked this specific call as stale (the user
        cancelled it AND the request-level watchdog already gave up waiting
        for it, see TarnoGrpcBridge._cleanup_orphaned_generation). Without
        this check, a request that hangs past the watchdog timeout but
        eventually completes in its orphaned background thread would still
        speak/broadcast a long-irrelevant answer with nothing left to
        suppress it (found via live testing: a cancelled request "Datei über
        Vögel" spoke "Mistral API nicht erreichbar" ~3 minutes after the
        user had already moved on)."""
        if is_stale is not None and is_stale():
            log.info("Antwort verworfen (veraltete/abgebrochene Anfrage): %r", text[:80])
            return
        self.event_bus.publish(ResponseReadyEvent(
            text=text,
            chat_id=self._current_chat_id,
            turn_id=self._current_turn_id or uuid.uuid4().hex,
        ))
        if self.speak_responses_enabled:
            self.synthesizer.speak(text)

    def _send_stream(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]] | None,
        trace_id: str,
        stop_heartbeat: threading.Event,
    ) -> LLMResponse:
        """Streams a chat completion, emits AgentTrace thought deltas for every
        token and returns the assembled LLMResponse."""
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        is_error = False
        first = True

        for delta in self.provider.send_stream(
            messages=messages,
            system=system,
            tools=tools,
            reasoning_effort=self.reasoning_effort,
            use_high_tier=self.use_high_tier_model,
        ):
            if delta.is_error:
                is_error = True

            if delta.text:
                if first:
                    stop_heartbeat.set()
                text_parts.append(delta.text)
                self.event_bus.publish(AgentTraceEvent(
                    trace_id=trace_id,
                    turn_id=self._current_turn_id,
                    chat_id=self._current_chat_id,
                    trace_type="thought_delta",
                    payload={
                        "title": "Antwort wird generiert..." if first else "",
                        "delta": delta.text,
                        "finished": False,
                        "collapsed": False,
                    },
                ))
                if first:
                    first = False

            if delta.tool_call:
                tool_calls.append(delta.tool_call)

            if delta.is_finished:
                break

        return LLMResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            is_error=is_error,
        )

    def _handle_input(
        self,
        user_text: str,
        image_data_url: str | None = None,
        chat_id: str = "default",
        turn_id: str = "",
        is_stale: Callable[[], bool] | None = None,
        workspace: dict[str, Any] | None = None,
        chat_mode: str | None = None,
    ) -> None:
        self._current_chat_id = chat_id
        self._current_turn_id = turn_id or uuid.uuid4().hex
        self._active_workspace = workspace
        self._ensure_conversation(self._current_chat_id)
        self.conversation.set_workspace(workspace)
        self.conversation.set_chat_mode(chat_mode)

        self._last_interaction_at = datetime.now()
        self.event_bus.publish(SpeechRecognizedEvent(text=user_text))
        self._auto_extract_facts(user_text)

        routine_result = self._extensions.try_routine(user_text)
        if routine_result is not None:
            self.conversation.add_assistant_response(routine_result.message)
            self._speak_if_not_stale(routine_result.message, is_stale)
            return

        # Vision-Anfragen direkt ausführen. Das Modell hat mit tool_choice=auto
        # wiederholt vergessen, describe_what_i_see aufzurufen, weil die
        # Historie voller gescheiterter Vision-Versuche ist. Ein direkter
        # Pfad umgeht diesen Ausfall zuverlässig.
        lower_text = user_text.lower()
        if any(
            trigger in lower_text
            for trigger in ["was siehst du", "siehst du mich", "was kannst du sehen", "schau mal"]
        ):
            self.conversation.add_user_message(user_text)
            result = self.tools.execute("describe_what_i_see", {})
            # Store a real tool_use/tool_result exchange (synthetic call, but
            # a truthful record of what actually happened) instead of a bare
            # text exchange - found live: a plain-text-only history (no
            # tool-call shape to imitate) left the model with zero example of
            # what a correct describe_what_i_see call looks like, and a
            # later ambiguous follow-up ("genauer") with a different model
            # (mistral-large-2512) then emitted a garbled pseudo-tool-call as
            # plain text instead of a real structured call.
            tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
            self.conversation.add_assistant_tool_use(
                LLMResponse(text="", tool_calls=[ToolCall(id=tool_call_id, name="describe_what_i_see", input={})])
            )
            if result.success:
                self.conversation.add_tool_result(tool_call_id, result.message, tool_name="describe_what_i_see")
                self.conversation.add_assistant_response(result.message)
                spoken = guard_response(result.message, user_text)
            else:
                fallback = result.message or "Ich konnte gerade kein Kamerabild erfassen."
                self.conversation.add_tool_result(tool_call_id, fallback, tool_name="describe_what_i_see")
                self.conversation.add_assistant_response(fallback)
                spoken = guard_response(fallback, user_text)
            if spoken:
                self._speak_if_not_stale(spoken, is_stale)
            return

        self.conversation.add_user_message(user_text, image_data_url=image_data_url)
        self.conversation.inject_relevant_memory(user_text)
        self.conversation.inject_relevant_kb(user_text)
        if self._persona_guard.should_reanchor():
            self.conversation.inject_persona_reminder(self._persona_guard.reanchor_text)

        print("\n  TARNO denkt nach...", end="", flush=True)
        self.event_bus.publish(ProcessingStageEvent(stage="llm_call", detail="Rufe LLM auf..."))

        tools = self.tools.get_tool_schemas() if self.provider.supports_tools else None

        trace_id = uuid.uuid4().hex
        stop_heartbeat = threading.Event()
        started_at = time.monotonic()

        def _heartbeat() -> None:
            while not stop_heartbeat.wait(0.5):
                elapsed = time.monotonic() - started_at
                self.event_bus.publish(AgentTraceEvent(
                    trace_id=trace_id,
                    turn_id=self._current_turn_id,
                    chat_id=self._current_chat_id,
                    trace_type="thought_delta",
                    payload={
                        "title": f"Rufe LLM auf... ({elapsed:.1f}s)",
                        "delta": "",
                        "finished": False,
                        "collapsed": False,
                    },
                ))

        heartbeat = threading.Thread(target=_heartbeat, daemon=True)
        heartbeat.start()

        try:
            if self.provider.capabilities.streaming:
                response = self._send_stream(
                    messages=self.conversation.get_messages(),
                    system=self.conversation.system_prompt,
                    tools=tools,
                    trace_id=trace_id,
                    stop_heartbeat=stop_heartbeat,
                )
            else:
                response = self.provider.send(
                    messages=self.conversation.get_messages(),
                    system=self.conversation.system_prompt,
                    tools=tools,
                    reasoning_effort=self.reasoning_effort,
                    use_high_tier=self.use_high_tier_model,
                )
        except Exception as exc:
            print()
            log.exception("LLM Fehler")
            stop_heartbeat.set()
            heartbeat.join(timeout=1.0)
            self.event_bus.publish(ErrorEvent(error=exc, module=self.provider.name))
            self._speak_if_not_stale(
                "Es scheint, als wäre die telepathische Verbindung zum Backend unterbrochen, sir. Ich würde vermuten, dass die Realität schlichtweg vor Ihrer letzten Anfrage kapituliert hat.",
                is_stale,
            )
            return

        stop_heartbeat.set()
        heartbeat.join(timeout=1.0)
        self.event_bus.publish(AgentTraceEvent(
            trace_id=trace_id,
            turn_id=self._current_turn_id,
            chat_id=self._current_chat_id,
            trace_type="thought_delta",
            payload={
                "title": "Antwort erhalten",
                "delta": "",
                "finished": True,
                "collapsed": True,
            },
        ))
        print("\r" + " " * 30 + "\r", end="", flush=True)

        if response.is_error:
            self.event_bus.publish(VoiceErrorEvent(message=response.text))

        self._maybe_publish_context_usage(response)
        self._maybe_trigger_summarization()
        if response.reasoning:
            self.event_bus.publish(ProcessingStageEvent(stage="reasoning", reasoning=response.reasoning))
        spoken = self._process_response(response, user_text, is_stale=is_stale)

        if spoken:
            self._speak_if_not_stale(spoken, is_stale)

    def _remember_proactive_episode(self, draft: ProactiveDraft) -> None:
        """Block 4, Phase 36: syncs the autonomous trigger engine (Block 3)
        with the memory layer. Every proactive interjection is recorded as an
        episode, tagged by its originating observer, so it resurfaces later
        via the existing MemoryRetriever/inject_relevant_memory() path
        ("Tarno erinnerte dich letzten Dienstag daran, dass...") instead of
        being forgotten the moment it's spoken."""
        if self.memory_store is None:
            return
        from tarno_backend.memory.embeddings import get_default_provider
        embedding = get_default_provider().embed_one(draft.message)
        self.memory_store.save_episode(
            summary=draft.message, tags=["proaktiv", draft.source], embedding=embedding,
        )

    def _on_reminder_due(self, message: str) -> None:
        """Fires when a scheduled reminder (add_reminder tool) becomes due -
        checked every minute by ExtensionCoordinator's scheduler. Behaves
        like a proactive trigger (Block 3): visible chat bubble + TTS, not
        just a silent audio-only notification, so the user can see it was
        actually said even if they weren't listening."""
        self.conversation.add_assistant_response(message)
        self.event_bus.publish(ResponseReadyEvent(text=message))
        if self.speak_responses_enabled:
            self.synthesizer.speak(message)

    def _add_reminder_tool(self, message: str, time: str, date: str | None = None) -> ActionResult:
        """LLM-facing tool: schedules a real time-triggered reminder via
        ReminderEngine (checked every minute, spoken + shown when due).

        Deliberately takes structured (time, optional date) rather than a
        single freeform datetime string the model would have to compute
        itself - an earlier live test showed the model getting "today's
        date" wrong when asked to produce a full ISO datetime on its own
        (e.g. writing 2024-07-23 for what was actually 2026-07-19). Today's
        date and the "already passed -> tomorrow" rollover are computed
        here in code instead, where they can't be hallucinated.
        """
        trigger_time, error = self._resolve_reminder_time(time, date, now=datetime.now())
        if error:
            return error

        self._extensions.reminder_engine.add(user_query=message, trigger_time=trigger_time, message=message)
        return ActionResult(
            success=True,
            message=f"Erinnerung gesetzt für {trigger_time.strftime('%d.%m.%Y %H:%M')}: {message}",
        )

    @staticmethod
    def _resolve_reminder_time(
        time: str, date: str | None, now: datetime,
    ) -> tuple[datetime | None, ActionResult | None]:
        """Pure computation, `now` injected for testability: turns (time,
        optional date) into an absolute trigger datetime, or an error
        ActionResult. Returns (trigger_time, None) on success, (None, error)
        on failure."""
        try:
            hour, minute = (int(part) for part in time.split(":", 1))
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                raise ValueError
        except ValueError:
            return None, ActionResult(
                success=False,
                message=f"Ungültiges Zeitformat: '{time}' (erwartet HH:MM, z.B. '19:20')",
                error_code="InvalidTime",
            )

        if date:
            try:
                target_date = datetime.strptime(date, "%Y-%m-%d").date()
            except ValueError:
                return None, ActionResult(
                    success=False,
                    message=f"Ungültiges Datumsformat: '{date}' (erwartet YYYY-MM-DD)",
                    error_code="InvalidDate",
                )
            trigger_time = datetime.combine(target_date, datetime.min.time()).replace(hour=hour, minute=minute)
            if trigger_time <= now:
                return None, ActionResult(
                    success=False,
                    message=(
                        f"Das Datum '{date}' liegt in der Vergangenheit (aktuelles Datum: "
                        f"{now:%Y-%m-%d %H:%M}). Bitte gib ein zukünftiges Datum an oder "
                        "lasse das Datum weg, wenn 'heute'/'morgen' zur genannten Uhrzeit gemeint ist."
                    ),
                    error_code="PastDate",
                )
        else:
            trigger_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if trigger_time <= now:
                trigger_time += timedelta(days=1)

        return trigger_time, None

    def _auto_extract_facts(self, user_text: str) -> None:
        """Block 4, Phase 32: automatic memory extraction after each user
        message. Purely local regex heuristics (see FactExtractor) - no extra
        LLM call. Complements the LLM's own explicit remember_fact tool call."""
        for key, value in self._fact_extractor.extract(user_text):
            self.conversation.remember_fact(key, value, source="auto_extracted")
            log.debug("Fakt automatisch extrahiert: %s = %s", key, value)

    def _remember_fact_tool(self, key: str, value: str) -> ActionResult:
        """LLM-facing tool handler - lets TARNO consciously decide to keep a
        fact the auto-extractor's narrow patterns wouldn't catch."""
        self.conversation.remember_fact(key, value, source="llm")
        return ActionResult(success=True, message=f"Fakt gespeichert: {key} = {value}")

    def _describe_what_i_see_tool(self) -> ActionResult:
        """LLM-facing tool handler for direct on-demand camera queries ("was
        siehst du gerade?"). Distinct from the Block 7 VisionObserver's
        passive, motion-triggered background loop - that one only speaks up
        autonomously and has no connection to the normal tool-call path, so
        a direct question got "keine visuelle Wahrnehmung" even with the
        camera running (found via live testing). Builds the vision observer
        on demand (same lazy construction as set_vision_enabled) if it
        doesn't exist yet, so this also works before the user has ever
        toggled the camera on."""
        if self._vision_observer is None:
            self._vision_observer = self._build_vision_observer(force=True)
            if self._vision_observer is None:
                return ActionResult(
                    success=False,
                    message="Keine Kamera oder kein Mistral-API-Key verfügbar - ich kann gerade nichts sehen.",
                )

        result = self._vision_observer.capture_and_describe()
        if result is None:
            return ActionResult(
                success=False,
                message="Ich konnte gerade kein Kamerabild erfassen oder auswerten.",
            )
        description, jpeg_bytes = result
        if jpeg_bytes:
            self.event_bus.publish(VisionAttachmentEvent(image_bytes=jpeg_bytes, source="camera"))
        return ActionResult(success=True, message=description, attachment=jpeg_bytes)

    def _maybe_publish_context_usage(self, response: LLMResponse) -> None:
        """Broadcast context-window usage for the UI's token indicator, if
        the provider reported it for this turn (not all providers/calls do).

        Also feeds the same numbers into this chat's ContextUsageTracker
        (Kontext-Effizienz Phase 3/5) - same choke point, so the "~75%"
        summarization trigger stays in sync with what the UI's token ring
        shows."""
        if response.tokens_used is not None and response.context_window is not None:
            self.event_bus.publish(ContextUsageEvent(
                tokens_used=response.tokens_used,
                context_window=response.context_window,
            ))
        tracker = self._get_usage_tracker(self._current_chat_id)
        if response.tokens_used is not None and response.context_window is not None:
            tracker.record(response.tokens_used, response.context_window)
        else:
            tracker.record_estimate(self.conversation.get_messages())

    def _get_usage_tracker(self, chat_id: str) -> ContextUsageTracker:
        """Eine ContextUsageTracker-Instanz pro Chat, mirrort _ensure_conversation
        (jeder Chat hat seinen eigenen ConversationManager UND damit auch
        seine eigene Kontext-Fuellstand-Verfolgung)."""
        if chat_id not in self._usage_trackers:
            self._usage_trackers[chat_id] = ContextUsageTracker()
        return self._usage_trackers[chat_id]

    def _maybe_trigger_summarization(self) -> None:
        """Kontext-Effizienz Phase 5: loest bei Bedarf eine rekursive
        Verlauf-Zusammenfassung aus. Inert, solange _context_efficiency_enabled
        False ist (Default - der echte Toggle folgt in Phase 6)."""
        if not self._context_efficiency_enabled:
            return
        tracker = self._get_usage_tracker(self._current_chat_id)
        if not tracker.should_summarize():
            return

        if self._history_summarizer is None:
            self._history_summarizer = HistorySummarizer(self.config)

        conversation = self.conversation
        messages = conversation.get_messages()
        previous_summary = conversation.get_current_summary_for_debug()
        try:
            new_summary = self._history_summarizer.summarize(messages, previous_summary)
        except Exception:
            log.exception("[CtxEff] Verlauf-Zusammenfassung fehlgeschlagen, wird uebersprungen")
            return

        log.debug(
            "[CtxEff] Verlauf zusammengefasst (chat_id=%s, %d Nachrichten, ratio=%.2f, "
            "geschätzt=%s):\nVorher: %s\nNachher: %s",
            self._current_chat_id, len(messages), tracker.last_known_ratio,
            tracker.last_ratio_is_estimate, previous_summary, new_summary,
        )
        conversation.replace_with_recursive_summary(new_summary)
        tracker.reset_after_summary()

    def _maybe_compress_tool_result(self, tool_name: str, message: str) -> str:
        """Kontext-Effizienz Phase 2/6: komprimiert Tool-Output (RTK-Prinzip)
        bevor er per add_tool_result zurueck in den LLM-Kontext fliesst.

        Immer aktiv im Code-Modus (wo Kommandoausgaben ueberhaupt am
        haeufigsten vorkommen); im normalen Chat-Modus nur, wenn der Nutzer
        den "Kontext-Effizienz"-Schalter explizit aktiviert hat (Phase 6) -
        Default aus, aendert das bestehende Chat-Standardverhalten also nicht."""
        if self.conversation._chat_mode != "code" and not self._context_efficiency_enabled:
            return message
        compressed, result = compress_tool_output_detailed(message, tool_name=tool_name)
        if result.strategy_notes:
            log.debug(
                "[CtxEff] Tool '%s' komprimiert: %d -> %d Zeichen (%s)",
                tool_name, result.original_chars, result.compressed_chars, result.strategy_notes,
            )
        return compressed

    def _should_continue_coding(self, response: LLMResponse) -> bool:
        """In code mode + ULTIMATE autonomy the loop keeps running until
        the assistant either calls a tool or explicitly finishes.

        A plain-text answer is considered unfinished unless it contains a
        strong done marker ("FERTIG", "erledigt", "abgeschlossen", ...).
        This forces the model to either act or clearly signal completion.
        """
        if self.conversation._chat_mode != "code" or self._active_workspace is None:
            return False
        command_tool = getattr(self, "command_tool", None)
        if command_tool is None or command_tool.autonomy_mode != AutonomyMode.ULTIMATE:
            return False
        if response.is_error or response.tool_calls:
            return False
        text = (response.text or "").lower()
        done_keywords = (
            "fertig", "erledigt", "abgeschlossen", "geschafft", "beendet",
            "done", "completed", "finished",
        )
        # Continue unless the assistant explicitly signals completion.
        return not any(k in text for k in done_keywords)

    def _process_response(
        self,
        response: LLMResponse,
        user_text: str = "",
        is_stale: Callable[[], bool] | None = None,
    ) -> str:
        """Process an LLM response, execute tool calls, and continue for code mode.

        In coding mode with an active workspace and ULTIMATE autonomy the loop
        keeps running until the assistant produces a final answer without a
        continuation signal. The total number of LLM responses processed is
        bounded by ``config.llm.max_coding_iterations``.
        """
        max_iterations = self.config.llm.max_coding_iterations
        if max_iterations <= 0:
            max_iterations = 10_000  # praktisch unbegrenzt, bricht über is_stale ab
        final_text = ""
        self._last_processed_response = response

        for _ in range(max_iterations):
            if is_stale is not None and is_stale():
                log.info("Coding-Schleife aufgrund Stale-Check abgebrochen")
                break

            self._last_processed_response = response

            if response.is_error:
                # API error placeholders are not persisted as real answers.
                final_text = guard_response(response.text, user_text)
                break

            if not response.tool_calls:
                text = response.text or (
                    "Entschuldigung, ich habe darauf keine Antwort erhalten. "
                    "Können Sie das anders formulieren?"
                )
                self.conversation.add_assistant_response(text)
                if self._should_continue_coding(response):
                    self.conversation.add_system_reminder(CONTINUE_REMINDER)
                    self.event_bus.publish(ProcessingStageEvent(stage="tool_followup", detail="Setze Coding-Schritt fort..."))
                    try:
                        response = self.provider.send(
                            messages=self.conversation.get_messages(),
                            system=self.conversation.system_prompt,
                            tools=self.tools.get_tool_schemas() if self.provider.supports_tools else None,
                            reasoning_effort=self.reasoning_effort,
                            use_high_tier=self.use_high_tier_model,
                        )
                    except Exception:
                        log.exception("LLM Continue-Fehler")
                        final_text = guard_response("Entschuldigung, Sir. Die Weiterführung ist fehlgeschlagen.", user_text)
                        break
                    self._maybe_publish_context_usage(response)
                    self._maybe_trigger_summarization()
                    if response.reasoning:
                        self.event_bus.publish(ProcessingStageEvent(stage="reasoning", reasoning=response.reasoning))
                    continue
                final_text = guard_response(text, user_text)
                break

            # Tool-call turn: record the assistant's tool_use, execute all
            # requested tools, then ask the provider for the next response.
            self.conversation.add_assistant_tool_use(response)
            for tc in response.tool_calls:
                self.event_bus.publish(ProcessingStageEvent(stage="tool_exec", detail=f"Führe Tool '{tc.name}' aus..."))
                result = self.tools.execute(tc.name, tc.input)
                compressed_message = self._maybe_compress_tool_result(tc.name, result.message)
                self.conversation.add_tool_result(tc.id, compressed_message, tool_name=tc.name)
                if tc.name == "describe_what_i_see" and result.success and result.message:
                    self.conversation.add_assistant_response(result.message)
                    self._last_processed_response = LLMResponse(text=result.message, tool_calls=[])
                    return guard_response(result.message, user_text)
                if not result.success:
                    log.warning("Tool %s meldet Fehler: %s", tc.name, result.error_code)

            try:
                self.event_bus.publish(ProcessingStageEvent(stage="tool_followup", detail="Warte auf Folge-Antwort..."))
                followup = self.provider.send_tool_result(
                    messages=self.conversation.get_messages(),
                    system=self.conversation.system_prompt,
                    tools=self.tools.get_tool_schemas() if self.provider.supports_tools else None,
                    reasoning_effort=self.reasoning_effort,
                    use_high_tier=self.use_high_tier_model,
                )
            except Exception:
                log.exception("LLM Followup-Fehler")
                final_text = guard_response(f"Aktion ausgeführt: {result.message}", user_text)
                break

            self._maybe_publish_context_usage(followup)
            self._maybe_trigger_summarization()
            if followup.reasoning:
                self.event_bus.publish(ProcessingStageEvent(stage="reasoning", reasoning=followup.reasoning))
            response = followup

        else:
            # max_coding_iterations reached without a final answer
            msg = "Entschuldigung, Sir. Zu viele Coding-Schritte erforderlich."
            self.conversation.add_assistant_response(msg)
            final_text = guard_response(msg, user_text)

        if not final_text:
            final_text = guard_response("Ich habe die Aktion ausgeführt.", user_text)

        return final_text

    def _shutdown(self) -> None:
        self._running = False
        self._extensions.stop()
        if self._proactive_engine is not None:
            self._proactive_engine.stop()
        if self._vision_observer is not None:
            self._vision_observer.close()
        ok, violations = self._audit_manager.verify_integrity()
        if not ok:
            log.warning("Audit-Integritätsverletzungen: %s", violations)
        self._audit_manager.rotate_old_logs(retention_days=365)
        if self.memory_store is not None:
            self.memory_store.prune_stale(
                episode_max_age_days=self.config.memory.episode_max_age_days,
                preference_min_confidence=self.config.memory.preference_min_confidence,
            )
        self.synthesizer.speak("Auf Wiedersehen, Sir. Es war mir ein Vergnügen.")
        log.info("TARNO beendet")

    def _run_briefing(self) -> None:
        """Generate and speak a proactive briefing."""
        try:
            prompt = self.config.briefing.prompt
            response = self.provider.send(
                messages=[{"role": "user", "content": prompt}],
                system=self.conversation.system_prompt,
                tools=self.tools.get_tool_schemas() if self.provider.supports_tools else None,
            )
            text = guard_response(response.text)
            self.event_bus.publish(ResponseReadyEvent(text=text))
            self.synthesizer.speak(text)
        except Exception:
            log.exception("Briefing fehlgeschlagen")

    def _register_default_tools(self) -> None:
        self.tools.register(build_coding_task_tool(self._coding_agent))

        self.tools.register(ToolDefinition(
            name="open_application",
            description="Öffnet eine Anwendung auf dem Computer (z.B. Notepad, Chrome, Calculator)",
            input_schema={
                "type": "object",
                "properties": {
                    "app_name": {"type": "string", "description": "Name der Anwendung"},
                },
                "required": ["app_name"],
            },
            handler=open_application,
        ))

        self.tools.register(ToolDefinition(
            name="open_file",
            description="Öffnet eine Datei mit der Standard-Anwendung",
            input_schema={
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Vollständiger Pfad zur Datei"},
                },
                "required": ["filepath"],
            },
            # KERNFIX (Code-Review): open_file() unterstuetzt Workspace-relative
            # Pfadaufloesung (wie alle anderen Datei-Tools hier: read_file,
            # write_file, list_directory, ...), wurde aber als nackte
            # Funktionsreferenz statt in ein workspace-injizierendes Lambda
            # registriert - relative Pfade im aktiven Workspace loesten sich
            # dadurch nie auf, anders als bei jedem anderen Datei-Tool.
            handler=lambda filepath: open_file(filepath, workspace=self._active_workspace),
        ))

        self.tools.register(ToolDefinition(
            name="web_search",
            description=(
                "Führt eine DuckDuckGo-Websuche durch und gibt die aktuellsten "
                "Suchergebnisse mit Titel, URL und Snippet zurück. Nutze dieses Tool "
                "immer, wenn aktuelle Echtzeit-Daten benötigt werden (z.B. aktuelle "
                "APIs, Bibliotheksversionen, Fehlermeldungen, News), anstatt auf "
                "veraltetes Trainingswissen zurückzugreifen."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Suchbegriff"},
                },
                "required": ["query"],
            },
            handler=web_search,
        ))

        self.tools.register(ToolDefinition(
            name="open_browser",
            description=(
                "Öffnet eine URL im Standard-Browser, sichtbar für den Nutzer. "
                "Nur verwenden, wenn der Nutzer die Seite selbst sehen/bedienen "
                "möchte. Für 'was steht auf der Seite'-Fragen stattdessen "
                "fetch_webpage nutzen - kein Browserfenster nötig."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "http:// oder https:// URL"},
                },
                "required": ["url"],
            },
            handler=open_browser,
        ))

        self.tools.register(ToolDefinition(
            name="fetch_webpage",
            description=(
                "Ruft eine URL serverseitig ab und gibt den bereinigten Lesetext "
                "zurück - kein sichtbares Browserfenster (genau wie ChatGPT/Gemini "
                "beim Ansehen einer Seite). Nutze dieses Tool, um Inhalte einer "
                "bekannten URL (z.B. aus einem web_search-Ergebnis) tatsächlich zu "
                "lesen, statt Inhalte zu erfinden oder zu raten. Bei dynamisch per "
                "JavaScript nachgeladenen Seiten liefert es evtl. keinen/wenig Text "
                "- das ehrlich so sagen, nicht kompensieren durch Erfinden."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "http:// oder https:// URL, z.B. aus einem web_search-Ergebnis"},
                },
                "required": ["url"],
            },
            handler=fetch_webpage,
        ))

        self.tools.register(ToolDefinition(
            name="browser_navigate",
            description=(
                "Öffnet eine URL in einem unsichtbaren, echten automatisierten "
                "Browser (nicht sichtbar für den Nutzer) und gibt den sichtbaren "
                "Seitentext zurück - im Gegensatz zu fetch_webpage wird hier "
                "echtes JavaScript ausgeführt, funktioniert also auch bei "
                "Seiten, die Inhalte dynamisch nachladen. Erster Schritt, bevor "
                "browser_type/browser_click auf dieser Seite genutzt werden "
                "können - jeder Aufruf navigiert dieselbe Browser-Sitzung neu."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "http:// oder https:// URL"},
                },
                "required": ["url"],
            },
            handler=browser_navigate,
        ))

        self.tools.register(ToolDefinition(
            name="browser_type",
            description=(
                "Tippt Text in ein Eingabefeld der aktuell in browser_navigate "
                "geöffneten Seite - das Feld wird anhand seines sichtbaren "
                "Labels oder Platzhaltertexts gefunden (z.B. 'Suche', "
                "'E-Mail-Adresse'), kein CSS-Selektor nötig."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "Sichtbares Label/Platzhalter des Eingabefelds"},
                    "value": {"type": "string", "description": "Einzugebender Text"},
                },
                "required": ["target", "value"],
            },
            handler=browser_type,
        ))

        self.tools.register(ToolDefinition(
            name="browser_click",
            description=(
                "Klickt einen Button oder Link auf der aktuell geöffneten Seite "
                "- anhand des sichtbaren Texts gefunden (z.B. 'Suchen', "
                "'Absenden', 'Anmelden'), kein CSS-Selektor nötig. Gibt den "
                "Seitentext nach dem Klick zurück."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "Sichtbarer Text des Buttons/Links"},
                },
                "required": ["target"],
            },
            handler=browser_click,
        ))

        self.tools.register(ToolDefinition(
            name="browser_read",
            description="Liest den aktuell sichtbaren Text der offenen automatisierten Browser-Seite erneut aus.",
            input_schema={"type": "object", "properties": {}},
            handler=browser_read,
        ))

        self.tools.register(ToolDefinition(
            name="browser_close",
            description="Schließt die automatisierte Browser-Sitzung (Ressourcen freigeben).",
            input_schema={"type": "object", "properties": {}},
            handler=browser_close,
        ))

        self.tools.register(ToolDefinition(
            name="get_system_info",
            description="Gibt Systeminformationen zurück (OS, CPU, RAM, Festplatte)",
            input_schema={"type": "object", "properties": {}},
            handler=lambda: get_system_info(),
        ))

        self.tools.register(ToolDefinition(
            name="take_screenshot",
            description="Erstellt einen Screenshot des Bildschirms",
            input_schema={"type": "object", "properties": {}},
            handler=lambda: take_screenshot(),
        ))

        self.tools.register(ToolDefinition(
            name="search_files",
            description=(
                "Sucht Dateien im aktiven Workspace nach Muster (z.B. '*.py', '**/*.cs'). "
                "Relative Pfade werden gegen den Workspace-Root aufgelöst."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "Verzeichnis zum Suchen (default: Workspace-Root)"},
                    "pattern": {"type": "string", "description": "Suchmuster (z.B. '*.py')"},
                },
                "required": ["pattern"],
            },
            # KERNFIX (Code-Review): 'directory' ist im Schema NICHT
            # required - ohne Default hier warf ein Tool-Aufruf ohne dieses
            # Feld (vom LLM legitim weggelassen) TypeError: missing 1
            # required positional argument, statt wie in der description
            # versprochen den Workspace-Root zu durchsuchen.
            handler=lambda pattern, directory=None: search_files(
                directory or ".", pattern, workspace=self._active_workspace
            ),
        ))

        self.tools.register(ToolDefinition(
            name="read_file",
            description=(
                "Liest den Inhalt einer Textdatei im aktiven Workspace. Relative Pfade "
                "werden gegen den Workspace-Root aufgelöst."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Pfad zur Datei (relativ zum Workspace-Root)"},
                },
                "required": ["filepath"],
            },
            handler=lambda filepath: read_file(filepath, workspace=self._active_workspace),
        ))

        self.tools.register(ToolDefinition(
            name="write_file",
            description=(
                "Schreibt eine Textdatei im aktiven Workspace. Relative Pfade "
                "werden gegen den Workspace-Root aufgelöst."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Zielpfad zur Datei (relativ zum Workspace-Root)"},
                    "content": {"type": "string", "description": "Inhalt der Datei"},
                },
                "required": ["filepath", "content"],
            },
            handler=lambda filepath, content: write_file(filepath, content, workspace=self._active_workspace),
        ))

        self.tools.register(ToolDefinition(
            name="list_directory",
            description=(
                "Listet Dateien und Verzeichnisse im aktiven Workspace auf. "
                "Relative Pfade werden gegen den Workspace-Root aufgelöst; leerer Pfad = Workspace-Root."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Pfad zum Verzeichnis (default: Workspace-Root)"},
                },
                "required": [],
            },
            # KERNFIX (Code-Review): 'path' ist im Schema NICHT required
            # ("leerer Pfad = Workspace-Root", siehe description) - ohne
            # Default hier warf ein Tool-Aufruf ohne dieses Feld TypeError,
            # statt den Workspace-Root aufzulisten.
            handler=lambda path=None: list_directory(path or "", workspace=self._active_workspace),
        ))

        self.tools.register(ToolDefinition(
            name="create_directory",
            description="Erstellt ein neues Verzeichnis im aktiven Workspace.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Pfad des zu erstellenden Verzeichnisses (relativ zum Workspace-Root)"},
                },
                "required": ["path"],
            },
            handler=lambda path: create_directory(path, workspace=self._active_workspace),
        ))

        self.tools.register(ToolDefinition(
            name="copy_file",
            description="Kopiert eine Datei im aktiven Workspace.",
            input_schema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Quellpfad (relativ zum Workspace-Root)"},
                    "destination": {"type": "string", "description": "Zielpfad (relativ zum Workspace-Root)"},
                },
                "required": ["source", "destination"],
            },
            handler=lambda source, destination: copy_file(source, destination, workspace=self._active_workspace),
        ))

        self.tools.register(ToolDefinition(
            name="move_file",
            description="Verschiebt eine Datei im aktiven Workspace.",
            input_schema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Quellpfad (relativ zum Workspace-Root)"},
                    "destination": {"type": "string", "description": "Zielpfad (relativ zum Workspace-Root)"},
                },
                "required": ["source", "destination"],
            },
            handler=lambda source, destination: move_file(source, destination, workspace=self._active_workspace),
        ))

        self.tools.register(ToolDefinition(
            name="delete_file",
            description="Löscht eine Datei im aktiven Workspace (Vorsicht — unwiderruflich).",
            input_schema={
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Pfad der zu löschenden Datei (relativ zum Workspace-Root)"},
                },
                "required": ["filepath"],
            },
            handler=lambda filepath: delete_file(filepath, workspace=self._active_workspace),
        ))

        self.tools.register(ToolDefinition(
            name="focus_application",
            description="Bringt ein Fenster anhand eines Titel-Suchbegriffs in den Vordergrund",
            input_schema={
                "type": "object",
                "properties": {
                    "title_substring": {"type": "string", "description": "Teil des Fenstertitels, z.B. 'Chrome'"},
                },
                "required": ["title_substring"],
            },
            handler=focus_window,
        ))

        self.tools.register(ToolDefinition(
            name="close_window",
            description="Schließt ein Fenster anhand eines Titel-Suchbegriffs (sauber, per WM_CLOSE)",
            input_schema={
                "type": "object",
                "properties": {
                    "title_substring": {"type": "string", "description": "Teil des Fenstertitels, z.B. 'Editor'"},
                },
                "required": ["title_substring"],
            },
            handler=close_window,
        ))

        self.tools.register(ToolDefinition(
            name="add_reminder",
            description=(
                "Legt eine ZEITBASIERTE Erinnerung an, die TARNO zur angegebenen "
                "Uhrzeit automatisch per Sprachausgabe UND Chat-Nachricht meldet "
                "(geprüft jede Minute). Nutze IMMER dieses Tool, wenn der Nutzer "
                "bittet, an einen Termin/eine Aufgabe zu einer bestimmten Zeit "
                "erinnert zu werden - 'remember_fact' speichert nur einen "
                "statischen Fakt und löst NIEMALS von selbst eine Erinnerung aus."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Was TARNO sagen soll, wenn die Erinnerung fällig ist"},
                    "time": {"type": "string", "description": "Uhrzeit im 24h-Format HH:MM, z.B. '19:20'"},
                    "date": {"type": "string", "description": "Optional: Datum als YYYY-MM-DD. Weggelassen = heute, oder morgen falls die Uhrzeit heute bereits vorbei ist."},
                },
                "required": ["message", "time"],
            },
            handler=self._add_reminder_tool,
        ))

        self.tools.register(ToolDefinition(
            name="remember_fact",
            description=(
                "Speichert einen wichtigen Fakt über den Nutzer dauerhaft im "
                "Langzeitgedächtnis (z.B. Präferenzen, wichtige persönliche Details). "
                "Löst KEINE zeitbasierte Erinnerung aus - dafür 'add_reminder' nutzen."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Kurzer Schlüssel, z.B. 'lieblingsfarbe'"},
                    "value": {"type": "string", "description": "Der zu speichernde Wert"},
                },
                "required": ["key", "value"],
            },
            handler=self._remember_fact_tool,
        ))

        self.tools.register(ToolDefinition(
            name="describe_what_i_see",
            description=(
                "Erfasst EIN aktuelles Bild von der Webcam und beschreibt in "
                "Worten, was gerade zu sehen ist. Nutze dieses Tool, wenn der "
                "Nutzer direkt fragt, was TARNO gerade sieht/wahrnimmt "
                "('was siehst du?', 'schau mal', etc.) - NICHT für "
                "Bildschirminhalte (dafür 'take_screenshot')."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=self._describe_what_i_see_tool,
        ))

        command_tool = CommandTool(self.config, permission_service=self._permission_service)
        self.command_tool = command_tool
        self.tools.register(ToolDefinition(
            name="execute_command",
            description=(
                "Führt einen Shell-Befehl im aktiven Workspace aus. Relativer "
                "Bezug ist der Workspace-Root; z.B. 'dotnet build' oder 'pytest'."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "intent": {"type": "string", "description": "Was der Nutzer will, z.B. run_tests"},
                    "shell": {"type": "string", "enum": ["powershell", "cmd", "bash"], "default": "powershell"},
                    "command": {"type": "string", "description": "Der auszuführende Befehl"},
                    "params": {"type": "object", "description": "Zusätzliche Parameter"},
                    "explanation": {"type": "string", "description": "Was der Befehl bewirkt"},
                    "user_query": {"type": "string", "description": "Originale Benutzeranfrage"},
                },
                "required": ["intent", "command"],
            },
            handler=lambda **kwargs: self.command_tool(workspace=self._active_workspace, **kwargs),
        ))

        log.info("%d Tools registriert", len(self.tools.get_tool_schemas()))

    def _load_plugins(self) -> None:
        if not self.config.plugins.enabled or not self.config.plugins.auto_load:
            return

        plugin_manager = PluginManager(self.tools, context={"config": self.config, "engine": self})
        builtin_dir = Path(__file__).resolve().parent.parent / "integrations"
        plugin_manager.load_from_directory(builtin_dir)

        for directory in self.config.plugins.directories:
            plugin_manager.load_from_directory(directory)

        log.info(
            "Plugins geladen: %s",
            ", ".join(plugin_manager.list_plugins()) or "keine",
        )
