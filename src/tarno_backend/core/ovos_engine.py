from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any

# PyInstaller extraction root; for normal source runs this is the project root.
# Deliberately NOT using tarno_backend.utils.paths.find_repo_root() here: a
# PyInstaller bundle has no .git/ or src/ marker to find (different, flattened
# layout), so this has to stay a fixed hop-count from this file's own bundled
# location rather than a marker search.
_BUNDLE_ROOT = Path(__file__).parent.parent.parent.parent

# Runtime home is always a persistent user directory so that config, cache,
# data and the persona file never leak into the project/source tree.
_RUNTIME_ROOT = Path(os.environ.get("TARNO_HOME", Path.home() / ".tarno"))
_BUNDLE_CONFIG_ROOT = Path(getattr(sys, "_MEIPASS", _BUNDLE_ROOT)) / "config"
_PROJECT_ROOT = _RUNTIME_ROOT


def _default_timezone() -> dict[str, Any]:
    """Return a timezone dict for the current system, falling back to Berlin."""
    try:
        from datetime import datetime

        from tzlocal import get_localzone

        tz = get_localzone()
        now = datetime.now(tz)
        offset = now.utcoffset()
        offset_hours = round(offset.total_seconds() / 3600) if offset else 1
        name = str(tz) if tz else "Europe/Berlin"
        return {"code": name, "name": name, "offset": offset_hours}
    except Exception:
        return {"code": "Europe/Berlin", "name": "Europe/Berlin", "offset": 1}


def _seed_user_config() -> None:
    """Copy the bundled OVOS config tree into the persistent user config directory.

    OVOS expects the user config file at ``XDG_CONFIG_HOME/mycroft/mycroft.conf``.
    The bundled ``config/`` tree is copied as-is, then the main ``mycroft.conf`` is
    validated/updated to contain the message-bus config, a persona path that points
    inside the runtime home, and a valid location/timezone section.
    """
    import json
    import logging

    log = logging.getLogger(__name__)
    log.info("Seeding OVOS config from %s to %s", _BUNDLE_CONFIG_ROOT, _PROJECT_ROOT / "config")

    if not _BUNDLE_CONFIG_ROOT.exists():
        log.warning("Bundle config root does not exist: %s", _BUNDLE_CONFIG_ROOT)
        return
    user_config = _PROJECT_ROOT / "config"
    user_config.mkdir(parents=True, exist_ok=True)
    marker = user_config / ".seeded.v2"

    if not marker.exists():
        for item in _BUNDLE_CONFIG_ROOT.rglob("*"):
            if item.is_file():
                rel = item.relative_to(_BUNDLE_CONFIG_ROOT)
                dest = user_config / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dest)
        marker.touch()
        log.info("Copied bundled config tree to %s", user_config)

    # The OVOS config loader looks for XDG_CONFIG_HOME/mycroft/mycroft.conf.
    # Make sure that exact file exists and contains the message-bus config.
    mycroft_dir = user_config / "mycroft"
    mycroft_dir.mkdir(parents=True, exist_ok=True)
    target_conf = mycroft_dir / "mycroft.conf"
    source_conf = _BUNDLE_CONFIG_ROOT / "mycroft" / "mycroft.conf"

    def _load_mycroft_config() -> dict[str, Any]:
        """Load target_conf, recovering from corruption via the bundled source."""
        if target_conf.exists():
            try:
                with open(target_conf, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError as exc:
                log.warning("Corrupted mycroft.conf at %s: %s", target_conf, exc)
                backup = target_conf.with_suffix(".conf.backup")
                try:
                    shutil.move(str(target_conf), str(backup))
                    log.warning("Backed up corrupted mycroft.conf to %s", backup)
                except Exception as move_exc:
                    log.warning("Could not backup corrupted mycroft.conf: %s", move_exc)

        if source_conf.exists():
            try:
                with open(source_conf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                log.info("Loaded bundled source config from %s", source_conf)
                shutil.copy2(source_conf, target_conf)
                log.info("Restored mycroft.conf from bundle at %s", target_conf)
                return data
            except json.JSONDecodeError as exc:
                log.warning("Bundled source config is also corrupted: %s", exc)

        log.warning("Creating minimal mycroft.conf at %s", target_conf)
        target_conf.write_text("{}", encoding="utf-8")
        return {}

    try:
        data = _load_mycroft_config()

        # Replace null values that would break the message bus loader.
        if not isinstance(data.get("websocket"), dict):
            data["websocket"] = {}
        if not isinstance(data.get("ssl"), bool):
            data["ssl"] = False

        data["websocket"].setdefault("host", "127.0.0.1")
        data["websocket"].setdefault("port", 8181)
        data["websocket"].setdefault("route", "/core")

        # Make persona path point to the same XDG_CONFIG_HOME tree we seeded.
        if not isinstance(data.get("intents"), dict):
            data["intents"] = {}
        data["intents"]["personas_path"] = str(user_config / "ovos_persona")

        # Ensure a full location section exists; OVOS locale helpers crash on missing keys.
        if not isinstance(data.get("location"), dict):
            data["location"] = {}
        location = data["location"]
        if not isinstance(location.get("city"), dict):
            location["city"] = {
                "code": "Berlin",
                "name": "Berlin",
                "state": {
                    "code": "BE",
                    "name": "Berlin",
                    "country": {"code": "DE", "name": "Germany"},
                },
            }
        if not isinstance(location.get("coordinate"), dict):
            location["coordinate"] = {"latitude": 52.52, "longitude": 13.405}

        tz = _default_timezone()
        if not isinstance(location.get("timezone"), dict):
            location["timezone"] = tz
        else:
            location["timezone"].setdefault("code", tz["code"])
            location["timezone"].setdefault("name", tz["name"])
            location["timezone"].setdefault("offset", tz["offset"])

        with open(target_conf, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        log.info("Seeded mycroft.conf at %s", target_conf)
    except Exception:
        log.exception("Konnte mycroft.conf nicht seeden")


# Set XDG paths BEFORE importing any ovos/json_database modules so their
# XDG defaults evaluate to the intended directories. Use direct assignment because
# the main entry point may have set defaults that point elsewhere.
os.environ["XDG_CONFIG_HOME"] = str(_PROJECT_ROOT / "config")
os.environ["XDG_CACHE_HOME"] = str(_PROJECT_ROOT / ".cache")
os.environ["XDG_DATA_HOME"] = str(_PROJECT_ROOT / ".data")
os.makedirs(Path(os.environ["XDG_CACHE_HOME"]) / "json_database", exist_ok=True)
os.makedirs(Path(os.environ["XDG_DATA_HOME"]) / "json_database", exist_ok=True)

# Debug: log the resolved environment before OVOS modules are imported.
import logging as _logging

_log = _logging.getLogger(__name__)
_log.info("OVOS runtime paths: XDG_CONFIG_HOME=%s XDG_CACHE_HOME=%s XDG_DATA_HOME=%s", os.environ["XDG_CONFIG_HOME"], os.environ["XDG_CACHE_HOME"], os.environ["XDG_DATA_HOME"])
_log.info("Project root: %s, bundle config root: %s, frozen: %s", _PROJECT_ROOT, _BUNDLE_CONFIG_ROOT, getattr(sys, "frozen", False))

from ovos_bus_client import MessageBusClient
from ovos_config.config import Configuration
from ovos_core.skill_manager import SkillManager
from ovos_messagebus import __main__ as messagebus_service
from ovos_utils.log import LOG, init_service_logger

from tarno_backend.core.agent_service import AgentService
from tarno_backend.core.config import TarnoConfig
from tarno_backend.core.proactive_briefing import ProactiveBriefing
from tarno_backend.voice.voice_service import VoiceService


def _no_op(*args, **kwargs) -> None:
    """No-op replacement for signal handlers."""
    pass


def _wait_for_event(event: threading.Event) -> None:
    """Wait until the event is set, used as replacement for wait_for_exit_signal."""
    try:
        event.wait()
    except Exception:
        pass


class TarnoOvosEngine:
    """TARNO engine that uses OpenVoiceOS as the brain and the message bus."""

    def __init__(self, project_root: str | None = None) -> None:
        self._project_root = Path(project_root or str(_PROJECT_ROOT))
        self._config_dir = self._project_root / "config"
        self._setup_xdg_config()
        self._bus_thread: threading.Thread | None = None
        self._bus_stop_event: threading.Event | None = None
        self._bus_client: MessageBusClient | None = None
        self._skill_manager: SkillManager | None = None
        self._agent_service: AgentService | None = None
        self._voice_service: VoiceService | None = None
        self._briefing_service: ProactiveBriefing | None = None
        self._running = False

    def _setup_xdg_config(self) -> None:
        """Point XDG config/cache/data to the runtime directory.

        The runtime directory is always ~/.tarno (or TARNO_HOME) so that
        config, cache, and data survive between runs and never leak into the
        project/source tree.
        """
        os.environ["XDG_CONFIG_HOME"] = str(self._config_dir)
        os.environ["XDG_CACHE_HOME"] = str(self._project_root / ".cache")
        os.environ["XDG_DATA_HOME"] = str(self._project_root / ".data")
        os.makedirs(self._config_dir / "ovos_persona", exist_ok=True)
        os.makedirs(self._project_root / ".cache" / "json_database", exist_ok=True)
        os.makedirs(self._project_root / ".data", exist_ok=True)

        # Seed the bundled config tree into the persistent runtime home and
        # validate mycroft.conf. This is safe to call repeatedly.
        _seed_user_config()

        # Ensure the TARNO persona is available in the XDG config directory
        persona_src = self._config_dir / "persona" / "tarno.json"
        persona_dst = self._config_dir / "ovos_persona" / "tarno.json"
        if persona_src.exists():
            self._install_persona(persona_src, persona_dst)

    def _install_persona(self, src: Path, dst: Path) -> None:
        """Copy the TARNO persona to the runtime config directory.

        The Mistral API key is intentionally NOT persisted here. It must be
        provided via the MISTRAL_API_KEY environment variable at runtime.
        """
        import json
        import shutil
        shutil.copy(src, dst)
        with open(dst, "r", encoding="utf-8") as f:
            persona = json.load(f)
        persona.setdefault("ovos-solver-openai-plugin", {})
        # Remove any key field that may have been present in the bundled
        # source so the runtime secret is never written to disk.
        persona["ovos-solver-openai-plugin"].pop("key", None)
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(persona, f, indent=2)

    def _start_message_bus(self) -> None:
        """Run the ovos-messagebus server in a daemon thread.

        A subprocess is not used because PyInstaller executables cannot run
        `-m` style sub-processes. Signal handlers are patched out because they
        can only run in the main interpreter thread on Windows.
        """
        LOG.info("Starting OpenVoiceOS message bus server")
        self._bus_stop_event = threading.Event()

        # Debug: dump the configuration that the message bus loader will use.
        try:
            cfg = Configuration()
            ws = cfg.get("websocket")
            loc = cfg.get("location")
            LOG.info("OVOS Configuration websocket section: %s (type=%s)", ws, type(ws).__name__)
            LOG.info("OVOS Configuration location section: %s (type=%s)", loc, type(loc).__name__)
            LOG.info("OVOS Configuration ssl: %s", cfg.get("ssl"))
        except Exception:
            LOG.exception("Could not dump OVOS configuration")

        original_reset_sigint = messagebus_service.reset_sigint_handler
        original_wait = messagebus_service.wait_for_exit_signal
        messagebus_service.reset_sigint_handler = _no_op
        messagebus_service.wait_for_exit_signal = lambda: _wait_for_event(self._bus_stop_event)

        def _run_bus() -> None:
            try:
                messagebus_service.main()
            except Exception as e:
                LOG.error("Message bus stopped: %s", e)
            finally:
                messagebus_service.reset_sigint_handler = original_reset_sigint
                messagebus_service.wait_for_exit_signal = original_wait

        self._bus_thread = threading.Thread(target=_run_bus, daemon=True)
        self._bus_thread.start()

    def _start_skill_manager(self) -> None:
        """Start ovos-core skill manager."""
        LOG.info("Starting OpenVoiceOS skill manager")
        self._bus_client = MessageBusClient()
        self._bus_client.run_in_thread()
        self._bus_client.connected_event.wait()

        self._skill_manager = SkillManager(
            self._bus_client,
            enable_file_watcher=True,
            enable_skill_api=True,
            enable_intent_service=True,
            enable_installer=False,
            enable_event_scheduler=True,
        )
        self._skill_manager.start()

    def _start_agent_service(self) -> None:
        """Start the TARNO agent service with LLM and tools."""
        LOG.info("Starting TARNO agent service")
        tarno_config = TarnoConfig.load()
        self._agent_service = AgentService(self._bus_client, tarno_config)
        self._agent_service.start()

    def _start_voice_service(self) -> None:
        """Start the microphone pipeline and wire it to the OVOS bus."""
        LOG.info("Starting TARNO voice service")
        tarno_config = TarnoConfig.load()
        self._voice_service = VoiceService(tarno_config, self._bus_client)
        self._voice_service.start()

    def _wait_for_message_bus(self, timeout: float = 10.0) -> bool:
        """Wait until the message bus server is accepting connections."""
        import socket
        ws_config = Configuration().get("websocket") or {}
        host = ws_config.get("host", "127.0.0.1")
        port = ws_config.get("port", 8181)
        LOG.info("Waiting for message bus on %s:%s", host, port)
        start = time.time()
        while time.time() - start < timeout:
            try:
                with socket.create_connection((host, port), timeout=1.0):
                    return True
            except (OSError, ConnectionRefusedError):
                time.sleep(0.2)
        return False

    def start(self) -> None:
        """Start the entire OVOS-based TARNO engine."""
        init_service_logger("tarno_backend")
        self._running = True
        self._start_message_bus()
        if not self._wait_for_message_bus():
            LOG.error("Message bus server failed to start")
            self.stop()
            raise RuntimeError("Message bus server failed to start")
        self._start_skill_manager()
        self._start_agent_service()
        self._start_voice_service()
        self._start_briefing_service()
        LOG.info("TARNO OVOS engine is ready")

    def _start_briefing_service(self) -> None:
        """Start the proactive briefing scheduler if enabled."""
        tarno_config = TarnoConfig.load()
        if not tarno_config.briefing.enabled:
            return
        self._briefing_service = ProactiveBriefing(
            tarno_config.briefing,
            self._agent_service,
        )
        self._briefing_service.start()

    def stop(self) -> None:
        """Shut down the engine."""
        self._running = False
        if self._briefing_service:
            self._briefing_service.stop()
        if self._voice_service:
            self._voice_service.stop()
        if self._agent_service:
            self._agent_service.stop()
        if self._skill_manager:
            self._skill_manager.shutdown()
        if self._bus_client:
            self._bus_client.close()
        if self._bus_stop_event:
            try:
                from tornado import ioloop
                ioloop.IOLoop.instance().add_callback(ioloop.IOLoop.instance().stop)
            except Exception:
                pass
            self._bus_stop_event.set()
        if self._bus_thread:
            self._bus_thread.join(timeout=2.0)
        LOG.info("TARNO OVOS engine shut down")

    def run(self) -> None:
        """Start the engine and block until interrupted."""
        self.start()
        from ovos_utils import wait_for_exit_signal
        wait_for_exit_signal()
        self.stop()


def main() -> None:
    """CLI entry point for the OVOS-based TARNO engine."""
    engine = TarnoOvosEngine()
    engine.run()


if __name__ == "__main__":
    main()
