"""Proactive briefing scheduler — generates a spoken summary at configured times."""

from __future__ import annotations

import datetime
import logging
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tarno_backend.core.config import BriefingConfig
    from tarno_backend.core.agent_service import AgentService

log = logging.getLogger(__name__)


def _next_timepoint(now: datetime.datetime, times: list[str]) -> datetime.datetime | None:
    """Find the next scheduled briefing time on or after `now`.

    `times` are strings in "HH:MM" 24-hour format.
    """
    if not times:
        return None

    candidates: list[datetime.datetime] = []
    today = now.date()
    for t in times:
        try:
            hour, minute = map(int, t.split(":", 1))
            candidate = datetime.datetime.combine(today, datetime.time(hour, minute))
            if candidate <= now:
                candidate += datetime.timedelta(days=1)
            candidates.append(candidate)
        except Exception:
            log.warning("Ungueltige Briefing-Zeit: %s", t)
            continue

    if not candidates:
        return None
    return min(candidates)


class ProactiveBriefing:
    """Schedules and triggers spoken TARNO briefings."""

    def __init__(
        self,
        config: BriefingConfig,
        agent_service: AgentService,
    ) -> None:
        self._config = config
        self._agent_service = agent_service
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._running = False

    def start(self) -> None:
        """Start the background scheduler."""
        if not self._config.enabled:
            log.info("Proaktive Briefings deaktiviert")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        log.info("Proaktive Briefings gestartet (Zeiten: %s)", self._config.times)

    def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        log.info("Proaktive Briefings gestoppt")

    def trigger_now(self) -> None:
        """Generate a briefing immediately."""
        self._agent_service.trigger_briefing()

    def _run(self) -> None:
        """Wait for the next briefing time and trigger it."""
        if self._config.trigger_on_start:
            try:
                self._agent_service.trigger_briefing()
            except Exception:
                log.exception("Start-Briefing fehlgeschlagen")

        while self._running and not self._stop_event.is_set():
            now = datetime.datetime.now()
            next_point = _next_timepoint(now, self._config.times)
            if next_point is None:
                log.warning("Keine gueltigen Briefing-Zeiten konfiguriert")
                break

            sleep_seconds = (next_point - now).total_seconds()
            log.debug("Naechstes Briefing um %s (in %d s)", next_point, int(sleep_seconds))
            if self._stop_event.wait(sleep_seconds):
                break

            try:
                self._agent_service.trigger_briefing()
            except Exception:
                log.exception("Geplantes Briefing fehlgeschlagen")

            # Avoid retriggering immediately; sleep until past the event minute.
            time.sleep(61)
