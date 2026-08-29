"""Generic background scheduler for briefings, reminders and routines."""

from __future__ import annotations

import datetime
import logging
import threading
from dataclasses import dataclass, field
from typing import Callable

log = logging.getLogger(__name__)


@dataclass
class ScheduledTask:
    id: str
    cron: str
    action: Callable[[], None]
    enabled: bool = True
    last_run: datetime.datetime | None = None
    metadata: dict = field(default_factory=dict)


class TarnoScheduler:
    """Lightweight cron-like scheduler for autonomous TARNO tasks.

    Supports two formats:
        - "HH:MM" daily tasks
        - "*/N" every N minutes tasks
    """

    def __init__(self, tick_seconds: int = 30) -> None:
        self._tasks: dict[str, ScheduledTask] = {}
        self._tick = tick_seconds
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._running = False

    def schedule(self, task: ScheduledTask) -> None:
        self._tasks[task.id] = task
        log.info("Aufgabe geplant: %s (%s)", task.id, task.cron)

    def unschedule(self, task_id: str) -> bool:
        if task_id in self._tasks:
            del self._tasks[task_id]
            log.info("Aufgabe entfernt: %s", task_id)
            return True
        return False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        log.info("Scheduler gestartet")

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        log.info("Scheduler gestoppt")

    def _run(self) -> None:
        while self._running and not self._stop_event.is_set():
            now = datetime.datetime.now()
            for task in list(self._tasks.values()):
                if not task.enabled:
                    continue
                if self._should_run(task, now):
                    task.last_run = now
                    self._execute(task)
            if self._stop_event.wait(self._tick):
                break

    def _should_run(self, task: ScheduledTask, now: datetime.datetime) -> bool:
        if task.last_run and (now - task.last_run).total_seconds() < 60:
            return False

        cron = task.cron.strip()
        if cron.startswith("*/"):
            try:
                minutes = int(cron[2:])
                return now.minute % minutes == 0
            except ValueError:
                log.warning("Ungültiger Intervall-Cron: %s", cron)
                return False

        try:
            hour, minute = map(int, cron.split(":", 1))
            return now.hour == hour and now.minute == minute
        except Exception:
            log.warning("Ungültiger Cron: %s", cron)
            return False

    def _execute(self, task: ScheduledTask) -> None:
        log.info("Führe geplante Aufgabe aus: %s", task.id)
        try:
            task.action()
            self.audit(task.id, "executed", task.metadata)
        except Exception:
            log.exception("Geplante Aufgabe fehlgeschlagen: %s", task.id)

    @staticmethod
    def audit(task_id: str, action: str, metadata: dict) -> None:
        log.info("[audit] task=%s action=%s metadata=%s", task_id, action, metadata)


class BriefingScheduler(TarnoScheduler):
    """Convenience scheduler that wires daily briefing times to a callback."""

    def __init__(self, times: list[str], callback: Callable[[], None]) -> None:
        super().__init__(tick_seconds=30)
        for idx, t in enumerate(times):
            self.schedule(ScheduledTask(
                id=f"briefing-{idx}",
                cron=t,
                action=callback,
                metadata={"type": "briefing"},
            ))
