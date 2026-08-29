"""Coordinator that bundles all autonomous TARNO extensions."""

from __future__ import annotations

import logging
from typing import Any, Callable

from tarno_backend.core.action_result import ActionResult
from tarno_backend.extensions.reminder import ReminderEngine
from tarno_backend.extensions.routines import RoutineRunner
from tarno_backend.extensions.scheduler import BriefingScheduler, TarnoScheduler, ScheduledTask
from tarno_backend.extensions.task_planner import PlannedTask, TaskPlanner

log = logging.getLogger(__name__)


class ExtensionCoordinator:
    """Owns the lifecycle of scheduler, reminders, routines and task planning.

    This is the single integration point used by Engine and ServiceMediator.
    """

    def __init__(
        self,
        tool_executor: Callable[[str, dict[str, Any]], Any],
        briefing_callback: Callable[[], None] | None = None,
        reminder_callback: Callable[[str], None] | None = None,
        approval_callback: Callable[[PlannedTask], bool] | None = None,
    ) -> None:
        self._tool_executor = tool_executor
        self._scheduler = TarnoScheduler(tick_seconds=30)
        self._briefing_scheduler: BriefingScheduler | None = None
        if briefing_callback:
            self._briefing_scheduler = BriefingScheduler([], briefing_callback)
            self._briefing_scheduler.start()

        self._reminder_engine = ReminderEngine(notify_callback=reminder_callback)
        self._routine_runner = RoutineRunner(tool_executor)
        self._task_planner = TaskPlanner(
            approval_callback=approval_callback or (lambda task: False),
            executor=tool_executor,
        )

    @property
    def reminder_engine(self) -> ReminderEngine:
        return self._reminder_engine

    @property
    def routine_runner(self) -> RoutineRunner:
        return self._routine_runner

    @property
    def task_planner(self) -> TaskPlanner:
        return self._task_planner

    def schedule_reminder_check(self) -> None:
        """Add a periodic reminder check to the scheduler."""
        self._scheduler.schedule(ScheduledTask(
            id="reminder-check",
            cron="*/1",
            action=self._reminder_engine.check_and_notify,
            metadata={"type": "reminder"},
        ))

    def set_briefing_times(self, times: list[str], callback: Callable[[], None]) -> None:
        """Configure daily briefing times."""
        if self._briefing_scheduler:
            self._briefing_scheduler.stop()
        self._briefing_scheduler = BriefingScheduler(times, callback)
        self._briefing_scheduler.start()

    def start(self) -> None:
        """Start background schedulers."""
        self.schedule_reminder_check()
        self._scheduler.start()
        if self._briefing_scheduler:
            self._briefing_scheduler.start()
        log.info("Extension Coordinator gestartet")

    def stop(self) -> None:
        """Stop background schedulers."""
        self._scheduler.stop()
        if self._briefing_scheduler:
            self._briefing_scheduler.stop()
        log.info("Extension Coordinator gestoppt")

    def try_routine(self, user_text: str) -> ActionResult | None:
        """If the user text matches a routine, run it and return a result."""
        routine = self._routine_runner.match(user_text)
        if routine is None:
            return None
        results = self._routine_runner.run(routine)
        return ActionResult(
            success=True,
            message=f"Routine '{routine.name}' ausgeführt.",
            details={"results": results},
        )

    def create_task(self, description: str, steps: list[dict[str, Any]]) -> PlannedTask:
        return self._task_planner.create(description, steps)

    def run_task(self, task_id: str) -> PlannedTask:
        return self._task_planner.run(task_id)
