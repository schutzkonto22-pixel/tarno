"""Shell execution backend for the command engine."""

from __future__ import annotations

import asyncio
import logging
import shlex
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

from tarno_backend.core.action_result import ActionResult
from tarno_backend.core.command_engine import CommandEngine, PreparedCommand
from tarno_backend.core.process_sandbox import ProcessSandbox
from tarno_backend.security.content_filter import is_output_safe

log = logging.getLogger(__name__)

# Maximum characters captured from stdout/stderr to avoid huge LLM context.
_OUTPUT_LIMIT = 4000

# Process-wide execution rate limit - Python-Pendant zu ExecutionGuard.kt.
# Modul-Ebene (nicht Instanz), damit das Limit unabhängig davon greift, ob
# ShellExecutor pro Aufruf frisch instanziiert wird. Schützt gegen schnelle
# Spawn-/Wiederhol-Schleifen (der Overload-Vorfall spawnte ~1 Prozess/Sekunde
# über Minuten).
_MAX_EXECUTIONS_PER_WINDOW = 20
_RATE_WINDOW_SECONDS = 10.0
_execution_timestamps: deque[float] = deque()


def _rate_limit_exceeded() -> bool:
    """Return True (and do NOT record) if the execution rate cap is hit;
    otherwise record the current call and return False."""
    now = time.monotonic()
    while _execution_timestamps and now - _execution_timestamps[0] > _RATE_WINDOW_SECONDS:
        _execution_timestamps.popleft()
    if len(_execution_timestamps) >= _MAX_EXECUTIONS_PER_WINDOW:
        return True
    _execution_timestamps.append(now)
    return False


class ShellExecutor:
    """Executes prepared shell commands asynchronously and captures output."""

    def __init__(self, default_timeout: float = 60.0) -> None:
        self._default_timeout = default_timeout

    async def execute(
        self,
        prepared: PreparedCommand,
        dry_run: bool = False,
        cwd: str | Path | None = None,
    ) -> ActionResult:
        """Run a prepared command and return an ActionResult."""
        if dry_run:
            return ActionResult(
                success=True,
                message=f"[DRY-RUN] {prepared.shell}: {prepared.command}",
                details={"command": prepared.command, "shell": prepared.shell},
            )

        # Letzte Verteidigungslinie (Defense-in-Depth): normalerweise hat der
        # Befehl bereits CommandEngine.prepare() durchlaufen, aber falls ein
        # PreparedCommand direkt konstruiert wird, darf ein Fork-Bomben-/
        # Endlosschleifen-Muster hier trotzdem nicht in create_subprocess_exec
        # gelangen.
        content_safe, content_reason = is_output_safe(prepared.command)
        if not content_safe:
            log.warning(
                "Befehl durch Content-Filter blockiert (Executor): %s (%s)",
                prepared.command, content_reason,
            )
            return ActionResult(
                success=False,
                message=f"Befehl blockiert (Sicherheitsrichtlinie): {content_reason}",
                error_code="SecurityPolicyViolation",
                details={"command": prepared.command, "shell": prepared.shell},
            )

        if _rate_limit_exceeded():
            log.warning(
                "Ausführungs-Rate-Limit erreicht (>%d in %.0fs) - Befehl blockiert: %s",
                _MAX_EXECUTIONS_PER_WINDOW, _RATE_WINDOW_SECONDS, prepared.command,
            )
            return ActionResult(
                success=False,
                message=(
                    f"Zu viele Befehlsausführungen in kurzer Zeit "
                    f"(>{_MAX_EXECUTIONS_PER_WINDOW} in {_RATE_WINDOW_SECONDS:.0f}s) - blockiert "
                    f"(Schutz gegen Spawn-Schleifen)."
                ),
                error_code="RateLimitExceeded",
                details={"command": prepared.command, "shell": prepared.shell},
            )

        command_list = self._build_command(prepared.shell, prepared.command)
        timeout = prepared.timeout if prepared.timeout is not None else self._default_timeout

        sandbox = ProcessSandbox()
        try:
            try:
                create_sub = asyncio.create_subprocess_exec(
                    *command_list,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=str(cwd) if cwd else None,
                )
                if timeout and timeout > 0:
                    proc = await asyncio.wait_for(create_sub, timeout=timeout)
                else:
                    proc = await create_sub
                # Job-Object-Sandbox (Phase 41): sofort nach dem Spawnen
                # zuweisen, bevor der Prozess eigene Kindprozesse erzeugen kann.
                sandbox.create_and_assign(proc.pid)
                communicate = proc.communicate()
                if timeout and timeout > 0:
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(communicate, timeout=timeout)
                else:
                    stdout_bytes, stderr_bytes = await communicate
            except asyncio.TimeoutError:
                log.warning("Timeout beim Ausführen: %s", prepared.command)
                return ActionResult(
                    success=False,
                    message=f"Timeout nach {timeout}s: {prepared.command}",
                    error_code="TimeoutError",
                    details={"command": prepared.command, "timeout": timeout},
                )
            except Exception as exc:
                log.exception("Ausführungsfehler: %s", prepared.command)
                return ActionResult.from_exception(exc)
        finally:
            # Schließt (und terminiert dank JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)
            # alle noch laufenden Nachkommen des Befehls - auch bei Timeout,
            # damit nichts über das Kommando-Lebensende hinaus weiterläuft.
            sandbox.close()

        stdout = self._limit(stdout_bytes.decode("utf-8", errors="replace"))
        stderr = self._limit(stderr_bytes.decode("utf-8", errors="replace"))

        engine = CommandEngine()
        return engine.build_result(
            prepared,
            returncode=proc.returncode,
            stdout=stdout,
            stderr=stderr,
        )

    def execute_sync(
        self,
        prepared: PreparedCommand,
        dry_run: bool = False,
        cwd: str | Path | None = None,
    ) -> ActionResult:
        """Synchronous wrapper around execute() for callers without an event loop."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.execute(prepared, dry_run=dry_run, cwd=cwd))
        # Already inside a running loop: schedule on a separate thread if needed.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                asyncio.run, self.execute(prepared, dry_run=dry_run, cwd=cwd)
            )
            return future.result()

    @staticmethod
    def _build_command(shell: str, command: str) -> list[str]:
        shell = shell.lower()
        if shell in ("powershell", "pwsh"):
            return [
                "powershell.exe" if sys.platform == "win32" else "pwsh",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                command,
            ]
        if shell == "cmd":
            return ["cmd", "/c", command]
        if shell == "bash":
            return ["bash", "-c", command]
        if shell == "sh":
            return ["sh", "-c", command]
        # Unknown shell: use shlex.split on the whole string as a fallback.
        return shlex.split(command)

    @staticmethod
    def _limit(text: str) -> str:
        if len(text) <= _OUTPUT_LIMIT:
            return text
        return text[:_OUTPUT_LIMIT] + f"\n... ({len(text) - _OUTPUT_LIMIT} Zeichen gekürzt)"
