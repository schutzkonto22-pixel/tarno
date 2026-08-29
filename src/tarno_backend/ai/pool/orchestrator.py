"""PoolOrchestrator: Lead-Decompose/Assign/Collect/Merge-Loop
(Coding-Agent-Pool-Plan, Phase 2).

Koordinationsmodell (bestaetigte Entscheidung, siehe Plan): Orchestrator +
Worker. Der Lead zerlegt die Aufgabe in Teilschritte, weist sie Workern zu,
sammelt Berichte ein und hat als EINZIGER die Merge-Autoritaet - der
Orchestrator wendet NIE Worker-Output direkt an, nur das Lead-Ergebnis.

Die tatsaechliche Dateiaenderung passiert NICHT hier, sondern erst danach
im Aufrufer via edit_apply.apply_merged_instruction() - das haelt die
Orchestrator-Logik unabhaengig von Aider/Git testbar.
"""

from __future__ import annotations

import asyncio
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable

from tarno_backend.ai.pool.models import PoolConfig, PoolMessage, SubTask
from tarno_backend.ai.pool.worker import PoolWorker, run_with_tool_loop, workspace_label
from tarno_backend.ai.prompts.pool_system import (
    LEAD_MERGE_SYSTEM_PROMPT,
    LEAD_SYSTEM_PROMPT,
    REVISION_MARKER,
    SUBTASK_MARKER,
)
from tarno_backend.core.workspace import resolve_for_read

if TYPE_CHECKING:
    from tarno_backend.core.config import TarnoConfig

log = logging.getLogger(__name__)

# Live-bestaetigter Bug (Kontext-Effizienz-Plan-Nachtrag): Lead/Worker
# bekamen bisher nur Dateinamen als TEXT, nie den Inhalt - jede
# Revisionsrunde wiederholte dadurch dieselbe "brauche Code"-Antwort.
# Diese Konstanten begrenzen, wie viel tatsaechlicher Dateiinhalt automatisch
# in die Prompts eingebettet wird (grosszuegig genug fuer ein paar
# Kern-Dateien, aber nicht unbegrenzt).
_MAX_CHARS_PER_FILE = 4000
_MAX_TOTAL_EMBED_CHARS = 16000


def _read_fnames_content(workspace: dict[str, Any] | None, fnames: list[str]) -> str:
    """Liest die vom Nutzer explizit angegebenen Dateien direkt vom
    Workspace-Root und baut einen Text-Block fuer die Prompts von Lead/
    Worker - ergaenzt (nicht ersetzt) das read_file-Tool aus worker.py,
    das fuer Dateien greift, die der Nutzer nicht explizit genannt hat."""
    if not workspace or not fnames:
        return ""
    blocks: list[str] = []
    total = 0
    for fname in fnames:
        if total >= _MAX_TOTAL_EMBED_CHARS:
            blocks.append("... (weitere Dateien aus Platzgründen nicht eingebettet, read_file-Tool nutzen)")
            break
        path = resolve_for_read(fname, workspace)
        if path is None:
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            content = f"(Fehler beim Lesen: {exc})"
        if len(content) > _MAX_CHARS_PER_FILE:
            content = content[:_MAX_CHARS_PER_FILE] + "\n...(gekürzt)"
        total += len(content)
        blocks.append(f"### {fname}\n```\n{content}\n```")
    if not blocks:
        return ""
    return "Dateiinhalte:\n" + "\n\n".join(blocks)

OnMessage = Callable[[PoolMessage], None]

_SUBTASK_PATTERN = re.compile(
    re.escape(SUBTASK_MARKER) + r"\s+(?P<slug>\S+)\s*\n(?P<desc>.*?)(?=\n"
    + re.escape(SUBTASK_MARKER) + r"\s+\S+|\Z)",
    re.DOTALL,
)


def _parse_subtasks(text: str, valid_slugs: set[str]) -> list[SubTask]:
    subtasks: list[SubTask] = []
    for i, m in enumerate(_SUBTASK_PATTERN.finditer(text)):
        slug = m.group("slug").strip()
        description = m.group("desc").strip()
        if slug not in valid_slugs or not description:
            continue
        subtasks.append(SubTask(id=f"st-{i + 1}", description=description, assigned_to=slug))
    return subtasks


class PoolOrchestrator:
    """Treibt einen kompletten Pool-Lauf (Decompose -> Dispatch -> Merge,
    ggf. mehrere Revisions-Runden) fuer eine Aufgabe an."""

    def __init__(
        self,
        config: "TarnoConfig",
        pool_config: PoolConfig,
        workers: dict[str, PoolWorker] | None = None,
    ) -> None:
        self._config = config
        self._pool_config = pool_config
        self._workers: dict[str, PoolWorker] = workers or {
            spec.slug: PoolWorker(spec, config) for spec in pool_config.agents
        }
        self._lead_slug = pool_config.lead.slug
        max_workers = max(len(pool_config.workers), 1)
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tarno_pool")

    def close(self) -> None:
        self._executor.shutdown(wait=False)

    async def run(
        self,
        prompt: str,
        fnames: list[str] | None = None,
        on_message: OnMessage | None = None,
        workspace: dict[str, Any] | None = None,
    ) -> PoolMessage:
        """Fuehrt einen vollstaendigen Pool-Lauf aus und gibt die finale
        Merge-Entscheidung des Leads zurueck (PoolMessage, kind='merge_decision').

        workspace (dict, gleiche Form wie tarno/grpc/server.py's workspace_dict)
        wird an Lead/Worker durchgereicht, damit sie echten Dateiinhalt sehen
        statt nur Dateinamen als Text (siehe _read_fnames_content) und Worker
        das read_file-Tool nutzen koennen (siehe worker.py)."""
        fnames = fnames or []
        history: list[PoolMessage] = []

        def _emit(message: PoolMessage) -> None:
            history.append(message)
            if on_message is not None:
                on_message(message)

        subtasks = await self._decompose(prompt, fnames, workspace, _emit)
        if not subtasks:
            text = "Der Lead-Agent konnte die Aufgabe nicht in gültige Teilschritte zerlegen."
            final = PoolMessage(from_agent=self._lead_slug, to_agent="broadcast", kind="merge_decision", text=text)
            _emit(final)
            return final

        # In die history emittieren (statt nur an den Lead-Decompose-Prompt
        # anzuhaengen), damit derselbe Dateiinhalt automatisch ueber
        # PoolWorker._build_prompt's "Bisheriger Kontext" auch bei JEDEM
        # Worker ankommt, nicht nur beim Lead. Die Workspace-Grundierung wird
        # IMMER emittiert (nicht nur wenn fnames/Dateiinhalt vorhanden sind) -
        # live bestaetigter Bug: ohne jede Erwaehnung des echten Workspace-
        # Pfads faellt das Modell auf generische Trainings-Annahmen
        # (Python/GitHub) zurueck, statt am tatsaechlichen Projekt zu arbeiten.
        label = workspace_label(workspace)
        file_content = _read_fnames_content(workspace, fnames)
        status_text = "\n\n".join(part for part in (label, file_content) if part)
        if status_text:
            _emit(PoolMessage(from_agent="pool", to_agent="broadcast", kind="status", text=status_text))

        max_iterations = max(1, self._config.coding.pool_max_lead_iterations)
        final_message: PoolMessage | None = None
        for iteration in range(max_iterations):
            reports = await self._dispatch(subtasks, history, workspace, _emit)
            merge_message = await self._merge(prompt, reports, workspace, _emit)
            if not merge_message.text.startswith(REVISION_MARKER):
                return merge_message
            final_message = merge_message
            log.info(
                "[Pool] Lead fordert Revision an (Runde %d/%d): %s",
                iteration + 1, max_iterations, merge_message.text,
            )

        # Iterationslimit erreicht, ohne dass der Lead eine finale Anweisung
        # ohne Revisions-Marker geliefert hat - bestmoegliches Ergebnis
        # zurueckgeben statt endlos weiterzumachen.
        assert final_message is not None
        return final_message

    async def _decompose(
        self, prompt: str, fnames: list[str], workspace: dict[str, Any] | None, emit: OnMessage
    ) -> list[SubTask]:
        worker_specs = self._pool_config.workers
        worker_list = "\n".join(f"- {w.slug}" for w in worker_specs)
        fnames_text = ", ".join(fnames) if fnames else "(keine Angabe - list_directory-Tool nutzen, um die echte Struktur zu sehen)"
        label = workspace_label(workspace)
        decompose_parts = [label] if label else []
        decompose_parts.append(
            f"Aufgabe:\n{prompt}\n\n"
            f"Verfügbare Worker:\n{worker_list}\n\n"
            f"Betroffene Dateien: {fnames_text}"
        )
        file_content = _read_fnames_content(workspace, fnames)
        if file_content:
            decompose_parts.append(file_content)
        decompose_prompt = "\n\n".join(decompose_parts)
        lead = self._workers[self._lead_slug]
        response = await self._call_provider(lead, decompose_prompt, LEAD_SYSTEM_PROMPT, workspace)
        text = response.strip()
        emit(PoolMessage(from_agent=self._lead_slug, to_agent="broadcast", kind="assign", text=text))

        valid_slugs = {w.slug for w in worker_specs}
        return _parse_subtasks(text, valid_slugs)

    async def _dispatch(
        self,
        subtasks: list[SubTask],
        history: list[PoolMessage],
        workspace: dict[str, Any] | None,
        emit: OnMessage,
    ) -> list[PoolMessage]:
        assignable = [st for st in subtasks if st.assigned_to in self._workers]
        for st in assignable:
            emit(PoolMessage(
                from_agent=self._lead_slug, to_agent=st.assigned_to, kind="assign",
                text=st.description, sub_task_id=st.id,
            ))

        context = list(history)
        tasks = [
            self._workers[st.assigned_to].run_subtask(
                st, context, executor=self._executor, workspace=workspace,
            )
            for st in assignable
        ]
        reports = list(await asyncio.gather(*tasks)) if tasks else []
        for report in reports:
            emit(report)
        return reports

    async def _merge(
        self,
        original_prompt: str,
        reports: list[PoolMessage],
        workspace: dict[str, Any] | None,
        emit: OnMessage,
    ) -> PoolMessage:
        reports_text = "\n\n".join(
            f"Bericht von {r.from_agent} (Teilaufgabe {r.sub_task_id}):\n{r.text}"
            for r in reports
        ) or "(keine Berichte erhalten)"
        label = workspace_label(workspace)
        merge_parts = [label] if label else []
        merge_parts.append(f"Ursprüngliche Aufgabe:\n{original_prompt}\n\nWorker-Berichte:\n{reports_text}")
        merge_prompt = "\n\n".join(merge_parts)

        lead = self._workers[self._lead_slug]
        response = await self._call_provider(lead, merge_prompt, LEAD_MERGE_SYSTEM_PROMPT, workspace)
        text = response.strip()
        message = PoolMessage(from_agent=self._lead_slug, to_agent="broadcast", kind="merge_decision", text=text)
        emit(message)
        return message

    async def _call_provider(
        self, worker: PoolWorker, prompt: str, system: str, workspace: dict[str, Any] | None = None,
    ) -> str:
        loop = asyncio.get_running_loop()
        max_iterations = (
            self._config.coding.pool_worker_max_tool_iterations if self._config else 1
        )
        try:
            return await loop.run_in_executor(
                self._executor,
                run_with_tool_loop,
                worker.provider, prompt, system, workspace, max_iterations,
            )
        except Exception:
            log.exception("[Pool] Lead-Aufruf ('%s') fehlgeschlagen", worker.agent_spec.slug)
            return ""
