"""Native agentic coding backend - replaces the aider_adapter.py diff-apply
approach with a tool-calling loop in the same style TARNO's own chat engine
(tarno/core/engine.py) and the agent-pool worker (tarno/ai/pool/worker.py)
already use: read/grep/edit/run_command tools, executed via the SAME
tool_registry.execute() plumbing, iterating until the model stops calling
tools or explicitly says "FERTIG".

Why this exists (see conversation/plan notes): the Aider-backed default was
configured with a small, cheap model (mistral-small-latest) and Aider's
SEARCH/REPLACE-diff edit format - a combination that's known to have a low
success rate at actually applying edits correctly, and the adapter never ran
tests/lints to self-verify a change before declaring it done. This backend
targets the SAME kind of competence/autonomy as an interactive coding agent:
explore first (read_file/grep/list_directory/search_files), make precise
targeted edits (edit_file, uniqueness-checked - no diff-format guessing), and
VERIFY with run_command before reporting success - primarily against Mistral
models (Codestral is the coding-specialized one, see CodingConfig.model),
but model-agnostic since it goes through the same LLMProvider interface
every other provider in this codebase implements.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterator

from tarno_backend.ai.coding.adapters.base import CodingBackend, OnOutput
from tarno_backend.ai.coding.native_tools import build_native_coding_tools
from tarno_backend.ai.coding.protocol import CodingOutput
from tarno_backend.ai.factory import create_provider
from tarno_backend.ai.prompts.code_system import CODE_SYSTEM_PROMPT, build_workspace_prompt

log = logging.getLogger(__name__)

# "FERTIG" is the SAME completion keyword CODE_SYSTEM_PROMPT already commits
# the model to using (see its rule #4) - checked case-sensitively at the end
# of the response, not just "contains", so a false-positive mid-sentence
# mention (e.g. quoting an old commit message) doesn't end the loop early.
_DONE_MARKER = "FERTIG"


def _is_done(text: str) -> bool:
    stripped = text.strip()
    return stripped.endswith(_DONE_MARKER) or stripped == _DONE_MARKER


class NativeAgentBackend(CodingBackend):
    """Runs a tool-calling coding agent directly against the configured
    LLM provider - no external CLI process, no diff-text parsing."""

    name = "native"

    def __init__(self, config, secrets: Any | None = None) -> None:
        self.config = config
        self.secrets = secrets

    def run(
        self,
        prompt: str,
        workspace: dict[str, Any],
        fnames: list[str],
        read_only: bool,
        on_output: OnOutput | None = None,
    ) -> Iterator[CodingOutput]:
        try:
            provider = create_provider(
                self.config, provider_name=self.config.coding.provider, model=self.config.coding.model
            )
        except Exception as exc:
            yield CodingOutput(kind="error", text=f"Provider '{self.config.coding.provider}' nicht verfügbar: {exc}")
            return

        command_timeout = getattr(self.config.command_execution, "timeout", 0.0) or 0.0
        tools = build_native_coding_tools(workspace, read_only=read_only, command_timeout=command_timeout)
        tool_schemas = tools.get_tool_schemas() if provider.supports_tools else None

        system = CODE_SYSTEM_PROMPT + build_workspace_prompt(workspace)
        if fnames:
            system += "\n\nVom Nutzer explizit genannte Dateien (zuerst ansehen, aber nicht darauf beschränkt):\n" + "\n".join(fnames)
        if read_only:
            system += (
                "\n\nREAD-ONLY-MODUS: Du darfst edit_file/write_file/run_command NICHT nutzen (nicht "
                "verfügbar) - analysiere und beantworte nur, ohne Änderungen vorzunehmen."
            )

        max_iterations = getattr(self.config.llm, "max_coding_iterations", 0)
        if max_iterations <= 0:
            max_iterations = 30  # deutlich niedriger als Aiders 10_000-Fallback - eine
            # ausufernde Tool-Schleife soll hier auffallen, nicht endlos weiterlaufen.
        timeout = getattr(self.config.coding, "timeout", 0.0) or 0.0
        start = time.monotonic()

        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        # KERNFIX (live gefunden per workspace/debug/tools/debug_coding_agent.py --advanced):
        # ohne diese Sperre kann das Modell denselben (fehlschlagenden)
        # edit_file-Aufruf wortgleich wiederholen, obwohl das Tool-Ergebnis
        # (inkl. der Datei-Inhalt aus einem zwischenzeitlichen read_file) klar
        # zeigt, dass sich nichts geändert hat - beobachtet: alter_string mit
        # EINER Leerzeile statt der tatsächlich vorhandenen ZWEI, 15x identisch
        # wiederholt, bis das Iterationslimit riss. Statt den Aufruf einfach
        # nochmal auszuführen (idempotent, aendert nichts), wird ein
        # identischer Folgeaufruf abgefangen und stattdessen eine deutlich
        # schaerfere Korrektur zurueckgegeben.
        last_call_signature: tuple[str, str] | None = None

        def _call_signature(name: str, tool_input: dict[str, Any]) -> tuple[str, str]:
            # edit_file spezifisch NUR auf (filepath, old_string) reduzieren,
            # nicht den ganzen Input: das Modell variiert new_string oft leicht
            # zwischen Versuchen (z.B. neuer Import ergänzt), waehrend
            # old_string - der eigentliche Fehlschlags-Grund - unveraendert
            # falsch bleibt. Live beobachtet: 15+ Wiederholungen mit
            # identischem alten_string, aber jedes Mal anderem neuen_string,
            # die ein reiner Volltreffer-Vergleich NICHT erkannt hätte.
            if name == "edit_file":
                return (name, repr((tool_input.get("filepath"), tool_input.get("old_string"))))
            return (name, repr(sorted(tool_input.items())))

        def _emit(kind: str, text: str) -> CodingOutput:
            output = CodingOutput(kind=kind, text=text, timestamp_ms=int(time.monotonic() * 1000))
            if on_output:
                on_output(output)
            return output

        try:
            response = provider.send(messages=messages, system=system, tools=tool_schemas)
        except Exception as exc:
            log.exception("native-agent: initialer Provider-Aufruf fehlgeschlagen")
            yield _emit("error", f"Fehler beim Aufruf von {provider.name}: {exc}")
            return

        if not tool_schemas:
            # Provider unterstützt keine Tools (oder read_only ohne jeden Tool-Zugriff
            # ergibt keinen Sinn) - einfach die Textantwort als Ergebnis liefern.
            yield _emit("summary", response.text.strip() or "Keine Antwort erhalten.")
            return

        for i in range(max_iterations):
            if timeout and (time.monotonic() - start) > timeout:
                yield _emit("error", f"Zeitlimit ({timeout:.0f}s) erreicht - Task abgebrochen.")
                return

            if response.text:
                yield _emit("text", response.text)

            if not response.tool_calls:
                # Kein Tool-Aufruf mehr: entweder sauber mit FERTIG beendet, oder das
                # Modell hat sich (regelwidrig, siehe CODE_SYSTEM_PROMPT Regel #2)
                # nicht daran gehalten - in beiden Fällen ist hier Schluss, ein
                # Continue-Prompt für eine bereits toolfreie Antwort würde nur
                # denselben Text wiederholen lassen.
                summary = response.text.strip() or "Coding-Task abgeschlossen."
                yield _emit("summary", summary)
                return

            content: list[dict[str, Any]] = []
            if response.text:
                content.append({"type": "text", "text": response.text})
            for tc in response.tool_calls:
                content.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input})
            messages.append({"role": "assistant", "content": content})

            for tc in response.tool_calls:
                signature = _call_signature(tc.name, tc.input)
                if signature == last_call_signature:
                    message = (
                        f"WIEDERHOLUNG ERKANNT: Das ist EXAKT derselbe {tc.name}-Aufruf wie eben - "
                        "der ist gerade schon fehlgeschlagen und wird beim erneuten Ausführen mit "
                        "identischem Input zwangsläufig wieder fehlschlagen, das Tool wurde deshalb "
                        "NICHT nochmal ausgeführt. Vergleiche alter_string ZEICHENGENAU (auch Anzahl "
                        "Leerzeilen/Einrückung) mit dem zuletzt per read_file gezeigten Inhalt, oder "
                        "wähle einen kürzeren, garantiert eindeutigen Ausschnitt als alter_string."
                    )
                    yield _emit("tool", f"{tc.name}({tc.input}) -> [übersprungen: Wiederholung] {message[:400]}")
                    messages.append({
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": tc.id, "content": message}],
                    })
                    continue

                result = tools.execute(tc.name, tc.input)
                last_call_signature = signature
                # 400 statt vorher 200 Zeichen: bei edit_file-Fehlschlägen
                # (Uneindeutigkeit/nicht gefunden) reicht 200 oft nicht, um im
                # UI/Log zu erkennen, WARUM ein alter_string nicht matchte -
                # betrifft nur diese Anzeige, das volle result.message geht
                # dem Modell unten ohnehin ungekürzt zurück.
                yield _emit("tool", f"{tc.name}({tc.input}) -> {result.message[:400]}")
                messages.append({
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": tc.id, "content": result.message}],
                })

            try:
                response = provider.send_tool_result(messages=messages, system=system, tools=tool_schemas)
            except Exception as exc:
                log.exception("native-agent: send_tool_result fehlgeschlagen (Iteration %d)", i + 1)
                yield _emit("error", f"Fehler beim Aufruf von {provider.name}: {exc}")
                return

            if _is_done(response.text):
                summary = response.text.strip()
                yield _emit("summary", summary)
                return
        else:
            yield _emit(
                "error",
                f"Iterationslimit ({max_iterations}) erreicht, ohne dass der Agent FERTIG gemeldet hat.",
            )
