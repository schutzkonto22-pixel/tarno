"""Tool set for the native agentic coding backend (native_agent.py).

Deliberately mirrors the shape of tarno/desktop/file_manager.py's existing
tools (read_file/write_file/list_directory/search_files) and reuses the SAME
workspace-path-containment security (tarno.core.workspace.resolve_for_read/
resolve_for_write/is_allowed) instead of re-implementing it - a second,
subtly-different path-safety check would be exactly the kind of drift that
turns into a sandbox-escape bug later.

Three tools are genuinely NEW here, because file_manager.py doesn't have
them and Aider's SEARCH/REPLACE-diff format is what this backend exists to
replace:
  - grep(pattern, path):    content search, not just filename glob
    (file_manager.search_files only matches filenames).
  - edit_file(...):         a targeted, uniqueness-checked string replace -
    read the whole file into the model's context only once, then apply small
    precise edits, instead of asking the model to reproduce (and risk
    mis-formatting) a SEARCH/REPLACE diff block or the entire file content.
  - run_command(command):   workspace-scoped shell execution so the agent can
    actually run tests/build/lint and react to real failures, instead of
    reporting a change as done without ever having verified it (see
    aider_adapter.py's known limitation, documented in the backend's own
    module docstring).
"""

from __future__ import annotations

import logging
import subprocess
from typing import Any

from tarno_backend.ai.tool_registry import ToolDefinition, ToolRegistry
from tarno_backend.core.workspace import is_allowed, list_roots, resolve_for_read, resolve_for_write
from tarno_backend.desktop.file_manager import list_directory, search_files, write_file

log = logging.getLogger(__name__)

# file_manager.read_file() truncates at 4000 chars, tuned for a chat answer
# read out loud/displayed inline - far too aggressive for a coding agent that
# genuinely needs full file content to edit correctly. A higher, still-finite
# cap keeps a single pathological huge file from blowing the whole context
# window, while covering the overwhelming majority of real source files whole.
CODING_READ_MAX_CHARS = 60_000

# Hard ceiling on run_command's timeout, independent of whatever
# CommandExecutionConfig.timeout says - an autonomous coding loop must never
# be able to hang the whole task indefinitely just because the config default
# happens to be 0 ("wait for completion"), which is fine for a user-initiated
# single command but not for a tool an LLM can call in a loop by itself.
RUN_COMMAND_MAX_TIMEOUT_S = 180.0
RUN_COMMAND_DEFAULT_TIMEOUT_S = 60.0
RUN_COMMAND_OUTPUT_MAX_CHARS = 8_000

# Text-ish extensions grep bothers to open - skips binaries (images, compiled
# artifacts, venvs' .pyc, ...) instead of trying to decode them, both for
# speed and because a UnicodeDecodeError on a random .png would otherwise
# have to be swallowed per-file anyway.
_GREP_TEXT_EXTENSIONS = {
    ".py", ".kt", ".kts", ".java", ".c", ".h", ".cpp", ".hpp", ".cs", ".xaml",
    ".js", ".ts", ".tsx", ".jsx", ".json", ".yaml", ".yml", ".toml", ".ini",
    ".cfg", ".md", ".txt", ".xml", ".html", ".css", ".sh", ".bat", ".ps1",
    ".gradle", ".properties", ".sql", ".proto", ".rs", ".go",
}
_GREP_SKIP_DIR_NAMES = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", "build", "dist",
    ".gradle", "obj", "bin", ".idea", ".vs",
}


def _coding_read_file(filepath: str, workspace: dict[str, Any] | None) -> str:
    path = resolve_for_read(filepath, workspace)
    if path is None:
        return f"Datei nicht gefunden: {filepath}"
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"Fehler beim Lesen: {exc}"
    if len(content) > CODING_READ_MAX_CHARS:
        content = content[:CODING_READ_MAX_CHARS] + f"\n...(abgeschnitten, {len(content)} Zeichen gesamt)"
    return content


def _grep(pattern: str, path: str, workspace: dict[str, Any] | None, max_results: int = 50) -> str:
    import re

    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return f"Ungültiger regulärer Ausdruck '{pattern}': {exc}"

    roots = list_roots(workspace)
    if not roots:
        return "Kein Workspace-Root vorhanden."

    # Optionaler Unterordner, in dem gesucht wird - wie grep() der anderen
    # Tools hier auf alle Roots angewendet, falls leer/'.'.
    subdir = path.strip() if path and path.strip() not in (".", "") else ""

    matches: list[str] = []
    for _id, _name, root in roots:
        base = root if not subdir else (root / subdir).resolve()
        # KERNFIX (Code-Review): anders als resolve_for_read/resolve_for_write
        # wurde 'base' hier nie gegen is_allowed() geprüft - ein path wie
        # '../../../../Windows/System32' hätte (root / subdir).resolve() aus
        # dem Workspace heraus eskalieren lassen können, ohne dass irgendeine
        # Grenze das abgefangen hätte. Gleicher Schutz wie list_directory()
        # in file_manager.py.
        if not is_allowed(base, workspace):
            continue
        if not base.is_dir():
            continue
        for file_path in base.rglob("*"):
            if len(matches) >= max_results:
                break
            if not file_path.is_file():
                continue
            if any(part in _GREP_SKIP_DIR_NAMES for part in file_path.parts):
                continue
            if file_path.suffix.lower() not in _GREP_TEXT_EXTENSIONS:
                continue
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    matches.append(f"{file_path}:{line_no}: {line.strip()[:200]}")
                    if len(matches) >= max_results:
                        break

    if not matches:
        return f"Keine Treffer für '{pattern}'."
    suffix = "" if len(matches) < max_results else f"\n...(auf {max_results} Treffer begrenzt)"
    return "\n".join(matches) + suffix


def _edit_file(filepath: str, old_string: str, new_string: str, workspace: dict[str, Any] | None) -> str:
    read_path = resolve_for_read(filepath, workspace)
    if read_path is None:
        return f"Datei nicht gefunden: {filepath}"

    try:
        content = read_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"Fehler beim Lesen: {exc}"

    occurrences = content.count(old_string)
    if occurrences == 0:
        return (
            f"'{old_string[:120]}' kommt nicht (mehr) exakt so in {filepath} vor - "
            "Datei evtl. schon anders formatiert, mit read_file den aktuellen Inhalt prüfen."
        )
    if occurrences > 1:
        return (
            f"'{old_string[:120]}' kommt {occurrences}x in {filepath} vor - "
            "old_string muss eindeutig sein (mehr umgebenden Kontext angeben)."
        )

    write_path = resolve_for_write(filepath, workspace)
    if write_path is None:
        return f"Sicherheitshalber darf nur innerhalb des Workspace geschrieben werden: {filepath}"

    new_content = content.replace(old_string, new_string, 1)
    try:
        write_path.write_text(new_content, encoding="utf-8")
    except Exception as exc:
        return f"Fehler beim Schreiben: {exc}"

    log.info("Datei bearbeitet: %s", write_path)
    return f"{filepath} bearbeitet (1 Ersetzung)."


def _run_command(command: str, workspace: dict[str, Any] | None, timeout: float) -> str:
    roots = list_roots(workspace)
    if not roots:
        return "Kein Workspace-Root vorhanden - kann keinen Befehl ausführen."
    cwd = str(roots[0][2])

    capped_timeout = min(max(timeout, 1.0), RUN_COMMAND_MAX_TIMEOUT_S)
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=capped_timeout,
        )
    except subprocess.TimeoutExpired:
        return f"Befehl nach {capped_timeout:.0f}s abgebrochen (Timeout): {command}"
    except Exception as exc:
        return f"Fehler beim Ausführen von '{command}': {exc}"

    output = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    if len(output) > RUN_COMMAND_OUTPUT_MAX_CHARS:
        output = output[:RUN_COMMAND_OUTPUT_MAX_CHARS] + "\n...(Ausgabe abgeschnitten)"
    return f"Exit-Code {proc.returncode}\n{output.strip()}"


def build_native_coding_tools(
    workspace: dict[str, Any] | None,
    read_only: bool,
    command_timeout: float,
) -> ToolRegistry:
    """Builds the tool set for one coding_task run - read tools always,
    write/execute tools only when NOT read_only. Mirrors the read_only
    gating CodingAgent.execute_task() already applies at the call level
    (PermissionService.request_coding_edit), so a second, redundant
    permission check per tool call isn't needed here."""
    registry = ToolRegistry()

    registry.register(ToolDefinition(
        name="read_file",
        description="Liest den vollständigen Inhalt einer Textdatei im Workspace (nur lesend).",
        input_schema={
            "type": "object",
            "properties": {"filepath": {"type": "string", "description": "Pfad relativ zum Workspace-Root"}},
            "required": ["filepath"],
        },
        handler=lambda filepath: _coding_read_file(filepath, workspace),
    ))
    registry.register(ToolDefinition(
        name="list_directory",
        description="Listet Dateien/Unterordner eines Verzeichnisses im Workspace auf.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Pfad relativ zum Workspace-Root, leer für die Wurzel(n)"}},
            "required": [],
        },
        handler=lambda path="": list_directory(path, workspace=workspace),
    ))
    registry.register(ToolDefinition(
        name="search_files",
        description="Findet Dateien nach Namens-/Glob-Muster (z.B. '*.py') im Workspace.",
        input_schema={
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "Startverzeichnis relativ zur Workspace-Wurzel, leer für alle Roots"},
                "pattern": {"type": "string", "description": "Glob-Muster, z.B. '*.py' oder '**/*.kt'"},
            },
            "required": ["pattern"],
        },
        handler=lambda pattern, directory="": search_files(directory, pattern, workspace=workspace),
    ))
    registry.register(ToolDefinition(
        name="grep",
        description=(
            "Durchsucht Dateiinhalte im Workspace per regulärem Ausdruck (nicht nur Dateinamen wie "
            "search_files) - der beste erste Schritt, um herauszufinden, WO im Projekt etwas steht."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regulärer Ausdruck"},
                "path": {"type": "string", "description": "Optionales Unterverzeichnis, leer durchsucht alles"},
            },
            "required": ["pattern"],
        },
        handler=lambda pattern, path="": _grep(pattern, path, workspace),
    ))

    if not read_only:
        registry.register(ToolDefinition(
            name="edit_file",
            description=(
                "Ersetzt EINE eindeutige Textstelle in einer bestehenden Datei durch neuen Text - "
                "der bevorzugte Weg, um Aenderungen vorzunehmen (praeziser und weniger fehleranfaellig "
                "als die ganze Datei neu zu schreiben). old_string muss exakt (inkl. Einrückung) und "
                "EINMALIG in der Datei vorkommen - bei Uneindeutigkeit mehr umgebenden Kontext angeben."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Pfad relativ zum Workspace-Root"},
                    "old_string": {"type": "string", "description": "Exakt zu ersetzender, eindeutiger Text"},
                    "new_string": {"type": "string", "description": "Ersatztext"},
                },
                "required": ["filepath", "old_string", "new_string"],
            },
            handler=lambda filepath, old_string, new_string: _edit_file(filepath, old_string, new_string, workspace),
        ))
        registry.register(ToolDefinition(
            name="write_file",
            description=(
                "Schreibt eine KOMPLETT NEUE Datei oder überschreibt eine bestehende vollständig. "
                "Für Änderungen an bestehenden Dateien IMMER edit_file bevorzugen, das hier nur für "
                "neue Dateien oder wenn wirklich der gesamte Inhalt ersetzt werden soll."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Pfad relativ zum Workspace-Root"},
                    "content": {"type": "string", "description": "Vollständiger neuer Dateiinhalt"},
                },
                "required": ["filepath", "content"],
            },
            handler=lambda filepath, content: write_file(filepath, content, workspace=workspace),
        ))
        registry.register(ToolDefinition(
            name="run_command",
            description=(
                "Führt einen Shell-Befehl im Workspace-Root aus (z.B. Tests, Build, Linter) und liefert "
                "Exit-Code + Ausgabe zurück - nutze das, um eigene Änderungen zu VERIFIZIEREN statt sie "
                "unverifiziert als erledigt zu melden."
            ),
            input_schema={
                "type": "object",
                "properties": {"command": {"type": "string", "description": "Auszuführender Shell-Befehl"}},
                "required": ["command"],
            },
            handler=lambda command: _run_command(command, workspace, command_timeout or RUN_COMMAND_DEFAULT_TIMEOUT_S),
        ))

    return registry
