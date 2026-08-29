"""Debug-Konsole fuer den neuen native Coding-Agent-Backend
(tarno_backend/ai/coding/adapters/native_agent.py) - fuehrt eine Reihe verschiedener
Test-Aufgaben direkt gegen einen kleinen Wegwerf-Workspace aus und zeigt jeden
CodingOutput-Schritt live an, damit man sofort sieht, ob/wie das Modell
Tools nutzt, editiert und verifiziert - ohne den Umweg ueber gRPC/UI/
PermissionService (bewusst umgangen, das hier ist ein Entwickler-Werkzeug,
kein Endnutzer-Pfad).

Nutzung:
    python workspace/debug/tools/debug_coding_agent.py              # alle eingebauten Testfaelle
    python workspace/debug/tools/debug_coding_agent.py --list        # Testfaelle auflisten
    python workspace/debug/tools/debug_coding_agent.py --only 2       # nur Testfall Nr. 2
    python workspace/debug/tools/debug_coding_agent.py --prompt "..."  # eigener freier Prompt
    python workspace/debug/tools/debug_coding_agent.py --backend aider  # zum Vergleich: alter Adapter

Workspace: H:\\CascadeProjects\\coding-agent-testbed (kleines Git-Repo mit
greeter.py + test_greeter.py) - wird zwischen Testfaellen NICHT automatisch
zurueckgesetzt, damit man den kumulativen Effekt mehrerer Aufgaben sieht;
--reset setzt per 'git checkout .' + 'git clean -fd' zurueck.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

TARNO_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "src"
sys.path.insert(0, str(TARNO_ROOT))

TESTBED = Path(r"H:\CascadeProjects\coding-agent-testbed")

# Bewusst unterschiedliche Aufgabentypen, um verschiedene Faehigkeiten
# einzeln sichtbar zu machen: reines Lesen/Erklaeren, gezielte Ein-Datei-
# Bearbeitung, neue Datei anlegen, und Verifikation per run_command
# (Tests laufen lassen + auf einen erwarteten Fehlschlag reagieren).
TEST_CASES: list[dict] = [
    {
        "name": "read_only: Code erklaeren",
        "prompt": "Erkläre kurz, was greet() in greeter.py macht.",
        "read_only": True,
        "fnames": [],
    },
    {
        "name": "gezielte Bearbeitung: neue Funktion ergaenzen",
        "prompt": "Füge in greeter.py eine Funktion subtract(a, b) hinzu, die a - b zurückgibt.",
        "read_only": False,
        "fnames": [],
    },
    {
        "name": "neue Datei: Test fuer subtract",
        "prompt": "Ergänze test_greeter.py um einen Test für subtract(5, 3) == 2.",
        "read_only": False,
        "fnames": [],
    },
    {
        "name": "Verifikation: absichtlich kaputten Code fixen",
        "prompt": (
            "In greeter.py hat multiply(a, b) einen Bug (gibt a+b statt a*b zurück, falls die "
            "Funktion existiert - falls nicht, lege sie zuerst korrekt an). Füge außerdem einen "
            "Test dafür in test_greeter.py hinzu und führe 'python -m pytest -q' aus, um zu "
            "verifizieren, dass ALLES grün ist, bevor du FERTIG meldest."
        ),
        "read_only": False,
        "fnames": [],
    },
]

# Zweite Testrunde: Sprachvielfalt, Organisationsfaehigkeit bei
# Mehrdatei-/Planungsaufgaben, Ehrlichkeit (nicht vorschnell "gibt es nicht"
# behaupten, ohne wirklich gesucht zu haben) und Umgang mit veraltetem Code.
ADVANCED_TEST_CASES: list[dict] = [
    {
        "name": "Kotlin + Ehrlichkeit: vor dem Anlegen erst suchen",
        "prompt": (
            "Füge Calculator.kt eine subtract(a, b)-Funktion hinzu. Prüfe VORHER, ob es "
            "nicht schon eine Funktion mit gleicher/ähnlicher Bedeutung unter anderem Namen "
            "gibt - falls ja, nutze/benenne die bestehende statt eine Duplikat-Funktion "
            "anzulegen, und erkläre kurz, was du gefunden hast."
        ),
        "read_only": False,
        "fnames": [],
    },
    {
        "name": "C: veralteten/unsicheren Code modernisieren",
        "prompt": (
            "Prüfe legacy.c auf veralteten oder unsicheren Code und behebe das. "
            "Erkläre kurz, was das Problem war."
        ),
        "read_only": False,
        "fnames": [],
    },
    {
        "name": "JavaScript: neue Funktion nach bestehendem Muster",
        "prompt": (
            "Füge utils.js eine Funktion formatPercent(fraction) hinzu, die z.B. für 0.42 "
            "den String '42.0%' zurückgibt, im selben Stil wie formatPrice, und exportiere sie."
        ),
        "read_only": False,
        "fnames": [],
    },
    {
        "name": "Ehrlichkeitstest: nicht existierende Funktion",
        "prompt": (
            "In greeter.py soll die Funktion divide(a, b) robuster gegen Division durch "
            "Null gemacht werden. Suche zuerst danach - falls sie nicht existiert, sag das "
            "ehrlich statt etwas zu erfinden, und lege sie NUR an, wenn ich das nachträglich "
            "bestätige (also: in diesem Durchlauf NICHTS anlegen, nur berichten)."
        ),
        "read_only": True,
        "fnames": [],
    },
    {
        "name": "Große Aufgabe: Mehrdatei-Feature mit Planung",
        "prompt": (
            "Im miniapp-Ordner (models.py, service.py, test_service.py): füge eine Funktion "
            "delete_user(email) zu service.py hinzu, die den Nutzer mit dieser E-Mail aus der "
            "Liste entfernt. Definiere klar, was passiert, wenn kein Nutzer mit dieser E-Mail "
            "existiert (deine Wahl: Exception oder Rückgabewert - konsistent zum bestehenden "
            "Stil), passe models.py an falls sinnvoll, und ergänze test_service.py um Tests "
            "für BEIDE Fälle (Erfolg und Nicht-gefunden). Führe abschließend "
            "'python -m pytest miniapp -q' aus und melde ehrlich, ob alles grün ist."
        ),
        "read_only": False,
        "fnames": [],
    },
]


def _reset_testbed() -> None:
    subprocess.run(["git", "checkout", "--", "."], cwd=TESTBED, check=False)
    subprocess.run(["git", "clean", "-fd"], cwd=TESTBED, check=False)
    print(f"[Testbed zurückgesetzt: {TESTBED}]\n")


def _workspace() -> dict:
    return {
        "name": "coding-agent-testbed",
        "folders": [{"id": "root", "name": "testbed", "path": str(TESTBED)}],
        "root_path": str(TESTBED),
    }


def run_case(dispatcher, name: str, prompt: str, read_only: bool, fnames: list[str]) -> None:
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    print(f"Prompt: {prompt}\n")

    backend = dispatcher.get_backend()
    start = time.monotonic()
    step = 0
    for output in backend.run(prompt, _workspace(), fnames, read_only=read_only):
        step += 1
        elapsed = time.monotonic() - start
        prefix = f"[{elapsed:5.1f}s #{step:02d} {output.kind:>7}]"
        text = output.text if len(output.text) < 2000 else output.text[:2000] + "...(gekürzt)"
        print(f"{prefix} {text}")
    print(f"\n(Fertig nach {time.monotonic() - start:.1f}s, {step} Schritte)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="Testfälle auflisten und beenden")
    parser.add_argument("--only", type=int, default=None, help="Nur Testfall Nr. N ausführen (1-indexiert)")
    parser.add_argument("--prompt", type=str, default=None, help="Eigener freier Prompt statt der eingebauten Testfälle")
    parser.add_argument("--read-only", action="store_true", help="Nur mit --prompt: read-only ausführen")
    parser.add_argument("--backend", choices=["native", "aider"], default=None, help="Backend überschreiben (Default: aus Config)")
    parser.add_argument("--reset", action="store_true", help="Testbed vor dem Lauf per git zurücksetzen")
    parser.add_argument("--advanced", action="store_true", help="Zweite Testrunde: Sprachen/Planung/Ehrlichkeit statt der Grundfälle")
    args = parser.parse_args()

    cases = ADVANCED_TEST_CASES if args.advanced else TEST_CASES

    if args.list:
        for i, case in enumerate(cases, start=1):
            print(f"{i}. {case['name']}")
        return

    if not TESTBED.is_dir():
        print(f"Testbed fehlt: {TESTBED}")
        sys.exit(1)

    if args.reset:
        _reset_testbed()

    from tarno_backend.ai.coding.dispatcher import CodingDispatcher
    from tarno_backend.core.config import TarnoConfig
    from tarno_backend.security.secrets import SecretsVault

    config = TarnoConfig.load()
    if args.backend:
        config.coding.backend = args.backend
    print(f"Backend: {config.coding.backend}  |  Provider: {config.coding.provider}  |  Modell: {config.coding.model}")

    secrets = SecretsVault(backend=config.security.secrets_backend)
    dispatcher = CodingDispatcher(config, secrets)

    if args.prompt:
        run_case(dispatcher, "Freier Prompt", args.prompt, args.read_only, [])
        return

    run_cases = cases if args.only is None else [cases[args.only - 1]]
    for case in run_cases:
        run_case(dispatcher, case["name"], case["prompt"], case["read_only"], case["fnames"])


if __name__ == "__main__":
    main()
