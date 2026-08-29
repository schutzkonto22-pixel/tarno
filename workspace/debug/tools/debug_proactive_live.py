"""Live-Test des JARVIS-Proaktiv-Layers mit ECHTEN Beobachtungen.

Unterschied zu debug_proactive_voice.py (das mit vorgefertigten
ProactiveDraft-Texten arbeitet, siehe SCENARIOS dort): dieses Skript startet
den echten ProactiveEngine mit den echten Observer-Klassen aus
tarno_backend/core/proactive_engine.py, die echte Systemwerte (psutil), das echte
aktuell fokussierte Fenster (get_active_window) und eine echte .ics-Kalender-
datei lesen. Es werden also reale, gerade gemessene Zustaende ausgewertet -
nicht erfundene Beispieltexte.

Damit die real ablaufenden Bedingungen (Ablenkung, Systemlast, Termin-
Naehe, Leerlauf) sich innerhalb einer ueberschaubaren Testdauer tatsaechlich
AENDERN koennen, werden zwei rein testbezogene Anpassungen vorgenommen -
beide betreffen nur die SCHWELLENWERTE/das TEMPO, nicht die gemessenen Daten
selbst:
  - kurze tick_seconds/source_cooldown_seconds, damit man nicht 5-10 Minuten
    Cooldown pro Beobachtung abwarten muss,
  - ein SystemObserver-Schwellenwert knapp UNTER dem tatsaechlich beim Start
    gemessenen Wert, damit die echte (nicht kuenstlich erzeugte) aktuelle
    Auslastung als "ueber Schwelle" zaehlt, statt kuenstlich 90%+ Last zu
    erzeugen.

Waehrend das Skript laeuft, sollen von aussen echte Veraenderungen passieren
(z.B. ein echtes Browser-Fenster mit "youtube" im Titel geoeffnet/fokussiert
werden, echte CPU-Last durch einen echten Prozess erzeugt werden) - das
Skript selbst wartet dafuer einfach in Echtzeit (--duration Sekunden) und
protokolliert JEDEN echten Trigger mit Rohwert, Quelle, umformuliertem Text,
gesprochenem Audio (Datei-Modus, damit mitschneidbar) und Zeitstempel.

Nutzung:
    python workspace/debug/tools/debug_proactive_live.py --duration 300 --calendar-in 90
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

TARNO_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "src"
sys.path.insert(0, str(TARNO_ROOT))

from tarno_backend.ai.factory import create_provider  # noqa: E402
from tarno_backend.ai.prompts.proactive_system import PROACTIVE_SYSTEM_PROMPT  # noqa: E402
from tarno_backend.core.calendar_service import CalendarService  # noqa: E402
from tarno_backend.core.config import TarnoConfig  # noqa: E402
from tarno_backend.core.proactive_engine import (  # noqa: E402
    IdleCuriosityObserver,
    ProactiveDraft,
    ProactiveEngine,
    SystemObserver,
    TimeCalendarObserver,
    UserBehaviorObserver,
)
from tarno_backend.extensions.task_planner import PlannedTask, TaskPlanner, TaskStatus  # noqa: E402
from tarno_backend.voice.synthesizer import SpeechSynthesizer  # noqa: E402

LOG_ROOT = Path(r"H:\CascadeProjects\proactive-voice-debug-logs")


def _write_test_calendar(path: Path, seconds_from_now: float) -> None:
    """Schreibt eine ECHTE .ics-Datei mit einem Termin, dessen Startzeit
    tatsaechlich `seconds_from_now` Sekunden nach dem jetzigen Realzeitpunkt
    liegt - TimeCalendarObserver liest diese Datei ganz normal ueber
    CalendarService, wie er es auch bei einer echten Nutzerkalenderdatei tun
    wuerde. Der Termin selbst ist Testdaten, der Lese-/Zeitvergleichs-
    mechanismus ist der echte Produktionscode.

    (KERNFIX: fruehere Version verwechselte die --calendar-in-Sekunden mit
    Minuten - timedelta(minutes=X) statt timedelta(seconds=X) - wodurch der
    Testtermin real ~90 Minuten statt ~90 Sekunden in der Zukunft lag und
    der TimeCalendarObserver waehrend der gesamten Testlaufzeit nie ausloeste.)"""
    start = datetime.now() + timedelta(seconds=seconds_from_now)
    ics = (
        "BEGIN:VCALENDAR\r\nVERSION:2.0\r\n"
        "BEGIN:VEVENT\r\n"
        f"DTSTART:{start.strftime('%Y%m%dT%H%M%S')}\r\n"
        "SUMMARY:Live-Test Team-Meeting\r\n"
        "END:VEVENT\r\n"
        "END:VCALENDAR\r\n"
    )
    path.write_text(ics, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=240.0, help="Testdauer in Sekunden Echtzeit")
    parser.add_argument("--calendar-in", type=float, default=90.0, help="Termin X Sekunden ab jetzt (echte Uhrzeit)")
    parser.add_argument("--no-speak", action="store_true")
    args = parser.parse_args()

    session_dir = LOG_ROOT / f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_dir.mkdir(parents=True, exist_ok=True)
    transcript: list[dict] = []
    tick_log: list[dict] = []

    config = TarnoConfig.load()
    provider = create_provider(config)
    synth: SpeechSynthesizer | None = None
    if not args.no_speak:
        config.audio.tts_streaming = False  # Datei-Modus, siehe debug_proactive_voice.py
        print("Initialisiere Sprachausgabe (Piper, Datei-Modus)...")
        synth = SpeechSynthesizer(config.audio)

    # --- Echte Kalenderdatei fuer den TimeCalendarObserver ---
    calendar_path = session_dir / "live_test_calendar.ics"
    _write_test_calendar(calendar_path, args.calendar_in)
    calendar_service = CalendarService(calendar_file=str(calendar_path))

    # --- Echte Systemmessung als Kalibrierungsbasis fuer den Testschwellenwert ---
    import psutil

    baseline_cpu = psutil.cpu_percent(interval=0.5)
    baseline_ram = psutil.virtual_memory().percent
    print(f"Echte Basiswerte gerade gemessen: CPU={baseline_cpu:.1f}%  RAM={baseline_ram:.1f}%")
    # Schwelle knapp unter dem echten Ist-Wert, NICHT ein erfundener Wert -
    # der Observer meldet also die tatsaechlich gemessene aktuelle Auslastung.
    system_observer = SystemObserver(
        cpu_threshold=max(1.0, baseline_cpu - 2.0),
        ram_threshold=max(1.0, baseline_ram - 2.0),
        disk_threshold=99.9,  # Platte bewusst nicht im Testfokus
    )

    # --- Echter TaskPlanner mit einer echten offenen Aufgabe ---
    def _approval_cb(_task: PlannedTask) -> bool:
        return False  # Testkontext: nichts wird tatsaechlich ausgefuehrt

    def _executor(_action: str, _params: dict) -> None:
        return None

    task_planner = TaskPlanner(
        approval_callback=_approval_cb,
        executor=_executor,
        storage_path=session_dir / "live_test_tasks.json",
    )
    task_planner.create("Live-Test: Rechnung an Kunde X versenden", steps=[])
    # Simuliert echten Leerlauf: "letzte Interaktion" liegt real in der
    # Vergangenheit vor dem Skriptstart, damit idle_minutes sofort ueberschritten ist.
    session_start = datetime.now() - timedelta(minutes=5)

    user_behavior_observer = UserBehaviorObserver(
        distraction_apps=["youtube", "netflix", "twitch"],
        calendar_service=calendar_service,
        minutes_before_action=0.05,  # 3 Sekunden Echtzeit statt 5 Minuten - nur das Tempo ist testangepasst
        calendar_lookahead_minutes=5,
        close_distraction_window=False,
    )
    calendar_observer = TimeCalendarObserver(calendar_service, lookahead_minutes=5)
    idle_observer = IdleCuriosityObserver(
        task_planner,
        get_last_interaction=lambda: session_start,
        idle_minutes=0.1,  # 6 Sekunden Echtzeit
    )

    def _on_trigger(draft: ProactiveDraft) -> None:
        print(f"\n>>> ECHTER TRIGGER von '{draft.source}' (score={draft.score})")
        print(f"    Roh: {draft.message}")
        t0 = time.monotonic()
        try:
            response = provider.send(
                messages=[{"role": "user", "content": f"Beobachtung von '{draft.source}': {draft.message}"}],
                system=PROACTIVE_SYSTEM_PROMPT,
            )
            text = (response.text or "").strip()
            spoken = text if text and not response.is_error else draft.message
            embellish_error = None if (text and not response.is_error) else (response.error_message or "leer")
        except Exception as exc:
            spoken = draft.message
            embellish_error = repr(exc)
        embellish_elapsed = time.monotonic() - t0
        print(f"    Gesprochen: {spoken}")

        audio_path = None
        if synth is not None:
            cache_dir = synth._cache_dir
            before = {p: p.stat().st_mtime for p in cache_dir.glob("*") if p.is_file()}
            try:
                synth.speak(spoken)
            except Exception:
                pass
            new_files = [p for p in cache_dir.glob("*") if p.is_file() and p.stat().st_mtime > before.get(p, 0.0)]
            if new_files:
                newest = max(new_files, key=lambda p: p.stat().st_mtime)
                dest = session_dir / f"trigger_{len(transcript) + 1:02d}_{draft.source}{newest.suffix}"
                shutil.copy2(newest, dest)
                audio_path = str(dest)

        transcript.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "source": draft.source,
            "score": draft.score,
            "raw_message": draft.message,
            "spoken_message": spoken,
            "embellish_fallback_used": bool(embellish_error),
            "embellish_error": embellish_error,
            "embellish_seconds": round(embellish_elapsed, 3),
            "audio_file": audio_path,
        })
        (session_dir / "transcript.json").write_text(
            json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    engine = ProactiveEngine(
        observers=[system_observer, user_behavior_observer, calendar_observer, idle_observer],
        on_trigger=_on_trigger,
        tick_seconds=3.0,
        score_threshold=1,  # niedrig, damit reale (nicht kuenstlich extreme) Messwerte zaehlen
        source_cooldown_seconds=15.0,
    )

    print(f"\nStarte echten ProactiveEngine-Loop fuer {args.duration:.0f}s Echtzeit...")
    print(f"Echter Test-Kalendertermin in {args.calendar_in:.0f}s: {calendar_path}")
    print("Waehrenddessen von aussen reale Bedingungen aendern (Fenster fokussieren, Last erzeugen, ...).\n")

    def _log_tick(label: str) -> None:
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent
        entry = {"timestamp": datetime.now().isoformat(timespec="seconds"), "label": label, "cpu": cpu, "ram": ram}
        tick_log.append(entry)
        print(f"[{entry['timestamp']}] {label} - CPU={cpu:.1f}% RAM={ram:.1f}%")

    engine.start()
    try:
        elapsed = 0.0
        step = 5.0
        while elapsed < args.duration:
            time.sleep(step)
            elapsed += step
            _log_tick(f"lebendig ({elapsed:.0f}/{args.duration:.0f}s)")
    finally:
        engine.stop()

    (session_dir / "system_ticks.json").write_text(
        json.dumps(tick_log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nFertig. {len(transcript)} echte(r) Trigger.")
    print(f"Transkript: {session_dir / 'transcript.json'}")
    print(f"System-Verlauf: {session_dir / 'system_ticks.json'}")
    print(f"Audio/Ordner: {session_dir}")


if __name__ == "__main__":
    main()
