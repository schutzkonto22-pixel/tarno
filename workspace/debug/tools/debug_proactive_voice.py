"""Debug-Konsole fuer den JARVIS-Proaktiv-Layer (ProactiveEngine + Umformulierung
+ echte Sprachausgabe, tarno_backend/core/proactive_engine.py + tarno_backend/ai/prompts/
proactive_system.py + tarno_backend/voice/synthesizer.py).

Ziel: reale, sich ueber die Zeit veraendernde Szenarien durchspielen (jede
Stufe baut auf der vorherigen TARNO-Reaktion auf - z.B. "Ablenkung seit 5min"
-> "seit 9min" -> "seit 9min + Termin in 3min"), dabei fuer jeden Schritt
- die rohe Beobachter-Nachricht (draft.message),
- die per PROACTIVE_SYSTEM_PROMPT umformulierte Version,
- die tatsaechlich gesprochene Ausgabe (echtes TTS ueber SpeechSynthesizer,
  keine Attrappe)
mitschneiden: vollstaendiges Text-Transkript (JSON + lesbares .log) UND die
synthetisierten Audio-Dateien werden in einen Session-Ordner kopiert.

Nutzung:
    python workspace/debug/tools/debug_proactive_voice.py                 # alle Szenarien, mit echter Sprachausgabe
    python workspace/debug/tools/debug_proactive_voice.py --no-speak       # nur Text/Log, keine Audiowiedergabe
    python workspace/debug/tools/debug_proactive_voice.py --list           # Szenarien auflisten
    python workspace/debug/tools/debug_proactive_voice.py --only 2         # nur Szenario Nr. 2 (alle Stufen)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

TARNO_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "src"
sys.path.insert(0, str(TARNO_ROOT))

from tarno_backend.ai.factory import create_provider  # noqa: E402
from tarno_backend.ai.prompts.proactive_system import PROACTIVE_SYSTEM_PROMPT  # noqa: E402
from tarno_backend.core.config import TarnoConfig  # noqa: E402
from tarno_backend.core.proactive_engine import ProactiveDraft  # noqa: E402
from tarno_backend.voice.synthesizer import SpeechSynthesizer  # noqa: E402

LOG_ROOT = Path(r"H:\CascadeProjects\proactive-voice-debug-logs")

# Jedes Szenario ist eine Liste von Stufen, die eine sich ueber Zeit
# entwickelnde reale Situation simulieren - Stufe N nimmt bewusst Bezug
# darauf, dass TARNO bei Stufe N-1 schon reagiert hat (z.B. steigende
# Ablenkungsdauer, sinkende Zeit bis zum Termin, wachsende CPU-Last),
# um zu pruefen, ob aufeinanderfolgende proaktive Meldungen sich nicht
# stumpf wiederholen und inhaltlich zur jeweils neuen Lage passen.
SCENARIOS: list[dict] = [
    {
        "name": "Ablenkung eskaliert bis Termin ueberlappt",
        "stages": [
            ProactiveDraft(
                source="user_behavior",
                message="Sie beschäftigen sich nun seit 5 Minuten mit 'youtube', sir. Nur zur Kenntnisnahme.",
                score=55,
            ),
            ProactiveDraft(
                source="user_behavior",
                message="Sie beschäftigen sich nun seit 12 Minuten mit 'youtube', sir. Nur zur Kenntnisnahme.",
                score=60,
            ),
            ProactiveDraft(
                source="user_behavior",
                message=(
                    "Sir, in Kürze steht \"Team-Meeting\" an, während Sie seit 14 Minuten mit "
                    "'youtube' beschäftigt sind."
                ),
                score=95,
            ),
        ],
    },
    {
        "name": "Systemlast steigt ueber mehrere Ticks",
        "stages": [
            ProactiveDraft(
                source="system",
                message="Ich würde anmerken, dass die Arbeitsspeicher-Auslastung bei 91% liegt, sir - deutlich über dem Schwellenwert von 90%.",
                score=62,
            ),
            ProactiveDraft(
                source="system",
                message="Ich würde anmerken, dass die Arbeitsspeicher-Auslastung bei 96% liegt, sir - deutlich über dem Schwellenwert von 90%.",
                score=72,
            ),
            ProactiveDraft(
                source="system",
                message="Ich würde anmerken, dass die Festplatten-Auslastung bei 98% liegt, sir - deutlich über dem Schwellenwert von 90%.",
                score=76,
            ),
        ],
    },
    {
        "name": "Termin rueckt naeher",
        "stages": [
            ProactiveDraft(
                source="calendar",
                message='Ein Termin steht bevor, sir: "Zahnarzt" in 15 Minuten.',
                score=70,
            ),
            ProactiveDraft(
                source="calendar",
                message='Ein Termin steht bevor, sir: "Zahnarzt" in 5 Minuten.',
                score=88,
            ),
            ProactiveDraft(
                source="calendar",
                message='Ein Termin steht bevor, sir: "Zahnarzt" in 1 Minuten.',
                score=100,
            ),
        ],
    },
    {
        "name": "Leerlauf-Neugier nach offener Aufgabe",
        "stages": [
            ProactiveDraft(
                source="idle_curiosity",
                message='Da Sie ohnehin gerade nicht beschäftigt sind, sir - die Aufgabe "Rechnung an Kunde X versenden" wartet noch auf Ihre Entscheidung.',
                score=88,
            ),
        ],
    },
]


def _run_stage(
    provider,
    synth: SpeechSynthesizer | None,
    draft: ProactiveDraft,
    stage_idx: int,
    session_dir: Path,
    transcript: list[dict],
) -> None:
    print(f"\n  -- Stufe {stage_idx + 1} ({draft.source}, score={draft.score}) --")
    print(f"     Roh:        {draft.message}")

    t0 = time.monotonic()
    try:
        response = provider.send(
            messages=[{
                "role": "user",
                "content": f"Beobachtung von '{draft.source}': {draft.message}",
            }],
            system=PROACTIVE_SYSTEM_PROMPT,
        )
        text = (response.text or "").strip()
        spoken = text if text and not response.is_error else draft.message
        embellish_error = None if (text and not response.is_error) else (response.error_message or "leere Antwort")
    except Exception as exc:  # gleiche Fallback-Logik wie engine.py._embellish_proactive_message
        spoken = draft.message
        embellish_error = repr(exc)
    embellish_elapsed = time.monotonic() - t0

    print(f"     Umformuliert: {spoken}")
    if embellish_error:
        print(f"     (Fallback auf Rohtext - Umformulierung fehlgeschlagen: {embellish_error})")
    print(f"     [{embellish_elapsed:.2f}s Umformulierung]")

    audio_path: str | None = None
    tts_elapsed: float | None = None
    tts_error: str | None = None
    if synth is not None:
        # SpeechNaturalizer.enhance() mischt zufaellige Fuellwoerter ein (siehe
        # speech_naturalizer.py, `import random`) - der final an die Engine
        # gehende Text ist damit pro Aufruf NICHT deterministisch, selbst bei
        # identischem Eingabetext. Den Cache-Dateinamen nachtraeglich per Hash
        # neu zu berechnen (erster Versuch) traf deshalb fast nie die
        # tatsaechlich gerade erzeugte Datei. Robuster: Cache-Verzeichnis vor/
        # nach dem speak()-Aufruf per mtime vergleichen und die neu
        # geschriebene(n) Datei(en) mitschneiden - unabhaengig vom Hash.
        cache_dir = synth._cache_dir
        before_mtimes = {p: p.stat().st_mtime for p in cache_dir.glob("*") if p.is_file()}

        t1 = time.monotonic()
        try:
            synth.speak(spoken)
        except Exception as exc:
            tts_error = repr(exc)
            print(f"     [TTS FEHLER: {tts_error}]")
        tts_elapsed = time.monotonic() - t1
        print(f"     [{tts_elapsed:.2f}s gesprochen]")

        try:
            new_or_touched = [
                p for p in cache_dir.glob("*")
                if p.is_file() and p.stat().st_mtime > before_mtimes.get(p, 0.0)
            ]
            if new_or_touched:
                newest = max(new_or_touched, key=lambda p: p.stat().st_mtime)
                dest = session_dir / f"stage_{len(transcript) + 1:02d}_{draft.source}{newest.suffix}"
                shutil.copy2(newest, dest)
                audio_path = str(dest)
            else:
                print("     [Warnung: keine neue/aktualisierte Cache-Datei gefunden - Audio nicht mitgeschnitten]")
        except Exception as exc:
            print(f"     [Warnung: Audio-Mitschnitt fehlgeschlagen: {exc!r}]")

    transcript.append({
        "stage": stage_idx + 1,
        "source": draft.source,
        "score": draft.score,
        "raw_message": draft.message,
        "spoken_message": spoken,
        "embellish_fallback_used": bool(embellish_error),
        "embellish_error": embellish_error,
        "embellish_seconds": round(embellish_elapsed, 3),
        "tts_seconds": round(tts_elapsed, 3) if tts_elapsed is not None else None,
        "tts_error": tts_error,
        "audio_file": audio_path,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="Szenarien auflisten und beenden")
    parser.add_argument("--only", type=int, default=None, help="nur Szenario Nr. N (1-basiert)")
    parser.add_argument("--no-speak", action="store_true", help="keine echte Sprachausgabe, nur Text/Log")
    args = parser.parse_args()

    if args.list:
        for i, sc in enumerate(SCENARIOS, 1):
            print(f"{i}. {sc['name']} ({len(sc['stages'])} Stufen)")
        return

    scenarios = SCENARIOS if args.only is None else [SCENARIOS[args.only - 1]]

    config = TarnoConfig.load()
    provider = create_provider(config)
    synth: SpeechSynthesizer | None = None
    if not args.no_speak:
        # Der Streaming-Pfad (config.audio.tts_streaming, hier Standard: true)
        # spielt Audio direkt aus dem Synthese-Stream ab und schreibt NIE in
        # den Datei-Cache (nur der File-Fallback in _speak_file() tut das) -
        # fuer den geforderten Mitschnitt erzwingen wir hier bewusst den
        # File-Pfad (identische Engine/Stimme, nur synth-to-file-dann-play
        # statt live-Chunks), damit jede Aeusserung tatsaechlich als .wav
        # existiert und kopiert werden kann.
        config.audio.tts_streaming = False
        print("Initialisiere Sprachausgabe (Piper, Datei-Modus fuer Mitschnitt)...")
        synth = SpeechSynthesizer(config.audio)

    session_dir = LOG_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=True)
    all_transcript: list[dict] = []

    for sc_idx, scenario in enumerate(scenarios, 1):
        print(f"\n=== Szenario {sc_idx}: {scenario['name']} ===")
        stage_transcript: list[dict] = []
        for stage_idx, draft in enumerate(scenario["stages"]):
            _run_stage(provider, synth, draft, stage_idx, session_dir, stage_transcript)
            # Realistische Pause zwischen Stufen, wie sie zwischen zwei
            # ProactiveEngine-Ticks/Cooldown-Ablaeufen tatsaechlich vergeht -
            # verhindert außerdem TTS-Ueberlappung beim Mitschnitt.
            time.sleep(1.0)
        all_transcript.append({"scenario": scenario["name"], "stages": stage_transcript})

    log_json = session_dir / "transcript.json"
    log_json.write_text(json.dumps(all_transcript, ensure_ascii=False, indent=2), encoding="utf-8")

    log_txt = session_dir / "transcript.log"
    lines = []
    for sc in all_transcript:
        lines.append(f"=== {sc['scenario']} ===")
        for st in sc["stages"]:
            lines.append(f"[{st['timestamp']}] Stufe {st['stage']} ({st['source']}, score={st['score']})")
            lines.append(f"  Roh:          {st['raw_message']}")
            lines.append(f"  Gesprochen:   {st['spoken_message']}")
            if st["embellish_fallback_used"]:
                lines.append(f"  ! Fallback auf Rohtext: {st['embellish_error']}")
            if st["tts_error"]:
                lines.append(f"  ! TTS-Fehler: {st['tts_error']}")
            if st["audio_file"]:
                lines.append(f"  Audio: {st['audio_file']}")
            lines.append("")
    log_txt.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nMitschnitt gespeichert: {log_json}")
    print(f"Lesbares Transkript:    {log_txt}")
    print(f"Audio-Dateien:          {session_dir}")


if __name__ == "__main__":
    main()
