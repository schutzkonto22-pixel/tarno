# `workspace/debug/docs/adr/` — Architecture Decision Records

Numbered, dated records of significant architectural decisions and why
they were made — write one when a decision is non-obvious enough that a
future reader would otherwise have to reverse-engineer the reasoning from
code or git history.

## Index

- **ADR-001 — UI-Framework-Wahl**: why WinUI 3 (not PySide6) is the primary
  frontend. See [`ui_README.md`](../../../../src/tarno_backend/ui/ui_README.md)
  / [`gui_README.md`](../../../../src/tarno_backend/gui/gui_README.md) for
  what the superseded PySide6 stacks still around look like (TD-006).
- **ADR-002 — OVOS-Abhängigkeit**: why/how the OpenVoiceOS bus integration
  is scoped to a separate `requirements-ovos.txt`, not the default install
  (TD-007).
- **ADR-003 — Repository-Struktur**: repository layout decisions.
- **ADR-004 — Rebranding-Kandidat "Kev" & Wake-Word-Training**: the naming
  decision that led to "Tarno" as the product name (still open on scope:
  visible-layer-only vs. also renaming code identifiers — see the
  `tarno` → `tarno_backend` package rename done later for the code side).
- **ADR-005 — Pegboard- und Multi-Monitor-UI**: the drag/dock/pegboard
  panel-layout system in `src/TARNO.UI/Services/` (`PegboardService.cs`
  etc.). Renumbered from a duplicate "ADR-004" during the workspace
  restructuring — see the commit that fixed the collision.
- **ADR-006 — API-Key-Provider-Onboarding über WebView2**: the onboarding
  flow implemented by `ProviderOnboardingService.cs`. Also renumbered from
  a duplicate "ADR-004" in the same fix.

## Numbering

ADR numbers must be unique — three files were briefly all titled "ADR-004"
before a workspace cleanup pass caught and fixed it (nothing else in the
repo referenced them by number, so renumbering was safe). Check this index
before adding a new one.

## Cross-references

- Referencing C# services: [`src/TARNO.UI/README.md`](../../../../src/TARNO.UI/README.md)
- Known issues these decisions relate to: [`technical-debt-catalog.md`](../docs_README.md)
