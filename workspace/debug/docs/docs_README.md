# `workspace/debug/docs/` — project documentation

Everything that isn't a `<name>_README.md` next to the code it describes:
architecture decisions, setup guides, audits, and status reports.

## Files

- **`adr/`**: Architecture Decision Records — see [`adr_README.md`](adr/adr_README.md).
- **`technical-debt-catalog.md`**: the running list of known issues (TD-001
  through TD-025+), most either resolved or explicitly deferred with a
  reason. Referenced throughout the `<name>_README.md` files by TD number —
  check here for the full writeup behind a TD reference.
- **`test-coverage-report.md`**: coverage summary by subsystem.
- **`dependency-audit.md`**: audit of third-party dependencies.
- **`api-keys.md`**: how to set up LLM provider API keys — linked directly
  from the first-start wizard's "Anleitung öffnen" button (see
  [`first_start_README.md`](../../../src/tarno_backend/first_start/first_start_README.md)).
- **`coding-agent-setup.md`**: quick-start guide for the coding-agent
  backend (see [`coding_README.md`](../../../src/tarno_backend/ai/coding/coding_README.md)).
- **`plugins.md`**: plugin developer guide (see
  [`plugins_README.md`](../../../src/tarno_backend/plugins/plugins_README.md)).
- **`deployment.md`**: deployment/release guide.
- **`ui-rework-ist-state-b831bc.md`**: point-in-time status report of a UI
  rework effort — historical record, not a living doc.
- **`liquid-glass-research.md`**: research/exploration notes for a "Liquid
  Glass" visual design direction — not a decision record (see `adr/` for
  those), just working notes.
- **`mesh/zte-tasker-setup.md`**: Tasker/MacroDroid setup instructions for
  the "Node A" phone in the Dynamic Hybrid Mesh feature — hardware/OS setup
  steps, not something expressible in the mesh integration's own code
  README (see
  [`integrations_README.md`](../../../src/tarno_backend/integrations/integrations_README.md)).

## Cross-references

- Code-level READMEs these docs are cross-referenced from:
  `src/tarno_backend/*/<name>_README.md`, [`src/TARNO.UI/README.md`](../../../src/TARNO.UI/README.md)
- Planning documents (distinct from decision records — see below):
  [`workspace/plans/`](../../plans/plans_README.md)
