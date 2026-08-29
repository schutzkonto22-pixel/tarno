# `workspace/` — everything that isn't shipped TARNO source

If it's not `src/` (product code), `config/` (runtime config), or a
repo-root project file, it lives under here. The point of this wrapper is
purely navigational: a first-time reader of the repo root sees `src/` and
immediately knows that's the product, without having to mentally filter
out tests, planning docs, and build scripts mixed in alongside it.

## Contents

- **[`debug/`](debug/debug_README.md)**: tests, docs, the installer
  pipeline, and manual debug tooling — see its README for the reasoning
  behind grouping these four together specifically.
- **[`plans/`](plans/plans_README.md)**: planning documents and specs
  (distinct from `debug/docs/adr/`'s finalized decision records).
- **[`scripts/`](scripts/scripts_README.md)**: local build/verify
  PowerShell automation.
- **[`future/`](future/future_README.md)**: parked/rewound ideas kept for
  reference, not current product code.

## Cross-references

- Product source: `src/tarno_backend/` (Python backend), `src/TARNO.UI/` (C# frontend)
- Repo-root build/verify entry points: [`CLAUDE.md`](../CLAUDE.md)
