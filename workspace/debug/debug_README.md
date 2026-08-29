# `workspace/debug/` — everything non-shipped

Tests, docs, the (incomplete) installer pipeline, and manual debug
tooling — bundled together specifically to keep them out of
`src/tarno_backend/`, so a first-time reader of `src/` sees only actual
product code, not test/tooling clutter mixed in next to it.

## Contents

- **[`tests/`](tests/tests_README.md)**: the automated `unittest` suite, run in CI.
- **[`docs/`](docs/docs_README.md)**: architecture decisions, setup guides, audits, status reports.
- **[`installer/`](installer/installer_README.md)**: Windows installer build — ⚠️ mostly a gap, see its README.
- **[`tools/`](tools/tools_README.md)**: standalone manual debug/diagnostic scripts, not run in CI.

## Cross-references

- Parent wrapper: [`workspace/`](../workspace_README.md)
- Product source these all test/document/tool around: `src/tarno_backend/`,
  `src/TARNO.UI/`
