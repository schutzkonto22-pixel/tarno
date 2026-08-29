# `workspace/future/` — parked ideas, not current TARNO source

Directions that were explored and rewound, kept for reference rather than
deleted outright. Nothing here is built, imported, or shipped by the
current product.

## Contents

- **`multiscreen-backup/`**: an earlier multi-monitor UI direction that
  was rewound. **Intentionally untracked** — see `.gitignore`'s
  `workspace/future/multiscreen-backup/` entry and comment. Present only
  on disk for whoever kept the local backup; not part of the git history
  going forward. The *current* multi-monitor/pegboard work that did ship
  lives in `src/TARNO.UI/Services/` (`PegboardService.cs` etc.) — see
  [ADR-005](../debug/docs/adr/adr_README.md) and
  [`src/TARNO.UI/README.md`](../../src/TARNO.UI/README.md), not here.

## Cross-references

- Active multi-monitor implementation: [`src/TARNO.UI/README.md`](../../src/TARNO.UI/README.md), ADR-005
- Adaptive UI canvas proposal (a *different*, not-yet-built follow-on idea):
  [`workspace/plans/block-8-adaptive-ui-canvas.md`](../plans/plans_README.md)
