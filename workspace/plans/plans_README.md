# `workspace/plans/` — planning documents

Working plans and specs, distinct from
[`workspace/debug/docs/adr/`](../debug/docs/adr/adr_README.md) (finalized
decision records) — these are the messier, earlier-stage documents: time
estimates, implementation megaplans, design clones, and one active feature
proposal. Largely historical once the feature they describe ships; check
git blame / dates before assuming a plan reflects the current state.

## Files

- **`tarno-70-phase-masterplan.md`**: the original 70-phase master plan
  toward full autonomy — referenced elsewhere as "Block N, Phase M" labels
  (e.g. `vision/` = "Block 7", see
  [`vision_README.md`](../../src/tarno_backend/vision/vision_README.md)).
- **`tarno-ai-engineering-spec-60e8e5.md`**: the original professional
  engineering specification and SWE-agent instructions.
- **`tarno-research-and-decision-log-60e8e5.md`**: research/decision log
  predating the formal `adr/` process.
- **`block-8-adaptive-ui-canvas.md`**: proposal for an adaptive UI canvas,
  multi-monitor pop-out, and cross-window state sync — references a
  not-yet-written `ADR-004-Adaptive-UI-Canvas.md` (numbering has since
  moved on, see `adr/adr_README.md`; if implementing this, claim the next
  free ADR number, not literally 004).
- **`tarno-ui-plan-60e8e5.md`** / **`tarno-buildmc-design-clone-60e8e5.md`**:
  planning for the deprecated PySide6 UI stacks (see
  [`ui_README.md`](../../src/tarno_backend/ui/ui_README.md) /
  [`gui_README.md`](../../src/tarno_backend/gui/gui_README.md), TD-006) —
  historical, the active frontend is `src/TARNO.UI/`.
- **`tarno-winui-plan-60e8e5.md`** / **`tarno-winui-debug-plan-60e8e5.md`**:
  implementation and XAML-compiler-debugging plans for the active WinUI 3
  frontend.
- **`tarno-installer-plan-60e8e5.md`**: original installer planning
  document — see [`installer_README.md`](../debug/installer/installer_README.md)
  for how far that plan actually got implemented (not very far).
- **`tarno-time-estimate-60e8e5.md`** / **`tarno-time-estimate-p0-p3-60e8e5.md`**
  / **`tarno-time-estimate-complete-60e8e5.md`**: three successive passes
  at time estimation (P0–P2, then P0–P3, then a "complete" refined
  version) — kept all three rather than overwriting, so the estimation
  history/reasoning isn't lost.

## Cross-references

- Finalized architecture decisions: [`workspace/debug/docs/adr/`](../debug/docs/adr/adr_README.md)
- Known issues these plans relate to: [`workspace/debug/docs/technical-debt-catalog.md`](../debug/docs/docs_README.md)
