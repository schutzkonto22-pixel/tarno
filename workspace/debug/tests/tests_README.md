# `workspace/debug/tests/` — automated test suite

The full `unittest`-based test suite, run in CI (see
[`.github/workflows/ci.yml`](../../../.github/workflows/ci.yml)) and via
[`workspace/scripts/verify.ps1`](../../scripts/scripts_README.md). ~55 files,
kept flat (one `test_<topic>.py` per subsystem/feature) rather than mirrored
into a nested tree matching `src/tarno_backend/`'s subpackages — with this
many independent feature areas, a flat namespace is faster to grep and
`unittest discover` doesn't care either way.

## Running

```
$env:PYTHONPATH = "src"; python -m unittest discover -s workspace/debug/tests
```

A single file: `python -m unittest workspace.debug.tests.test_core` (with
`PYTHONPATH=src` set) — running a single file directly, not through
`unittest discover`, is what actually executes assertions rather than just
import-checking them; see the note below.

## Naming convention

Test file names generally track the `src/tarno_backend/` module or feature
they cover (`test_command_engine.py` → `core/command_engine.py`,
`test_integrations.py` → `integrations/`, `test_vision_block7.py` →
`vision/`, named after the phased build plan's "Block 7" label — see
[`vision_README.md`](../../../src/tarno_backend/vision/vision_README.md)).
Not a strict 1:1 mapping — some files cover one narrow feature
(`test_language_swap.py`, `test_headless_std_streams.py`), others span a
whole subpackage (`test_memory.py`, `test_memory_block4.py`,
`test_memory_integration.py` together cover `memory/`).

## Optional/legacy dependency gating

A handful of tests depend on packages that are intentionally **not** in the
root `requirements.txt` (OVOS, PySide6, `pynput` — see that file's "Legacy /
optional feature deps" comment block and TD-005/TD-006/TD-007 in the
tech-debt catalog). Those tests skip themselves cleanly via
`unittest.SkipTest` / `@unittest.skipUnless(...)` when the dependency is
absent, rather than installing the dependency in CI just to make the test
run — see `test_core.py`, `test_command_engine.py`'s
`test_gui_mode_uses_qt_presenter`, and `test_integrations.py`'s
`test_push_to_talk` for the pattern to follow if you add a similarly-gated
test.

## Discovery vs. execution — a recurring gotcha

`unittest discover` only *imports* every test module; it does not catch
bugs that only surface when a test actually *runs* — `mock.patch("...")`
string targets and `Path(__file__).parent...`-based path computations
both resolve lazily, at call time. Several real bugs during this repo's
restructuring (stale `@patch()` targets, broken logger names, wrong
relative-path hop counts after a folder move) were only caught by running
the affected test files directly, not by discovery. When moving files
under `src/tarno_backend/` or `workspace/`, re-run the specific tests that
touch the moved code, not just `unittest discover`.

## Cross-references

- CI workflow: [`.github/workflows/ci.yml`](../../../.github/workflows/ci.yml)
- Manual/standalone debug scripts (not run in CI): [`workspace/debug/tools/`](../tools/tools_README.md)
- Centralized path resolver used by several tests: `tarno_backend/utils/paths.py`
  (see [`utils_README.md`](../../../src/tarno_backend/utils/utils_README.md))
- Test coverage summary: [`workspace/debug/docs/test-coverage-report.md`](../docs/docs_README.md)
