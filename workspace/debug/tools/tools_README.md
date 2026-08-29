# `workspace/debug/tools/` — standalone debug/diagnostic scripts

Manual, run-it-yourself scripts for interactively exercising a subsystem
against real backends (real API calls, real hardware) — not part of the
automated test suite (`workspace/debug/tests/`) and not run in CI. Run with
`PYTHONPATH=src` from the repo root, e.g.:

```
$env:PYTHONPATH = "src"; python workspace/debug/tools/debug_coding_agent.py
```

## Files

- **`debug_coding_agent.py`**: debug console for the native coding-agent
  backend (`tarno_backend/ai/coding/adapters/native_agent.py`) — runs a
  series of test tasks against a disposable workspace and prints every
  `CodingOutput` step live, without going through gRPC/UI.
- **`debug_proactive_voice.py`**: debug console for the proactive layer
  (`ProactiveEngine` + rephrasing + real speech output) using
  pre-scripted `ProactiveDraft` scenarios (see its `SCENARIOS`).
- **`debug_proactive_live.py`**: same proactive layer, but with the *real*
  observer classes and real system values (psutil) instead of scripted
  drafts — see the file's docstring for the exact contrast with
  `debug_proactive_voice.py`.
- **`vision_calibration.py`**: measures real end-to-end latency between a
  detected motion trigger and the parsed model response, against the real
  camera and real Mistral Vision API (no mock) — used to validate
  `VisionConfig` latency assumptions.
- **`soak_test.py`**: accelerated soak-test harness for TARNO's core loops.
  Not a pytest test — meant to run standalone for extended periods (up to
  the 24h the original masterplan called for) to catch slow memory leaks a
  short unit test can't.
- **`mesh_mock_sender.py`**: dev-only mock UDP sender for the Dynamic
  Hybrid Mesh feature — simulates Node A/B traffic matching
  `tarno_backend/integrations/mesh/payload.py`'s wire format, so the mesh
  plugin can be exercised end-to-end without real phone/ESP32 hardware.

## Cross-references

- Automated tests (CI-run): [`workspace/debug/tests/`](../tests/tests_README.md)
- Mesh integration these scripts exercise: `src/tarno_backend/integrations/mesh/`
  (see [`integrations_README.md`](../../../src/tarno_backend/integrations/integrations_README.md))
