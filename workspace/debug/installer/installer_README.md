# `workspace/debug/installer/` — Windows installer build (⚠️ incomplete)

**This directory is mostly a gap, not a working pipeline.** Only
`version.nsh` actually exists here. `.github/workflows/release_installer.yml`
already references a full pipeline that was never built out:

- `workspace/debug/installer/tests/` (unittest discover target)
- `workspace/debug/installer/pipeline/build_all.py` (`python -m
  workspace.debug.installer.pipeline.build_all --edition ...`)
- `workspace/debug/installer/scripts/sign_installer.py`,
  `workspace/debug/installer/scripts/checksums.py`
- `workspace/debug/installer/dist/artifacts/` (output directory)

None of these exist yet. The `release_installer.yml` workflow (triggered
on `v*` tags or manual dispatch) will fail immediately at the "Run
installer tests" step until they're written. This isn't new breakage from
the workspace restructuring — the paths were kept consistent with the
planned structure for when the pipeline is actually implemented, not
invented by it.

## What does exist

- **`version.nsh`**: NSIS include file, presumably meant to be generated/
  updated by the (not-yet-written) build pipeline with the current version
  number for the installer.

## Related, working pieces (elsewhere)

- **[`workspace/scripts/build-installer.ps1`](../../scripts/scripts_README.md)**:
  a working local PowerShell build script — this is what actually builds
  an installer today, independent of the CI pipeline above.
- **[`workspace/plans/tarno-installer-plan-60e8e5.md`](../../plans/plans_README.md)**:
  the original installer planning document this gap traces back to.
- TD-010 in the tech-debt catalog (installer post-install validation) was
  resolved against `TARNO_Installer.nsi` — a file not currently tracked in
  this repo either; check with whoever owns the installer work before
  assuming it's been lost vs. never committed.

## If you're picking this up

Before writing `pipeline/build_all.py` etc. from scratch, check whether an
untracked local copy already exists on a contributor's machine (the
`release_installer.yml` workflow reads like it was written against a real,
working implementation) rather than reconstructing it blind.

## Cross-references

- CI workflow expecting this pipeline: [`.github/workflows/release_installer.yml`](../../../.github/workflows/release_installer.yml)
- Known issues: [`workspace/debug/docs/technical-debt-catalog.md`](../docs/docs_README.md) (TD-010)
