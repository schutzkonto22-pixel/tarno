# `workspace/scripts/` — build & verify automation

PowerShell scripts for local build/verification workflows (not CI — see
[`.github/workflows/`](../../.github/workflows/) for that).

## Files

- **`verify.ps1`**: the "does everything still work" script referenced
  from [`CLAUDE.md`](../../CLAUDE.md)'s Build section — sequentially runs
  the WinUI build and the Python test suite. Handles `PYTHONPATH` itself
  (`Join-Path $root 'src'`), so it can be run from anywhere without manual
  env setup.
- **`build-installer.ps1`**: builds the Windows installer end-to-end
  (Python backend via PyInstaller + WinUI 3 frontend + NSIS packaging).
  Resolves `makensis.exe` dynamically (`Get-Command`, then
  `%ProgramFiles(x86)%\NSIS` / `%ProgramFiles%\NSIS`) instead of a
  hardcoded install path, per the project's no-hardcoded-paths rule.

## Usage

From the repo root:

```powershell
workspace/scripts/verify.ps1
workspace/scripts/build-installer.ps1
```

## Cross-references

- CI equivalent of `verify.ps1`: [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml)
- The (incomplete) CI-driven installer pipeline `build-installer.ps1`'s
  manual build overlaps with: [`workspace/debug/installer/`](../debug/installer/installer_README.md)
- Centralized path resolution these scripts' logic parallels (Python side):
  `tarno_backend/utils/paths.py` (see
  [`utils_README.md`](../../src/tarno_backend/utils/utils_README.md))
