# Contributing

The LIOM Toolkit is mid-modernization toward 1.0.0. Contributions that fix
bugs, improve correctness, add tests, or modernize the codebase are welcome.
This page covers the development environment, test suite, lint/type checks,
and the correctness rules every change to this package must respect.

## Development environment

The project uses [`uv`](https://docs.astral.sh/uv/) for environment and
dependency management. The `uv.lock` file pins the full resolved dependency
set; `.venv` is reconciled against it.

First-time setup, or after pulling:

```bash
uv sync --all-extras          # core + dev + all optional extras
```

Run anything through `uv run` so it executes in the reconciled `.venv`:

```bash
uv run python -c "import liom_toolkit; print('ok')"
uv run pytest
uv run liom-convert-hdf5-to-zarr --help
```

`uv run` reconciles the `.venv` against `uv.lock` before launching, so the
deps the command needs are guaranteed present. Prefer it over activating the
venv manually.

Add a runtime dependency with `uv add "<pkg>>=<lower>"` (updates
`pyproject.toml` `[project.dependencies]` and re-locks). Add a dev-only
dependency with `uv add --dev <pkg>` (writes to `[dependency-groups].dev`).
Add an optional/heavy dependency with `uv add --optional <extra> "<pkg>"`,
then gate its import behind a `try/except ImportError: raise ImportError(...)`
in the module that uses it. After changing dependencies, run `uv lock` then
`uv sync` and commit both `pyproject.toml` and `uv.lock` together.

## Running tests

```bash
uv run pytest                 # fast iteration (parallel via -n auto)
uv run pytest tests/ -q       # explicit path, quiet
uv run pytest -k "io"         # subset by keyword
uv run pytest --cov=liom_toolkit   # coverage
```

pytest config lives in `pyproject.toml` under `[tool.pytest.ini_options]`:
`testpaths = ["tests"]`, parallel via `pytest-xdist` (`-n auto
--dist=loadscope`). Only `test_*.py` files are collected.

The test layout mirrors the package (`tests/test_<subpkg>/test_<module>.py`).
Use plain `assert` (pytest style). Write to `tmp_path` for anything that
touches disk — never write into the repo tree.

**Established test patterns:**

- **Pure-logic** functions: direct import + call + assert on small synthetic
  arrays. No mocking, no I/O.
- **IO round-trip** (`save_zarr`→`load_zarr`): write a small real zarr into
  `tmp_path`, read it back, assert data equality + shape + dtype + axes
  metadata + pyramid level count.
- **Conversion correctness**: generate a tiny HDF5/NIfTI/NRRD on the fly in
  `tmp_path`, convert, assert.
- **Optional-dep gating**: `pytest.importorskip("torch")` for any `vseg/`
  test that touches the model; `pytest.importorskip("ants")` for
  registration tests. This mirrors the package's own `try/except
  ImportError` gates.

Do **not** write static-source tests (reading a `.py` file as text and
asserting on its string content). Test behavior by calling the function.

## Lint and type checking

```bash
uv run ruff check             # lint
uv run ruff format --check    # format check
uv run ty check               # type check (blocking in CI)
```

Per file:

```bash
uv run ruff check <changed_file>
uv run ty check <changed_file>
```

Fix any errors before considering the task done. `ty` is blocking in CI — a
nonzero exit fails the gate. Place as few `# type: ignore` comments as
possible; if you find yourself adding many, consider whether the code can be
refactored to be more compliant.

## Correctness rules

This package processes microscopy volumes where silent data corruption is
the dominant failure mode — a wrong axis order, a clobbered slice, or a
divide-by-zero returns a plausible-looking number that propagates into
published analysis. Correctness and explicit failure matter more than
convenience. Every change must respect these rules:

- **No silent data loss / wrong-data fallbacks.** A function that cannot
  produce a correct result must raise or return `None` — never return a
  zero-filled array, a `NaN`, or a plausible-shaped-but-wrong value. When
  you touch a path that could hit an empty/edge case, make failure
  explicit.
- **`assert` is not validation.** `assert` is stripped under `python -O`.
  For new code, use `if ...: raise ValueError(...)` for input validation.
  Do not add new `assert`-based validation.
- **No bare `except` / `except Exception` swallowing errors.** Catch the
  specific exceptions you handle. The optional-dependency
  `try/except ImportError: raise ImportError(...)` guards are the one
  sanctioned broad-ish catch, and they re-raise with a user-facing message.
- **Eager `.compute()` is a bug, not a feature.** Do not add new
  `.compute()` calls in the conversion/IO pipeline. Keep data as Dask
  arrays through the pipeline; materialize only at the boundaries that
  genuinely require it.
- **Do not "fix" a known bug by hiding it.** If a task touches a known bug,
  fix the root cause; do not paper over it with a broader `try/except` or
  a clamp that makes the symptom disappear. If a fix is out of scope, leave
  the bug and note it.

## Python version support

The package targets `requires-python = ">=3.12"` and is CI-tested on Python
3.12 and 3.14. Do not use syntax or stdlib features that require 3.13+ in
library code — 3.12 support is a hard requirement (some functionality is
integrated into a sibling project capped at 3.12). `from __future__ import
annotations` is acceptable for 3.12 compatibility where deferred annotation
evaluation is needed; on 3.14 it is a no-op.

## Versioning

Starting at 1.0.0 the package follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
The pre-1.0 releases used date-based calver (`2025.06.12`); 1.0.0 is a clean
break with no deprecation shims (see [`CHANGELOG.md`](https://github.com/LIOMLab/liom-toolkit/blob/main/CHANGELOG.md)).

| Bump | When | Backward-compatible? |
|------|------|----------------------|
| **1.0.x** (patch) | bugfixes, doc updates, dependency upper-bound widening | yes |
| **1.1** (minor) | new backward-compatible features, new optional extras | yes |
| **2.0** (major) | breaking API changes, removed public names, dropped Python versions | no |

**Releases are cut via git tags.** The version is derived from git tags at
build time by [`setuptools-scm`](https://setuptools-scm.readthedocs.io/)
(`dynamic = ["version"]` + `[tool.setuptools_scm]` in `pyproject.toml`).
To cut a release:

```bash
git tag v1.0.1
git push --tags
```

The `v*` tag push triggers the `release.yml` GitHub Actions workflow, which
builds the wheel/sdist, uploads to PyPI, and creates a GitHub Release with
notes extracted from the `## [<version>]` section of `CHANGELOG.md`. There
is no manual `pyproject.toml` version edit per release — the tag is the
single source of truth.

This gives the downstream `~/code/lightsheet` consumer a predictable
pinning strategy: `liom-toolkit>=1.0,<2` for stable-within-major,
`liom-toolkit~=1.0` for patch-only, or pin an exact tag for full
reproducibility.

## Optional dependencies

Heavy/optional deps (`ants`, `torch`, `wandb`) are extras — not
installed by default. The Allen atlas is downloaded directly (no extra).
Modules that need them must lazy-import so the package
imports cleanly with only core deps:

1. Wrap the import: `try: import ants except ImportError: raise
   ImportError("Please install ANTsPy to use the registration module of the
   LIOM toolkit.")`.
2. Re-raise with a user-facing message naming the missing package and the
   affected module — match the existing message wording style.
3. If the dep is only used inside one function, prefer a function-scope lazy
   import to break circular dependencies.
4. Add a `pytest.importorskip("<dep>")` to any test that exercises the
   module.

The top-level `liom_toolkit/__init__.py` is intentionally empty so importing
the package root pulls nothing heavy.
