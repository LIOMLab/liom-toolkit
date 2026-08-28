# AGENTS.md — Instructions for AI agents working in this repo

This is an instruction set, not a reference dump. Read it before making changes.
The rules below exist because this package processes microscopy volumes where
silent data corruption is the dominant failure mode — a wrong axis order, a
clobbered slice, or a divide-by-zero returns a plausible-looking number that
propagates into published analysis. Correctness and explicit failure matter
more than convenience here.

## 1. What this project is

**LIOM Toolkit** — a Python package for processing and analyzing light-sheet
fluorescence microscopy (LSFM) data: format conversion (HDF5/NIfTI/NRRD →
OME-Zarr), brain registration to the Allen Atlas (ANTsPy), vessel segmentation
(classical + PyTorch U-Net), and morphometric statistics. It supports the
Laboratoire d'Imagerie Optique et Moléculaire at Polytechnique Montréal and is
published to PyPI for the broader neuroimaging community.

Core value: other labs can `pip install liom-toolkit` and run the full
conversion → registration → segmentation → stats pipeline without source edits
or hardcoded lab config. The package must import cleanly with only the core
dependencies installed; heavy/optional deps (`ants`, `torch`) are extras that
degrade gracefully.

This is a **library, not an application** — there is no GUI, no hardware, no
operator-in-the-loop. The CLIs (`liom-convert-hdf5-to-zarr`, `liom-create-mask`,
`liom-segment-2d`, `liom-align-annotations`, `liom-build-template`,
`liom-compute-slice-metrics`, `liom-train-model`) are thin wrappers over library
functions. Write code that is safe to call from a notebook or another library,
not just from the CLI.

The package is at 1.0.0. The public API surface is curated via explicit `__all__`
in each subpackage barrel; breaking changes should be deliberate and documented.

## 2. Correctness rules (read first, always)

There is no physical-safety surface here. The equivalent hard constraints are
about **not silently producing wrong data**:

- **No silent data loss / wrong-data fallbacks.** A function that cannot
  produce a correct result must raise or return `None` — never return a
  zero-filled array, a `NaN`, or a plausible-shaped-but-wrong value. When you
  touch a path that could hit an empty/edge case, make failure explicit. The
  canonical correct patterns: `compute_average_diameter` raises `ValueError` on
  empty masks before `np.mean` (no `NaN` + `RuntimeWarning` can escape);
  `predict_one` raises `ValueError` on all-zero input and `NotImplementedError`
  for `patching=True` (pointing to `predict_volume`) instead of silently
  returning plausible-shaped-but-wrong output.
- **`assert` is not validation.** `assert` is stripped under `python -O`. Use
  `if ...: raise ValueError(...)` for input validation, and include the
  offending value in the message so the error is actionable. There are zero
  `assert` statements in `liom_toolkit/` source — preserve that.
- **No bare `except` / `except Exception` swallowing errors.** The package has
  zero bare-except clauses. Catch the specific exceptions you handle. The
  optional-dependency `try/except ImportError: raise ImportError(...)` guards
  are the one sanctioned broad-ish catch, and they re-raise with a user-facing
  message. The `except BaseException` occurrences in `checkpoint.py`,
  `allen_sdk.py`, and `vseg/dataset.py` are the sanctioned
  cleanup-then-reraise pattern for atomic temp-file writes (unlink the
  `.partial` temp, then `raise`) — they never swallow.
- **Do not disable PIL's decompression-bomb guard.** `PIL.Image.MAX_IMAGE_PIXELS`
  is set to a high **finite** limit (`2_000_000_000`) in `stats.py` and
  `vseg/utils.py` — not `None`. Disabling it entirely is a DoS vector for
  untrusted inputs. Keep the finite limit. The test suite resets this global
  around every test via an autouse fixture in `tests/conftest.py`.
- **Eager `.compute()` is a bug, not a feature — except at genuine boundaries.**
  Keep data as Dask arrays through the pipeline; materialize only at boundaries
  that genuinely require a real array. Do not add new `.compute()` calls in the
  conversion/IO pipeline. Boundary-required `.compute()` calls (e.g. NIfTI
  writes via nibabel, SimpleITK needs a real array, the Dask→ANTsImage bridge,
  scalar materialization after `client.gather`) MUST carry a comment explaining
  why removing it would return the wrong type.

## 3. Execution environment

This is a library with no hardware surface — there is no rig, no deployment
target, no operator-in-the-loop. Development and CI run on ordinary
workstations and Linux runners.

| | Target environment |
|---|---|
| Python | `requires-python = ">=3.12"`; 3.12 is the primary/floor version, 3.14 is also CI-tested |
| Heavy deps | `ants`, `torch` are **extras** — not installed by default; lazy-imported |
| What runs | Pure-logic tests, IO round-trip tests on small synthetic volumes, lint/type checks, CLI smoke tests |

**Default assumption: only core deps are installed.** Everything you write
must import and run with `uv sync` (no extras). Code that needs `ants`/`torch`
must lazy-import it (see §9) so the module loads without the extra. Tests that
need an extra gate with `pytest.importorskip("ants")` / `"torch"`.

**Python version pinning.** The package floor is Python 3.12 (a hard
requirement — some downstream integrations are capped at 3.12). CI additionally
tests on 3.14. Do not use syntax or stdlib features that require 3.13+ in
library code. `from __future__ import annotations` is present in most modules
and is load-bearing for 3.12 forward-ref support (a no-op on 3.14) — do not
remove it. Ruff `target-version = "py312"` is load-bearing for the same reason.

**antsypy has no cp314 wheel.** Registration is fully testable on 3.12 (the CI
`3.12+all` leg installs antspyx and runs real registration tests). On 3.14 the
`3.14+all` leg is best-effort (`continue-on-error`) and installs only the `ai`
extras; registration tests there are mock-only. The lazy-import guard ensures
the package still imports without ants on 3.14.

## 4. uv — environment, lock, and dependency management

`uv` is the canonical environment and lock tooling. The `uv.lock` file pins
the full resolved dependency set; `.venv` is reconciled against it. Do not use
`pip` directly, do not hand-edit `uv.lock`, do not commit a `requirements.txt`
for the package (the one in `docs/` is for the Sphinx build only).

**Install uv** (if not already present — see
<https://docs.astral.sh/uv/getting-started/installation/> for all platforms):
```bash
brew install uv          # macOS
curl -LsSf https://astral.sh/uv/install.sh | sh   # Linux/macOS installer
```

**First-time setup / after pulling:**
```bash
uv sync --all-extras                 # create/update .venv from uv.lock (core + dev + all extras)
```

**Run anything through `uv run`** so it executes in the reconciled `.venv`:
```bash
uv run python -c "import liom_toolkit; print('ok')"
uv run pytest                        # (test suite — see §5)
uv run liom-convert-hdf5-to-zarr --help
```
`uv run` reconciles the `.venv` against `uv.lock` before launching, so the
deps the command needs are guaranteed present. Prefer it over activating the
venv manually.

**Add a runtime dependency:**
```bash
uv add "zarr>=3.0"
```
This updates `pyproject.toml` `[project.dependencies]` and re-locks. Prefer a
version published at least 7 days ago; avoid floating ranges (`latest`, `*`,
unbounded `>=`) that auto-resolve to brand-new releases. Pin a lower bound
that reflects the API you actually use, and prefer an upper bound for docs-only
deps.

**Add a dev-only dependency** (pytest, ruff, type stubs — not shipped to
users):
```bash
uv add --dev ruff
```
This writes to `[dependency-groups].dev` (PEP 735). uv installs the `dev`
group automatically with `uv sync`, so dev deps are present in the default
environment — there is no separate `uv sync --extra dev` step.

**Add an optional/heavy dependency** (a new extra, e.g. a new DL framework):
```bash
uv add --optional newdl "onnxruntime>=1.18"
```
Then gate its import behind `try/except ImportError: raise ImportError(...)`
in the module that uses it (see §8).

**Lockfile hygiene:**
- After changing dependencies, `uv lock` then `uv sync` and commit both
  `pyproject.toml` and `uv.lock` together. Never commit one without the
  other.
- `uv.lock` must resolve on both Python 3.12 and 3.14. If a dependency caps
  its Python upper bound, that is a blocker — do not work around it by pinning
  the package to a broken version; escalate.
- Do not add `--no-sync` / `--frozen` shortcuts to bypass lock reconciliation
  to "make it install" — if the lock is out of date, re-lock.

**Building the package (reproduces CI):**
```bash
uv build                             # produces dist/ sdist + wheel
```
The version is dynamic via setuptools-scm (derived from git tags; written to
`liom_toolkit/_version.py` at build time, which is gitignored). Release to
PyPI is tag-triggered via `.github/workflows/release.yml` — do not publish
manually unless explicitly asked.

## 5. Running tests, lint, and type checks

```bash
uv run pytest                         # full suite (parallel via xdist, 60% coverage gate)
uv run pytest tests/ -q               # explicit path, quiet
uv run pytest -k "io"                 # subset by keyword
uv run pytest -m "not ai"             # skip torch-gated tests
uv run pytest -m "not antspy"         # skip ants-gated tests
uv run ruff check                     # lint
uv run ruff format --check            # format check (read-only)
uv run ruff format                    # apply formatting
uv run ty check                       # type check
```

pytest config lives in `pyproject.toml` under `[tool.pytest.ini_options]`:
`testpaths = ["tests"]`, `--strict-markers`, pytest-xdist parallelism
(`-n auto --dist=loadscope`), and bare `--cov` (coverage source/report/gate
read from `[tool.coverage.*]`; `fail_under = 60`). CI overrides the gate to 0
on the 3.14 legs via `--cov-fail-under=0` because `importorskip`-gated tests
skip there. Do NOT add a zero-value `--cov-fail-under` in `addopts` —
pytest-cov uses the last value and `addopts` is prepended to the CLI, so a
zero would defeat the ratchet everywhere.

**Markers** (registered, `--strict-markers`): `@pytest.mark.ai` (requires the
`ai` extra / torch), `@pytest.mark.antspy` (requires antspyx; 3.12-only),
`@pytest.mark.slow`.

**Deprecation filters are targeted, not blanket.** The only
`filterwarnings` entry is a message-scoped filter for the pynrrd
`datetime.utcnow()` deprecation (unfixable upstream). Do not add new blanket
warning filters. When a third-party deprecation noises output, filter that
specific warning by message/module, not the whole class.

**Test layout** mirrors the package: `tests/test_<subpkg>/test_<module>.py`
plus cross-cutting tests at the `tests/` top level (`test_imports.py`,
`test_api_surface.py`, `test_package_metadata.py`, `test_pytest_config.py`).
New tests go in `tests/test_<subpkg>/test_<module>.py`. Use plain `assert`
(pytest style), not `self.assertEqual`. Write to `tmp_path` for anything that
touches disk — never write into the repo tree.

**Established test patterns:**
- **Pure-logic / known-answer** (`fix_even`, `calculate_metrics`, `cl_dice`,
  `erode_mask`, `generate_axes_dict`): direct import + call + assert on small
  synthetic arrays. No mocking, no I/O.
- **IO round-trip** (`save_zarr`→`load_zarr`, `save_label_to_zarr`→`load_zarr`,
  HDF5/NIfTI/NRRD conversion, TIFF/PNG extraction): write a small real file
  into `tmp_path`, read it back, assert data equality + shape + dtype + axes
  metadata + pyramid level count + NGFF v0.5 `ome.version` + anisotropic
  per-level `coordinateTransformations`.
- **Optional-dep gating**: `pytest.importorskip("torch")` for any `vseg/`
  test that touches `VsegModel`/`train_model`/`DiceLoss`/`predict_one`;
  `pytest.importorskip("ants")` for registration tests. Put `importorskip` at
  the **first line of the test body**, never at module top (pytest #9542 would
  skip the whole module including the mock tests). Gate inside fixtures too.
- **Mock heavy orchestration deps** (NOT compute deps): `patch("...ants")` /
  inject a MagicMock into `sys.modules` for lazy-imported `ants`; mock
  `dask_client_manager`. Do **not** mock `numpy`/`scipy`/`scikit-image` — test
  real image-processing functions with small synthetic arrays. Do **not** mock
  `zarr`/`h5py`/`nrrd`/`nibabel` — use `tmp_path` with real small files. Do
  **not** mock `torch` — use a real minimal `torch.nn.Module` stub model.
- **Config-as-data tests are permitted**: parse committed config files
  (`pyproject.toml` via `tomllib`, `.github/workflows/*.yml` via PyYAML) as
  structured data and assert on the parsed structure. This is NOT a
  static-source test.

**Do NOT write static-source tests** — reading a `.py` file as text and
asserting on its string/regex content. They are fragile and exercise no code.
Test behavior by calling the function.

Always run the suite after non-trivial changes. If you add a feature, add or
extend a `test_*.py` covering it. Every test file and every test should have a
docstring stating the behavior being asserted and the why.

## 6. Running the CLIs

Seven console scripts are registered in `pyproject.toml` under
`[project.scripts]`:
```bash
uv run liom-convert-hdf5-to-zarr --help
uv run liom-create-mask --help
uv run liom-segment-2d --help
uv run liom-align-annotations --help
uv run liom-build-template --help
uv run liom-compute-slice-metrics --help
uv run liom-train-model --help
```
Each resolves to a `main()` function in `liom_toolkit/scripts/liom_*.py`. Each
script module defines `_build_argument_parser()` (with
`parents=[build_common_parser()]`) and `main()`, and ends with
`if __name__ == "__main__": main()`. The shared parent parser in
`scripts/_common.py` provides `--dask_scheduler`, `--n_workers`, `--log-level`,
and `--resume`. The package is installed editable, so both forms work from any
CWD.

When adding a CLI, follow the existing pattern: create
`liom_toolkit/scripts/liom_<verb>_<object>.py` with
`_build_argument_parser()` + `main()`, register it in `[project.scripts]` as
`liom-<verb>-<object> = "liom_toolkit.scripts.liom_<verb>_<object>:main"`, call
`logging.basicConfig(...)` in `main()`, optionally wire `--dask_scheduler` to
`dask_client_manager.set_client(...)` before calling domain logic, and add a
test at `tests/test_scripts/test_liom_<verb>_<object>.py`.

## 7. Repo layout

```
liom_toolkit/                  importable package (importable as `liom_toolkit`)
  __init__.py                  library-side NullHandler + configure_logging export only
  _logging.py                  one-call logging setup for notebooks
  _version.py                  generated by setuptools-scm at build time (gitignored)
  conversion/                  format conversion to/from OME-Zarr
    __init__.py                explicit import list + __all__
    conversion.py              HDF5/NIfTI/NRRD -> Zarr, multichannel, full pipeline
  registration/                ANTs-based registration & template building
    __init__.py                explicit import list + __all__
    register.py                rigid/SyN registration, atlas/annotation alignment
    templating.py              template creation, pre-registration, build_template
  segmentation/                vessel/brain segmentation (classical + DL)
    __init__.py                explicit import list + __all__
    plane_segmentation.py      2D Frangi + threshold vessel segmentation
    volume_segmentation.py     3D SimpleITK watershed brain mask
    stats.py                   per-region vessel metrics, Allen region filtering
    vseg/                      PyTorch U-Net vessel segmentation subpackage
      __init__.py              exports predict_one, predict_volume + torch guard (__all__)
      model.py                 VsegModel U-Net (Conv/Encoder/Decoder blocks) — torch at top
      dataset.py               OmeZarrDataset / OmeZarrLabelDataSet
      training.py              train_model loop, W&B logging, checkpointing
      prediction.py            predict_one, predict_volume, do_predict
      validation.py            validate_model, metrics CSV, diff images
      loss.py                  DiceLoss, DiceBCELoss — torch at top
      cldice.py                cl_dice topology metric
      utils.py                 CLAHE, metrics, file sorting
  visualization/               slice/MIP extraction from OME-Zarr
    __init__.py                explicit import list + __all__
    slice_extraction.py        extract_single_slice, extract_slices, colour_image
  utils/                       cross-cutting utilities
    __init__.py                explicit import list + __all__
    io.py                      OME-Zarr read/write, masks, labels, atlas save
    zarr_writer.py             OmeZarrWriter, AnalysisOmeZarrWriter (streaming NGFF v0.5)
    dask_client.py             DaskClientManager singleton
    ants.py                    dask/zarr -> ANTsImage bridge (lazy ants)
    allen_sdk.py               Allen atlas/template HTTP download, ReferenceSpace
    checkpoint.py              ResumeManager + manifest/.done marker helpers
    utils.py                   fix_even, clean_dir, convert_to_png_for_saving
  scripts/                     CLI entry points (registered in pyproject.toml)
    __init__.py                docstring only (no exports)
    _common.py                 build_common_parser() — shared parent parser
    liom_convert_hdf5_to_zarr.py
    liom_create_mask.py
    liom_segment_2d.py
    liom_align_annotations.py
    liom_build_template.py
    liom_compute_slice_metrics.py
    liom_train_model.py
tests/                         pytest suite (mirrors package layout)
  conftest.py                  shared synthetic-volume fixtures
  test_<subpkg>/test_<module>.py
docs/                          Sphinx documentation (published to readthedocs)
  source/conf.py               sphinx-autoapi + autodoc config
.github/workflows/ci.yml       CI: lint (ruff+ty) + test matrix (3.12 gating, 3.14 best-effort)
.github/workflows/release.yml  tag-triggered PyPI publish + GitHub Release
.readthedocs.yaml              ReadTheDocs build config (native uv, Python 3.14)
pyproject.toml                 package metadata, deps, extras, console scripts, pytest, ruff, ty
uv.lock                        uv lockfile (pinned full resolved dep set)
```

The top-level `liom_toolkit/__init__.py` is deliberately minimal — importing
`liom_toolkit` alone pulls no heavy deps (only `configure_logging` and
`__version__`). Do not add top-level re-exports of domain symbols there; keep
the public surface per-subpackage.

## 8. Dependency structure (pyproject.toml)

Dependencies are declared as **optional-dependencies** (PEP 621) for extras and
**dependency-groups** (PEP 735) for dev/docs. Follow the existing form:

- `[project.dependencies]` — core, always installed: `tqdm`,
  `scikit-image>=0.26`, `imageio>=2.30`, `ome-zarr>=0.18.0`, `nibabel`,
  `zarr>=3.0`, `h5py`, `pynrrd`, `pandas`, `requests`, `PyWavelets`,
  `SimpleITK`, `dask`, `bokeh>=3.1.0`, `distributed`, `opencv-python`,
  `natsort`, `openpyxl>=3.1`, `tifffile>=2024.0`. Everything in
  `liom_toolkit/` must import with only these present.
- `[dependency-groups].dev` — `pytest>=8.0`, `pytest-cov>=7.1.0`,
  `pytest-xdist>=3.8.0`, `hypothesis`, `ruff>=0.16.4`, `ty==0.0.74`,
  `types-tqdm`, `types-opencv-python`, `scipy-stubs`, `pandas-stubs`,
  `pyarrow>=25.0.1`, `types-openpyxl`, `pyyaml>=6.0`. Installed automatically
  by `uv sync` (default group); also covered by `uv sync --all-extras`.
- `[dependency-groups].docs` — Sphinx + theme/extensions + notebook tooling,
  all bounded ranges. Docs-build-only (RTD installs this group via
  `.readthedocs.yaml`); NOT offered as a user-facing extra.
- `[project.optional-dependencies].ai` — `torch`, `torchvision`, `timm`,
  `einops`, `wandb`, `scikit-learn`. Required by `vseg/`.
- `[project.optional-dependencies].antspy` — `antspyx>=0.6.3`. Required by
  `registration/`.
- `[project.optional-dependencies].all` — convenience aggregate of
  `ai` + `antspy`.

`[project.scripts]` declares the seven console scripts (see §6). The package
ships a `py.typed` marker (`[tool.setuptools.package-data]`).

## 9. Optional-dependency import pattern — follow it for new heavy deps

Heavy/optional deps (`ants`, `torch`, `wandb`) are wrapped in
`try/except ImportError: raise ImportError("Please install X to use ...")` so
the package degrades gracefully. See `registration/__init__.py`,
`segmentation/vseg/__init__.py`, `utils/ants.py`, and the function-scope guards
in `register.py`, `templating.py`, `conversion.py`, `prediction.py`. The
top-level `liom_toolkit/__init__.py` is minimal so importing the package root
pulls nothing heavy.

When adding a module that needs an optional dep:
1. Wrap the import: `try: import ants except ImportError as e: raise
   ImportError("Please install ANTsPy to use the registration module of the
   LIOM toolkit.") from e`.
2. Re-raise with a user-facing message naming the missing package and the
   affected module — match the existing message wording style.
3. If the dep is only used inside one function, prefer a **function-scope
   lazy import** to break circular dependencies (e.g. `io.py` ↔ `zarr_writer.py`).
   Module-top guards are for deps the whole module needs; function-scope
   imports are for breaking cycles or for deps only one function touches.
4. Put heavy-dep *types* under `if TYPE_CHECKING:` so the module imports
   cleanly without the extra (e.g. `from ants.core.ants_image import ANTsImage`,
   `import torch`). Runtime imports stay at function scope.
5. Add a `pytest.importorskip("<dep>")` to any test that exercises the module.

Do **not** move a runtime import that a pydantic model field needs into
`TYPE_CHECKING` — that breaks validation. (No pydantic models exist here yet,
but the rule applies if any are introduced.)

## 10. Module / architecture conventions

- **Subpackage `__init__.py` files are barrels**: they re-export the public
  API with an explicit `from .module import name1, name2` list plus an
  explicit `__all__`. Import from the subpackage, not the module, in external
  code: `from liom_toolkit.utils import load_zarr` (not
  `from liom_toolkit.utils.io import load_zarr`). Internal cross-package
  imports also use the subpackage form: `from liom_toolkit.segmentation
  import segment_3d`. Keep `__all__` in sync when you add/remove a public
  name — `tests/test_api_surface.py` guards the curated surface against drift.
- **CLI-exposed functions are deliberately decoupled from `__all__`.**
  `compute_slice_metrics` and `train_model` are CLI entry points but are
  intentionally NOT in their subpackage `__all__` (the library API and the CLI
  contract are guarded separately). `tests/test_api_surface.py` asserts every
  `liom-*` console script resolves to a callable.
- **Singletons are module-level and documented**:
  `dask_client_manager = DaskClientManager()` in `utils/dask_client.py`.
  Access via `from liom_toolkit.utils.dask_client import dask_client_manager`.
- **Dask arrays flow through the pipeline; materialize only at boundaries.**
  `load_hdf5` returns a `da.Array`; conversion/IO functions should keep data
  as `da.Array` and write zarr from it, not `.compute()` into RAM first. The
  streaming writer (`zarr_writer.py`) writes NGFF v0.5 without materializing
  the full volume. Boundary-required `.compute()` calls must carry a comment
  (see §2).
- **Resume / checkpoint**: `utils/checkpoint.py` provides `ResumeManager` and
  manifest/`.done` marker helpers for resumable pipelines. The CLIs accept
  `--resume`. Atomic file writes use the write-to-`.partial`-then-`os.replace`
  pattern with `except BaseException: unlink(.partial); raise` cleanup.

## 11. Code style

- **Formatter/linter: ruff.** `uv run ruff format` (line-length 100,
  4-space indent, double quotes for string literals — the `Q` rule enforces
  this). `uv run ruff check` for lint. No black/flake8/mypy.
- **Type checker: ty.** `uv run ty check`. Rules in `pyproject.toml`
  `[tool.ty.rules]` — strictness based on ty's "even stricter" config;
  false-positive-prone rules are warnings so real bugs surface without failing
  CI on third-party stub imprecision. `replace-imports-with-any` maps untyped
  optional deps (`ants`, `torch`, `wandb`, `sklearn`, `patchify`,
  `matplotlib`, `allensdk`) to `Any` in the base env.
- **Type hints**: add to new function signatures; use PEP 604 union syntax
  (`str | None`, `int | float`, `da.Array | Future`) — NOT `Optional`/`Union`.
  Use generic builtins (`list[Node]`, `dict[str, Any]`, `tuple[int, int]`) —
  NOT `typing.List`/`Dict`/`Tuple`. Use `numpy.typing.NDArray[np.uint8]` and
  `ArrayLike` for array types.
- **Imports**: stdlib → third-party → local `liom_toolkit.*`, blank line
  between groups (enforced by `I`/isort via ruff). Absolute
  (`from liom_toolkit.utils import ...`) or relative within a package
  (`from .utils import ...`, `from ..segmentation import segment_3d`). No
  `sys.path` manipulation.
- **Naming**: `snake_case` modules/functions/vars; `PascalCase` classes;
  `UPPER_CASE` module constants; `_`-prefix for module-private helpers;
  `__`-dunder for true internals (`__create_local_cluster__`). CLI scripts
  prefixed `liom_` to match the `liom-` console-script names. Uppercase
  single-letter math names (`H`, `W`, `N`, `C`) are sanctioned in tensor/U-Net
  forward passes (`N806`/`N803`/`N812` ignores).
- **Docstrings**: numpy convention (enforced by `D`/`DOC` rules,
  `pydocstyle.convention = "numpy"`). Every public function and class has a
  docstring. Always include `:type` for every `:param` and `:rtype` for every
  `:return`. Class docstrings document constructor params the same way.
  Include a `Raises` section listing `ValueError`/`TypeError`/`ImportError`/
  `NotImplementedError` with conditions. Signature↔docstring consistency is
  enforced by `DOC` (pydoclint, preview). Do not drop the `:type`/`:rtype`
  lines when editing a docstring.
- **Logging**: stdlib `logging` throughout — module-level
  `logger = logging.getLogger(__name__)`. Use `logger.info`/`logger.debug`
  for stage announcements and diagnostics; keep `tqdm.auto.tqdm` for progress
  bars (those are appropriate). CLI `main()` configures the root logger via
  `logging.basicConfig(level=..., format="%(levelname)s %(name)s: %(message)s")`.
  No `print()` in package code (the `T20` hard-gate enforces this). For
  notebook/library consumers, `liom_toolkit.configure_logging(level="INFO")`
  attaches a `StreamHandler` to the `liom_toolkit` logger (idempotent).
- **Error handling**: no custom exception classes — errors propagate as
  library-native exceptions (`ants.*`, `torch.*`, `ValueError`, `TypeError`,
  `ImportError`, `NotImplementedError`). No `try/except` around core
  processing — let exceptions bubble; callers (CLI `main`, notebooks) handle
  top-level. The one sanctioned broad-ish catch is the optional-dep
  `try/except ImportError` (see §9), which re-raises. The `except BaseException`
  cleanup-then-reraise pattern for atomic writes is also sanctioned (see §2).
- **Memory**: side-effecting functions `del` large intermediates at the end
  (e.g. `del image, mask, frangi, vessel_mask_raw, vessel_mask, cleaned`).
  Free memory eagerly for volume-scale data — preserve this in functions you
  touch.
- **Function size**: long orchestration pipelines that thread a single `tqdm`
  through sequential stages are acceptable (`create_full_zarr_volume`,
  `align_brain_region_to_atlas`); do not split them artificially. Pure-logic
  functions should stay under ~50 lines. Complexity rules are suppressed in
  `pyproject.toml`.
- **Comments**: explain the "why" and "how" of the implementation, not the
  development process. Reference upstream issues / unfixable deprecations by
  name. Do not add `return`/`break`/`continue` inside `finally` blocks (PEP 765
  `SyntaxWarning` on 3.14).

## 12. Known anti-patterns to avoid repeating

- **Eager `.compute()` defeating Dask** — do not add new `.compute()` calls
  in the conversion/IO pipeline. Keep `da.Array` through the pipeline;
  materialize only at genuine boundaries with a justifying comment (see §2).
- **`assert` for input validation** — stripped under `-O`. Use
  `if ...: raise ValueError(...)` for new code (see §2).
- **Bare `except` / `except Exception`** — the package has zero; preserve
  that. Catch specific exceptions (see §2, §11).
- **Silent wrong-data fallbacks** — zero-filled arrays on failure, `NaN`
  from empty-mask stats, slice-range clobbering. Make failure explicit (see
  §2).
- **`sys.path.append(".")` / `sys.path` manipulation** — not used here; do
  not introduce it.
- **`tempfile.mktemp`** — insecure and deprecated. Use `tempfile.mkdtemp()`
  or `tempfile.NamedTemporaryFile(delete=False)` / `tempfile.mkstemp`.
- **Hardcoded lab config in library code** — no `/Users/`, `/home/`, `/mnt/`,
  `/data/`, lab org/project strings, or absolute paths in `liom_toolkit/`
  source. `train_model` takes `wandb_project`/`wandb_entity`/`wandb_mode` as
  parameters (default `None` → the user's wandb default); `VsegModel` takes
  `pretrained_artifact` as a parameter. Do not add new hardcoded lab
  org/project strings.
- **`PIL.Image.MAX_IMAGE_PIXELS = None`** — disables decompression-bomb
  protection. Use the finite limit `2_000_000_000` (see §2).
- **Mutable default args** — use `None` as a sentinel for "not specified"
  when the default is mutable or when `False`/`0` is a meaningful user value.
  Never use `list`/`dict` defaults.
- **Blanket deprecation filters** — filter specific warnings by
  message/module, not the whole `DeprecationWarning` class (see §5).
- **`__main__` blocks with empty-string placeholders** — if you add a
  `__main__` block, give it real `argparse` or guard it with a clear
  `raise SystemExit(...)` explaining the real entrypoint.

## 13. Before you finish a task

1. Run `uv run pytest`. Fix failures you caused. If you touched
   style-sensitive code, run `uv run ruff check`, `uv run ruff format --check`,
   and `uv run ty check`.
2. Confirm your change imports cleanly with **only core deps** installed
   (`uv sync` with no extras) — not just with `--all-extras`. A module that
   imports `ants`/`torch` at module top breaks the library for every user
   who did not install the extra.
3. Re-read §2 — confirm no correctness control was weakened (no new silent
   fallbacks, no new `assert`-validation, no new bare `except`, no new eager
   `.compute()` in the pipeline, no new hardcoded lab config, no
   `MAX_IMAGE_PIXELS = None`).
4. Commit `pyproject.toml` and `uv.lock` together when deps change; never
   one without the other.
5. Keep `__all__` in subpackage barrels in sync with the public names you
   add/remove.
