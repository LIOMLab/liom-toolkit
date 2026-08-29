# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.2.0]

The CLI correctness-audit release. The `liom-*` command-line surface is
hardened before the vseg/DDP work adds an 8th CLI: every CLI flag is
normalized to the hyphenated form, a confirmed silent mis-assignment bug in
`liom-build-template` is fixed, and structural input validation (file
existence, enumerated choices, odd-integer checks) lands at the argparse
boundary so bad input produces a clear CLI error instead of a raw traceback
or a silently wrong result.

> **Semver deviation note:** this release ships **breaking** CLI changes
> under a **minor** version, deviating from strict semver (which reserves
> breaking changes for a major version bump). This is a deliberate choice
> following the project's established v1.0 "clean break, no deprecation
> shims" precedent: the vseg overhaul is already the planned v1.2 core
> release event, and a flag rename does not warrant pre-empting the 2.0
> major slot. The breaking changes are documented explicitly below so
> external PyPI users see the migration path.

### Changed

- **Breaking:** All CLI flags normalized to hyphenated form. The
  `dest` names argparse derives are unchanged (hyphens map to underscores),
  so `args.dask_scheduler`, `args.n_workers`, `args.resolution_level`,
  etc. read sites in every `main()` are unaffected — only the flag strings
  users type on the command line change. The 16 renamed flags:

  | Old flag | New flag | CLI |
  |----------|----------|-----|
  | `--dask_scheduler` | `--dask-scheduler` | shared (all 7 CLIs) |
  | `--n_workers` | `--n-workers` | shared (all 7 CLIs) |
  | `--resolution_level` | `--resolution-level` | liom-build-template |
  | `--template_resolution` | `--template-resolution` | liom-build-template |
  | `--atlas_resolution` | `--atlas-resolution` | liom-build-template |
  | `--voxel_size` | `--voxel-size` | liom-compute-slice-metrics |
  | `--fill_holes` | `--fill-holes` | liom-create-mask |
  | `--rigid_type` | `--rigid-type` | liom-align-annotations |
  | `--deformable_type` | `--deformable-type` | liom-align-annotations |
  | `--output_train` | `--output-train` | liom-train-model |
  | `--batch_size` | `--batch-size` | liom-train-model |
  | `--learning_rate` | `--learning-rate` | liom-train-model |
  | `--wandb_entity` | `--wandb-entity` | liom-train-model |
  | `--wandb_project` | `--wandb-project` | liom-train-model |
  | `--wandb_mode` | `--wandb-mode` | liom-train-model |
  | `--pretrained_artifact` | `--pretrained-artifact` | liom-train-model |

  The already-hyphenated flags (`--log-level`, `--resume`,
  `--frangi-sigma-range`, `--frangi-black-ridges`, `--local-threshold`,
  `--local-threshold-size`, `--scales`, `--chunks`, `--resolution`,
  `--iterations`, `--epochs`) are unchanged. The console-script entry-point
  names (`liom-convert-hdf5-to-zarr`, etc.) are NOT renamed — only the
  flags inside each CLI.

- **Breaking:** `liom-build-template` `zarr_files` / `brain_names`
  converted from consecutive `nargs="+"` positional arguments to
  repeatable `--zarr-files` / `--brain-names` options (`nargs="+"`,
  `required=True`). The previous positional form silently mis-assigned
  arguments when given more than one of each (e.g.
  `out.nii b1.zarr b2.zarr brain1 brain2` resolved to
  `zarr_files=['b1.zarr','b2.zarr','brain1']`,
  `brain_names=['brain2']`). The `build_template_for_resolution` domain
  signature is unchanged — `main()` still passes `zarr_files=` /
  `brain_names=` lists through. Old vs new invocation:

  ```bash
  # old (silently mis-assigns with >1 brain)
  liom-build-template out.nii b1.zarr b2.zarr brain1 brain2

  # new (unambiguous)
  liom-build-template out.nii --zarr-files b1.zarr b2.zarr \
      --brain-names brain1 brain2
  ```

- `--resolution` (liom-align-annotations) now enforces
  `choices=[10, 25, 50, 100]` (the help text already said "must be
  10/25/50/100" but no constraint was enforced; a wrong value silently
  produced a wrong atlas alignment).
- `--local-threshold-size` (liom-segment-2d) now validates the value is
  an odd integer (the help text already said "must be odd" but it was
  unvalidated; an even value flowed into Sauvola thresholding).
- Positional input-file paths across the CLIs that take file paths now
  gain file-existence checks at the argparse boundary, so a nonexistent
  path produces a clear `error: input file does not exist: <path>`
  (argparse exit 2) instead of a raw `imageio` / `h5py` /
  `ants.image_read` traceback.

### Migration

Replace every snake_case flag in your `liom-*` shell scripts and notebooks
with its hyphenated replacement from the table above. The `dest` names
argparse derives are unchanged, so any code reading `args.*` (e.g. notebooks
calling `_build_argument_parser().parse_args(...)`) is unaffected. For
`liom-build-template`, switch the positional `zarr_files brain_names` form
to the repeatable `--zarr-files ... --brain-names ...` options shown above.

## [1.1.0] - 2026-08-29

The lightweight-IO release. The core dependency set is slimmed to the
OME-Zarr saver/loader + HDF5/NIfTI/NRRD readers so a bare
`pip install liom-toolkit` is genuinely lightweight, and the heavy
segmentation / stats / atlas deps are split into optional install extras.
The full conversion→registration→segmentation→stats pipeline is preserved
via the new `[pipeline]` meta-extra.

### Added

- New optional `[io]` install extra: `pip install liom-toolkit[io]` pulls
  only the OME-Zarr saver/loader + HDF5/NIfTI/NRRD readers (the lightweight
  IO set: dask, distributed, bokeh, zarr, ome-zarr, h5py, nibabel, pynrrd,
  imageio, tifffile, natsort, tqdm). Core is now slimmed to this set, so
  `[io]` is a no-op extra documented for discoverability.
- New `[seg]` extra (classical segmentation deps: scikit-image, SimpleITK,
  scipy, opencv-python).
- New `[stats]` extra (morphometric stats deps: pandas, openpyxl, scipy).
- New `[pipeline]` meta-extra: `pip install liom-toolkit[pipeline]` pulls
  the full conversion→registration→segmentation→stats dep set in one line
  (io + seg + stats + ai + antspy).

### Changed

- **Breaking:** Core `[project.dependencies]` slimmed to the IO set
  (dask, distributed, bokeh, zarr, ome-zarr, h5py, nibabel, pynrrd,
  imageio, tifffile, natsort, tqdm). A bare `pip install liom-toolkit`
  now installs ONLY the IO set. To restore the prior full-pipeline
  behavior, add `[pipeline]` (or `[seg]`/`[stats]`/`[antspy]` as needed):
  `pip install liom-toolkit[pipeline]`.
- `pandas` and `requests` (used by `utils/allen_sdk.py` for Allen atlas
  download + structure tree) moved into the `[antspy]` extra; `pandas`
  is also in `[stats]`.
- `utils/allen_sdk.py` now lazy-imports `pandas` and `requests` inside the
  functions that use them, so `liom_toolkit.utils` imports cleanly on an
  IO-only install.
- `liom_toolkit.segmentation` now raises `ImportError` with a user-facing
  message ("Please install liom-toolkit[seg]...") instead of a bare
  `ModuleNotFoundError` when the segmentation deps are absent.

### Migration

If you relied on a bare `pip install liom-toolkit` for segmentation,
stats, or atlas work, switch to `pip install liom-toolkit[pipeline]`
(or the specific extra: `[seg]`, `[stats]`, `[antspy]`). The bare install
now provides only the IO/conversion surface.

## [1.0.0] - 2026-08-28

The first stable release. This is a **clean break** from the pre-1.0
date-based (calver) versions — there are no deprecation shims. Review the
**Breaking** entries below before upgrading from any `0.x` / `2025.*` build.
The modernization milestone (v0.5 → 1.0.0) spanned Phases 1–9; the breaking
changes are grouped by Keep a Changelog category and tagged with the phase
that introduced them.

### Added

- Five new `liom-*` console-script entry points: `liom-align-annotations`,
  `liom-build-template`, `liom-segment-2d`, `liom-compute-slice-metrics`,
  and `liom-train-model` (joining the existing `liom-convert-hdf5-to-zarr`
  and `liom-create-mask` for a total of seven CLIs).
- `liom_toolkit.configure_logging(level=...)` one-call logging helper for
  notebook users who hit the silent-defaults trap; library modules use the
  NullHandler pattern so importing the package never configures the root
  logger.
- Resume / checkpointing for `create_full_zarr_volume`,
  `build_template_for_resolution`, and `train_model` via a sidecar-JSON
  manifest + per-step `.done` markers + atomic complete sentinel, with
  stale-params-hash invalidation.
- `tifffile`, `openpyxl`, and `requests` declared as direct core
  dependencies (previously transitive via scikit-image / pandas / allensdk).
- `py.typed` marker shipped in the package for downstream type-checking.
- Dask `LocalCluster` worker count is now configurable with a sensible
  default cap (`min(cpu_count()-1, 8)`); `DaskClientManager` gained
  `close()` / context-manager (`__enter__` / `__exit__`) lifecycle and
  raises `ValueError` on `n_workers < 1`.
- Shared CLI args `--log-level`, `--resume`, `--n_workers`, and
  `--dask_scheduler` available across the new CLIs via a shared parent
  parser.

### Changed

- **Breaking:** Switched from date-based calver (`2025.06.12`) to semantic
  versioning. The version is now derived from git tags at build time via
  `setuptools-scm` (`dynamic = ["version"]` + `[tool.setuptools_scm]` in
  `pyproject.toml`); cut a release with `git tag v1.0.0`.
- **Breaking:** `save_zarr` / `save_label_to_zarr` / `save_atlas_to_zarr`
  migrated from the deprecated `CustomScaler` / `scaler=` /
  `coordinate_transformations=` API to
  `write_image(scale_factors=..., method=..., scale=...)` (the ome-zarr
  0.14+ modern path). `CustomScaler` is deleted (see Removed).
- **Breaking:** `generate_axes_dict()` now returns axis strings
  (`["z", "y", "x"]`) and takes an explicit `unit: str = "micrometer"`
  parameter (was a dict-with-`unit`). The default preserves existing
  callers and on-disk files; pass `unit="millimeter"` for the linumpy-style
  mm convention. `unit` is validated against NGFF UDUNITS-2 and raises
  `ValueError` (not `assert`) on unknown units.
- **Breaking:** `save_label_to_zarr` / `save_atlas_to_zarr` `resolution_level`
  now means "the input label's level" (was "CustomScaler input_layer").
  Low-res labels are upscaled to full-res via `resize(order=0)` using the
  main image's level-0 shape from the same `zarr_file`.
- **Breaking:** `convert_nifti_to_zarr` preserves the source dtype via
  `np.asanyarray(dataobj)` instead of upcasting to float64 via
  `get_fdata()`.
- **Breaking:** `predict_one` uses int-modulo tiling arithmetic and declares
  explicit `norm: bool = True` / `patching: bool = False` named parameters
  (was `**kwargs`). `patching=True` raises `NotImplementedError` pointing
  to `predict_volume`; `norm=False` skips CLAHE.
- **Breaking:** `extract_zarr_to_png` renamed to `extract_zarr_to_image`;
  the default output switched from PNG-per-slice to a multi-page TIFF.
  Pass `format="png"` to preserve the old PNG-per-slice behavior.
- **Breaking:** `calculate_density` returns `0.0` where the math is defined
  (vessel-free region) and `compute_average_diameter` raises `ValueError`
  on an empty mask (was a crash / `NaN` + `RuntimeWarning`). The two
  functions now have a deliberate semantic split: `0.0` where math is
  defined, `ValueError` where it is not.
- **Breaking:** `compute_slice_metrics` omits the mean-diameter row entirely
  for vessel-free regions (no `NaN` / blank / sentinel cell crosses the
  DataFrame boundary).
- **Breaking:** The public API surface was curated SciPy-style: 30 internal
  helpers were dropped from subpackage `__all__` and 36 public names
  retained. Underscore-prefix is the convention for internal helpers.
- **Breaking:** wandb config parameterized to `None` defaults (entity
  `"liom-lab"`, project `"vseg"`, pretrained artifact path removed) — the
  package is now lab-config-free on import. `VsegModel(pretrained=True,
  None)` raises `ValueError` (no silent fallback to a hardcoded lab
  artifact).
- **Breaking:** `ants.registration()` calls explicitly pass
  `use_legacy_histogram_matching=False` at the four public API entry points
  (`align_brain_region_to_atlas`, `align_annotations_to_volume`,
  `align_volume_to_allen`, `build_template_for_resolution`). antspyx 0.6.3
  flipped the default to OFF; the explicit kwarg immunizes the package
  against a future default flip.
- **Breaking:** `build_template` re-forked onto antspyx 0.6.3 upstream;
  keeps the fork's `masks=` parameter and per-iteration
  `remove_temp_output` cleanup, adopts upstream affine averaging +
  `useNoRigid`.
- **Breaking:** `print()` calls replaced with `logging.getLogger(__name__)`
  throughout the package; `tqdm` progress bars are kept.
- **Breaking:** Python floor raised to `>=3.12` (was `>=3.11`); CI-tested
  on 3.12 and 3.14.
- The PyPI `Development Status` classifier bumped from `4 - Beta` to
  `5 - Production/Stable`.

### Removed

- **Breaking:** `CustomScaler` class deleted (deprecated in ome-zarr
  0.14.0; the modern `write_image(scale_factors=..., method=...)` API
  replaces it).
- **Breaking:** `create_transformation_dict()` deleted (no longer needed
  after the `scale_factors` migration).
- **Breaking:** `allensdk` extra removed; `utils/allen_sdk.py` rewritten to
  direct NRRD download via `requests` + `nrrd.read` + `json.load` (no SDK
  dependency). `allensdk` was deprecated by the Allen Institute and broken
  on Python 3.12+ (pins numpy 1.23.5, needs removed `distutils`).
- **Breaking:** `patchify` removed from core dependencies; `create_patches`
  now uses `skimage.util.view_as_windows` (numpy-2-safe). `patchify` 0.2.3
  hard-pinned `numpy<2` and conflicted with the project's numpy 2.x.
- **Breaking:** `extract_slices_form_zarr` → `extract_slices_from_zarr`
  (typo rename, no deprecation shim).
- **Breaking:** `extract_and_save_slices_form_zarr` →
  `extract_and_save_slices_from_zarr` (typo rename, no deprecation shim).
- **Breaking:** `extract_zarr_to_png` → `extract_zarr_to_image` (rename,
  no deprecation shim — see Changed for the TIFF default).
- **Breaking:** Dead `use_memmap` / `map_file` code path and the
  `--use_memmap` CLI flag removed from `convert_hdf5_to_zarr`.
- **Breaking:** scikit-image 0.26 API renames applied: `skeletonize_3d` →
  `skeletonize`, `binary_erosion` → `erosion`,
  `remove_small_objects(min_size=)` → `max_size=`.
- **Breaking:** `skimage.io` replaced with `imageio.v3` across all modules.
- **Breaking:** `from zarr.convenience import open` → `zarr.open(...)`
  (zarr v3 API).
- **Breaking:** `assert`-based validation replaced with
  `if ...: raise ValueError(...)` throughout (survives `python -O`).
- **Breaking:** `PIL.Image.MAX_IMAGE_PIXELS = None` replaced with a finite
  `2_000_000_000` limit at `stats.py` and `vseg/utils.py` (preserves the
  decompression-bomb guard for untrusted inputs).
- **Breaking:** The blanket `filterwarnings = ["ignore::DeprecationWarning"]`
  in pytest config replaced with targeted message-scoped filters.
- **Breaking:** The broken `training.py` `__main__` block deleted (replaced
  by the `liom-train-model` CLI).

### Fixed

- `extract_slices_from_zarr` slice-range clobber: `full_volume[i:, :, :]`
  was overwriting every slice from `i` onward; now uses single-slot
  `full_volume[i, :, :]` assignment.
- `compute_average_diameter` empty-mask crash: now raises `ValueError`
  before `np.mean` instead of emitting `NaN` + `RuntimeWarning`.
- `calculate_density` divide-by-zero: returns `0.0` for vessel-free
  regions where the math is defined.
- `load_hdf5` file-handle leak: now uses `with h5py.File(...)` and reads
  each dataset to numpy before `da.from_array`.
- `save_zarr` mkdir race: uses the symlink-aware `create_directory`
  helper with `overwrite=True` at all write sites.
- `mktemp` TOCTOU insecurity in `build_template`: replaced with
  `tempfile.mkdtemp` (atomically creates a unique directory).
- `align_brain_region_to_atlas` None-reorient dead code: the
  `registration_volume is None` fallback now runs BEFORE
  `ants.reorient_image2` (was unreachable dead code after the reorient).
- `use_custom_atlas=False` broken stub in `convert_hdf5_to_zarr`: now
  downloads the Allen atlas at the matching resolution via
  `download_allen_atlas`.
- `predict_one` missing kwargs: `norm` and `patching` are now explicit
  named parameters (no more `**kwargs`).
- `chucks` → `chunks` typo in `convert_nifti_to_zarr` /
  `convert_nrrd_to_zarr`.
- `get_valid_indices` fork-mutation: now uses a sha256-metadata disk cache
  that does not mutate between forked workers.
- `create_heatmap` reset logic fixed.
