# LIOM Toolkit

[![PyPI version](https://badge.fury.io/py/liom-toolkit.svg)](https://badge.fury.io/py/liom-toolkit) [![Build Status](https://github.com/LIOMLab/liom-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/LIOMLab/liom-toolkit/actions/workflows/ci.yml) [![Release](https://github.com/LIOMLab/liom-toolkit/actions/workflows/release.yml/badge.svg)](https://github.com/LIOMLab/liom-toolkit/actions/workflows/release.yml) [![Documentation Status](https://readthedocs.org/projects/liom-toolkit/badge/?version=latest)](https://liom-toolkit.readthedocs.io/en/latest/?badge=latest)

**LIOM Toolkit** is a Python package for processing and analyzing light-sheet
fluorescence microscopy (LSFM) data. It provides an end-to-end pipeline:

1. **Conversion** — HDF5 / NIfTI / NRRD volumes to the
   [OME-Zarr](https://ngff.openmicroscopy.org/) format, with multichannel
   support and out-of-core (Dask) processing.
2. **Registration** — rigid and SyN registration of mouse brains to the
   [Allen Brain Atlas](https://atlas.brain-map.org/) via
   [ANTsPy](https://github.com/ANTsX/ANTsPy), plus template building from a
   set of volumes.
3. **Segmentation** — classical vessel segmentation (2D Frangi + thresholding,
   3D SimpleITK watershed brain masking) and a PyTorch U-Net (`vseg`) trained
   on LSFM vasculature.
4. **Statistics** — per-region vessel morphometrics against the Allen
   structure tree, with Excel export.

The toolkit supports the
[Laboratoire d'Imagerie Optique et Moléculaire](https://liom.polymtl.ca/) at
Polytechnique Montréal and is published to
[PyPI](https://pypi.org/project/liom-toolkit/) so other neuroimaging labs can
`pip install liom-toolkit` and run the full pipeline without source edits or
hardcoded lab config.

## Installation

```bash
pip install liom-toolkit
```

For the heavy/optional dependencies, use the extras:

```bash
pip install "liom-toolkit[ai]"        # torch/timm/einops/wandb — for the vseg U-Net
pip install "liom-toolkit[antspy]"    # antspyx — for registration
pip install "liom-toolkit[all]"       # everything
```

The recommended way to work on the package itself is with
[`uv`](https://docs.astral.sh/uv/):

```bash
uv sync                    # core package + dev tools (pytest, ruff, sphinx)
uv sync --extra ai         # add the vseg deep-learning stack
uv sync --extra antspy     # add antspyx (registration)
uv sync --all-extras       # everything (heavy)
```

## Supported Python versions

LIOM Toolkit supports **Python 3.12 and 3.14** for the full package,
including registration. The core package (conversion, segmentation,
visualization, stats, utils, and the CLIs) works on both versions with
only the default dependencies.

The **registration** module (`liom_toolkit.registration`) depends on
[`antspyx`](https://github.com/ANTsX/ANTsPy) (ANTsPy). On Python 3.12 it
installs from PyPI (cp312 wheel). On Python 3.14 antspyx has no upstream
cp314 wheel, so this repo publishes prebuilt cp314 wheels via a GitHub
Actions workflow and consumes them through a uv flat index
(`[tool.uv.sources]` in `pyproject.toml`). The `antspy` extra works on
both versions.

## Command-line tools

LIOM Toolkit ships **7 `liom-*` console scripts** (registered in
`pyproject.toml` under `[project.scripts]` and documented in the
[CLI Reference](https://liom-toolkit.readthedocs.io/en/latest/cli.html)):

| CLI | Purpose |
|-----|---------|
| `liom-convert-hdf5-to-zarr` | Convert HDF5 volumes to OME-Zarr |
| `liom-create-mask` | Generate a brain mask from an OME-Zarr volume |
| `liom-segment-2d` | 2D Frangi + threshold vessel segmentation |
| `liom-align-annotations` | Register a volume to the Allen Atlas annotations |
| `liom-build-template` | Build a registration template from a set of volumes |
| `liom-compute-slice-metrics` | Per-region vessel morphometrics + Excel export |
| `liom-train-model` | Train the U-Net vessel-segmentation model |

The new CLIs share a parent parser with `--log-level`, `--resume`,
`--n_workers`, and `--dask_scheduler` flags. Run any CLI with `--help` for
the full argument list, e.g. `uv run liom-build-template --help`.

## Usage

The package is a **library** — import it from a notebook or another package.
A one-call logging helper is provided for notebook users who want visible
progress output (the library itself stays silent by default via the
`NullHandler` pattern):

```python
import liom_toolkit

liom_toolkit.configure_logging(level="INFO")
```

Demonstrations of the package's functionality can be found in the notebooks in
the LIOM Notebooks repository:
[LIOM Notebooks](https://github.com/LIOMLab/liom-notebooks).

## Package structure

```
liom_toolkit/
├── conversion/      HDF5/NIfTI/NRRD → OME-Zarr, multichannel, full pipeline
├── registration/    ANTsPy-based registration & template building
├── segmentation/    Classical + deep-learning vessel/brain segmentation
│   └── vseg/        PyTorch U-Net vessel segmentation (model, training, prediction)
├── visualization/   Slice / MIP extraction from OME-Zarr volumes
├── utils/           OME-Zarr IO, Dask client, ANTs bridge, Allen atlas, checkpointing
└── scripts/         The 7 `liom-*` CLI entry points
```

### Conversion

`liom_toolkit.conversion` — format conversion to OME-Zarr. Converts HDF5, NIfTI,
and NRRD volumes to multiscale OME-Zarr, builds multichannel and full volumes,
and writes via a streaming Dask-backed writer (`OmeZarrWriter`) that keeps data
out of RAM through the pipeline.

### Registration

`liom_toolkit.registration` — ANTsPy-based registration of mouse brains to the
Allen Atlas (rigid and SyN transforms, annotation alignment) and template
building from a set of volumes. **Requires the `antspy` extra and Python 3.12.**

### Segmentation

`liom_toolkit.segmentation` — vessel and brain segmentation. The classical path
covers 2D Frangi + threshold vessel segmentation (`segment_2d_image`) and 3D
SimpleITK watershed brain masking (`segment_3d`). The `vseg` submodule provides
a PyTorch U-Net for vasculature segmentation, with training, prediction
(`predict_one` / `predict_volume`), validation, and the cl-DICE topology
metric. **The U-Net requires the `ai` extra (torch/timm/einops/wandb).**

### Visualization

`liom_toolkit.visualization` — slice and maximum-intensity-projection
extraction from OME-Zarr volumes, with optional PNG/TIFF export.

### Utils

`liom_toolkit.utils` — cross-cutting utilities: OME-Zarr read/write
(`load_zarr`, `save_zarr`, `save_label_to_zarr`, `save_atlas_to_zarr`,
`extract_zarr_to_image`), the `DaskClientManager` singleton, an ANTs
bridge, Allen atlas/template download and reference-space construction, and
resume/checkpoint helpers (`ResumeManager`).

## Changelog

**1.0.0 is a clean break from the pre-1.0 date-based (calver) versions**
(`2025.*`). There are no deprecation shims — renames and removals are
one-way. If you are upgrading from a `0.x` / `2025.*` build, read
[`CHANGELOG.md`](./CHANGELOG.md) for the full breaking-change narrative and
migration notes (CustomScaler deletion, `allensdk` removal, the four hard
renames, `__all__` curation, the `calculate_density` /
`compute_average_diameter` semantic split, the TIFF-default
`extract_zarr_to_image` rename, and the wandb lab-config parameterization).

## Documentation

Full API documentation, a CLI reference, and a getting-started guide are
published on Read the Docs:
[liom-toolkit.readthedocs.io](https://liom-toolkit.readthedocs.io/).

## Requirements

The package requires **Python 3.12 or 3.14** (see
[Supported Python versions](#supported-python-versions) above; the package
declares `requires-python = ">=3.12"`). Core dependencies include numpy,
scikit-image, ome-zarr, zarr, h5py, pynrrd, nibabel, SimpleITK, dask,
opencv-python, tifffile, and pandas. The `ai` extra adds the deep-learning
stack (torch, timm, einops, wandb); the `antspy` extra adds antspyx for
registration.

On macOS, `h5py` needs the HDF5 system library — install it via Homebrew
before installing the package:

```bash
brew install hdf5
```

## License

LIOM Toolkit is released under the
[GPL-3.0-or-later](https://github.com/LIOMLab/liom-toolkit/blob/main/LICENSE)
license.
