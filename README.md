# Liom Toolkit

This package supports the research being done by the Laboratoire d’Imagerie Optique et Moléculaire at
Polytechnique Montréal. It hosts a collection of scripts used to process and analyze data collected by the lab.

[![PyPI version](https://badge.fury.io/py/liom-toolkit.svg)](https://badge.fury.io/py/liom-toolkit) [![Build Status](https://github.com/LIOMLab/liom-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/LIOMLab/liom-toolkit/actions/workflows/ci.yml) [![Release](https://github.com/LIOMLab/liom-toolkit/actions/workflows/release.yml/badge.svg)](https://github.com/LIOMLab/liom-toolkit/actions/workflows/release.yml) [![Documentation Status](https://readthedocs.org/projects/liom-toolkit/badge/?version=latest)](https://liom-toolkit.readthedocs.io/en/latest/?badge=latest)

## Installation

The package can be installed using pip:

```bash
pip install liom-toolkit
```

Due to the complicated requirements, a detailed installation guide is provided below.

## Supported Python versions

LIOM Toolkit supports **Python 3.12 and 3.14** for the core package
(conversion, segmentation, visualization, stats, utils, and the CLIs).

The **registration** module (`liom_toolkit.registration`) depends on
[`antspyx`](https://github.com/ANTsX/ANTsPy) (ANTsPy), which is **supported on
Python 3.12 only**. antspyx does not currently publish `cp314` wheels, so on
Python 3.14 installing it requires a source build (ITK/VTK C++ compile) that
frequently fails. **Users who need registration should use Python 3.12.**

On Python 3.14:

- the `antspy` extra is not installed in CI and is expected to fail to install
  locally;
- registration tests are mocked (`-m "not antspy"` deselects the real
  `@pytest.mark.antspy` round-trip tests; the unmarked mock-orchestration tests
  still run);
- all other modules work as on 3.12.

3.14 support for registration is pending upstream `cp314` wheels from antspyx.

## Usage

Demonstrations of some of the functionalities of the package can be found in the notebooks in the LIOM Notebooks
repository.
The repository can be found here: [LIOM Notebooks](https://github.com/LIOMLab/liom-notebooks)

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

## Changelog

**1.0.0 is a clean break from the pre-1.0 date-based (calver) versions**
(`2025.*`). There are no deprecation shims — renames and removals are
one-way. If you are upgrading from a `0.x` / `2025.*` build, read
[`CHANGELOG.md`](./CHANGELOG.md) for the full breaking-change narrative and
migration notes (CustomScaler deletion, `allensdk` removal, the four hard
renames, `__all__` curation, the `calculate_density` /
`compute_average_diameter` semantic split, the TIFF-default
`extract_zarr_to_image` rename, and the wandb lab-config parameterization).

## Requirements

The package requires **Python 3.12 or 3.14** (see
[Supported Python versions](#supported-python-versions) above; the package
declares `requires-python = ">=3.12"`). The recommended way to install is
with [`uv`](https://docs.astral.sh/uv/):

```bash
# Core package + dev tools (pytest, sphinx)
uv sync                              # create/update .venv from uv.lock (core deps + dev group)
uv sync --extra ai                   # add torch/timm/einops/wandb (only if you need vseg)
uv sync --extra antspy               # add antspyx (only if you need registration; 3.12 only)
uv sync --all-extras                 # everything (heavy; use only when you need it all)
```

To create an anaconda environment instead, run the following commands:

```bash
conda create -n <name>
conda activate <name>
conda install python=3.12

# Install Pytorch at this point, follow the instructions on the Pytorch website:
# https://pytorch.org/get-started/locally/
# Make sure the right version is installed for your system. Check for CUDA compatibility.
# For example, for Linux with a CUDA compatible GPU:
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# For MacOS:
conda install pytorch::pytorch torchvision torchaudio -c pytorch

# The lines below are for Apple Silicon specifically.
# Hdf5 needs to be installed using homebrew (used by h5py).
brew install hdf5

# From now on pip will be used to install the packages. Some packages are not available on conda, or are out of date.
pip install antspyx                  # registration module (3.12 only)
pip install liom-toolkit

# To build the documentation of the package
pip install sphinx-rtd-theme
pip install sphinxcontrib-apidoc

# To run the LIOM Notebooks
pip install jupyter
```

## Package Structure

The package contains the following modules:

### Registration

The registration module is concerned with performing registration on brain imagery. It hosts a collection of scripts for
registering mouse brains to the Allen Atlas as well as functions for creating brain templates to use in registration.

### Segmentation

The segmentation module is concerned with segmenting brain imagery. It contains methods for segmenting brain images into
different regions of interest. The vseg submodule contains methods for segmenting vasculature using deep learning using
a U-net architecture. The pretrained model is trained on LSFM data.

### Utils

Various utility functions used by the other modules. These include function for converting between the different data
files used within the lab.
