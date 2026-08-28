=================
Package Structure
=================

The ``liom_toolkit`` package is organised into five domain subpackages plus a
``scripts/`` directory of thin CLI wrappers. Each subpackage re-exports its
public API through its ``__init__.py`` (barrel imports), so external code
imports from the subpackage, not the module: ``from liom_toolkit.utils import
load_zarr`` rather than ``from liom_toolkit.utils.io import load_zarr``.

The top-level ``liom_toolkit/__init__.py`` is intentionally empty — importing
the package root pulls nothing heavy. Heavy/optional dependencies
(``ants``, ``torch``, ``allensdk``) are extras, lazy-imported at module top
or inside the function that needs them (see :doc:`contributing`).

conversion/
===========

Format conversion to and from OME-Zarr, the package's canonical storage
format. ``conversion.py`` provides ``load_hdf5``, ``convert_hdf5_to_zarr``,
``convert_nifti_to_zarr``, ``convert_nrrd_to_zarr``,
``create_multichannel_zarr``, and ``create_full_zarr_volume``. Source data
flows through as Dask arrays and is materialised only at the zarr write
boundary — eager ``.compute()`` calls that pull full volumes into RAM are
treated as bugs, not features.

registration/
=============

ANTs-based image registration to templates and the Allen Atlas, plus
template generation. ``register.py`` implements rigid and SyN registration
and atlas alignment; ``templating.py`` implements template creation,
pre-registration, and ``build_template``. Importing this subpackage eagerly
requires the ``antspy`` extra (the import guard re-raises with a user-facing
install message if ``ants`` is missing).

segmentation/
=============

Vessel and brain segmentation, plus quantitative metrics. Three classical
modules — ``plane_segmentation.py`` (2D Frangi + threshold vessel
segmentation), ``volume_segmentation.py`` (3D SimpleITK watershed brain
mask), and ``stats.py`` (per-region vessel metrics, Allen region filtering)
— plus the ``vseg/`` subpackage for PyTorch U-Net vessel segmentation.
``vseg/`` contains the model (``model.py``), dataset loaders (``dataset.py``),
training loop (``training.py``), prediction (``prediction.py``), validation
and metrics (``validation.py``), losses (``loss.py``), the ``cl_dice``
topology metric (``cldice.py``), and utilities (``utils.py``). The ``vseg``
subpackage guards its ``torch`` import and re-raises with an install message
if the ``ai`` extra is missing.

visualization/
==============

Slice and maximum-intensity-projection extraction from OME-Zarr volumes.
``slice_extraction.py`` provides ``extract_single_slice``, ``extract_slices``,
and ``colour_image``. This subpackage has no heavy optional dependencies.

utils/
======

Cross-cutting utilities shared across the domain subpackages:
``io.py`` (OME-Zarr read/write, multiscale scaling, mask and label
persistence — the largest module in the package), ``dask_client.py`` (the
``DaskClientManager`` singleton), ``ants.py`` (a Dask/zarr → ANTsImage
bridge), ``allen_sdk.py`` (Allen atlas/template download and
``ReferenceSpace``), and ``utils.py`` (``fix_even``, ``clean_dir``,
``convert_to_png_for_saving``). The ``dask_client_manager`` singleton is
module-level and accessed via ``from liom_toolkit.utils.dask_client import
dask_client_manager``.

scripts/
========

Seven thin CLI wrappers that expose domain functions as console scripts,
registered in ``pyproject.toml`` under ``[project.scripts]``. Each script
module defines ``_build_argument_parser()`` (which builds the argparse tree
without importing heavy deps) and ``main()`` (which lazy-imports the heavy
deps and wires an optional ``--dask_scheduler`` argument to
``dask_client_manager.set_client(...)`` before calling domain logic). The
seven CLIs are documented in :doc:`cli` via sphinx-argparse, which auto-
generates the reference from each script's ``_build_argument_parser()`` —
eliminating CLI-argument drift between the docs and the code.
