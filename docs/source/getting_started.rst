==============
Getting Started
==============

This page covers installing the LIOM Toolkit, choosing the right optional
extras for your workload, and running your first conversion or segmentation
from the command line.

Installation
============

The LIOM Toolkit is published on PyPI. Install it with pip into any
Python 3.12+ environment:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   pip install liom-toolkit

Verify the install and inspect the available command-line tools:

.. code-block:: bash

   python -c "import liom_toolkit; print('ok')"
   liom-convert-hdf5-to-zarr --help
   liom-create-mask --help

The package imports cleanly with only the core dependencies installed —
heavy/optional dependencies (``ants``, ``torch``, ``allensdk``) are extras
that degrade gracefully (see `Optional extras`_).

Conda environment
=================

If you prefer conda, create an environment and install the package from PyPI
inside it:

.. code-block:: bash

   conda create -n liom python=3.12
   conda activate liom
   pip install liom-toolkit

The toolkit itself is not on conda-forge; pip-installing into a conda
environment is the supported path. The heavy native dependencies
(``antspyx``, ``torch``) ship their own wheels and install cleanly under pip
inside a conda environment.

Optional extras
===============

The toolkit splits its dependencies into a small core (always installed)
and three optional extras for the heavy workloads. Install only what you
need:

.. code-block:: bash

   pip install liom-toolkit[ai]        # PyTorch U-Net vessel segmentation
   pip install liom-toolkit[antspy]    # ANTsPy brain registration to Allen Atlas
   pip install liom-toolkit[all]       # everything

``ai``
   Adds ``torch``, ``torchvision``, ``timm``, ``einops``, ``wandb``, and
   ``scikit-learn``. Required for the ``liom-train-model`` CLI and the
   ``liom_toolkit.segmentation.vseg`` subpackage (PyTorch U-Net vessel
   segmentation, training loop, prediction).

``antspy``
   Adds ``antspyx>=0.6.3``. Required for the ``liom-build-template`` and
   ``liom-align-annotations`` CLIs and the ``liom_toolkit.registration``
   subpackage (rigid/SyN registration, atlas alignment, template building).

   .. note::

      ``antspyx`` publishes wheels for CPython 3.12 only — there is no
      ``cp314`` wheel. If you run Python 3.14, install the toolkit into a
      3.12 environment (conda or pyenv) before adding the ``antspy`` extra.

``all``
   Convenience meta-extra that pulls in both ``ai`` and ``antspy``.

The package's optional-dependency guards re-raise with a user-facing message
naming the missing package and the affected module, so a missing extra
fails loudly with install instructions rather than a cryptic ``ImportError``.

Apple Silicon (hdf5)
====================

On Apple Silicon (macOS ARM), the ``h5py`` build may need the Homebrew HDF5
library. If a fresh ``pip install liom-toolkit`` fails while building
``h5py`` from source, install HDF5 via Homebrew and point the build at it:

.. code-block:: bash

   brew install hdf5
   export HDF5_DIR="$(brew --prefix hdf5)"
   pip install liom-toolkit

Pre-built ``h5py`` wheels for macOS ARM are usually available, so this step
is only needed when a wheel is missing for your Python version.

First run
=========

The command-line tools are the primary entry point for batch usage. The full
reference — auto-generated from each script's argparse definition — lives in
:doc:`cli`. Two common first runs:

.. code-block:: bash

   # Convert an HDF5 volume to OME-Zarr
   liom-convert-hdf5-to-zarr --input volume.h5 --output volume.ome.zarr

   # Build a 3D brain mask from an OME-Zarr volume
   liom-create-mask --input volume.ome.zarr --output mask.ome.zarr

For worked examples covering the full conversion → registration →
segmentation → stats pipeline, see the :doc:`notebooks/index`. For a prose
overview of how the pipeline stages fit together, see :doc:`usage`.
