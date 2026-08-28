============
Liom Toolkit
============

**LIOM Toolkit** is a Python package for processing and analyzing light-sheet
fluorescence microscopy (LSFM) data. It supports the `Laboratoire d'Imagerie
Optique et Moléculaire <https://liom.ca/>`_ at Polytechnique Montréal and is
published to PyPI for the broader neuroimaging community.

The toolkit covers the full LSFM workflow: format conversion
(HDF5/NIfTI/NRRD → OME-Zarr), brain registration to the Allen Atlas (ANTsPy),
vessel segmentation (classical + PyTorch U-Net), and morphometric statistics.
Other labs can ``pip install liom-toolkit`` and run the complete
conversion → registration → segmentation → stats pipeline without source
edits or hardcoded lab config.

.. image:: https://github.com/LIOMLab/liom-toolkit/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/LIOMLab/liom-toolkit/actions/workflows/ci.yml
   :alt: Build Status

.. image:: https://readthedocs.org/projects/liom-toolkit/badge/?version=latest
   :target: https://liom-toolkit.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Getting Started
      :link: getting_started
      :link-type: doc

      Installation, first run, and the optional ``ai`` / ``antspy`` / ``all``
      extras for deep learning and atlas registration.

   .. grid-item-card:: CLI Reference
      :link: cli
      :link-type: doc

      Auto-generated reference for the 7 ``liom-*`` command-line tools,
      rendered from each script's ``_build_argument_parser()``.

   .. grid-item-card:: Library API
      :link: reference/autoapi/index
      :link-type: doc

      Auto-generated reference for the ``liom_toolkit`` Python package,
      produced by sphinx-autoapi from static AST analysis.

   .. grid-item-card:: Example Notebooks
      :link: notebooks/index
      :link-type: doc

      Worked examples: conversion, segmentation, templating, and stats —
      rendered with stored outputs.

   .. grid-item-card:: Usage
      :link: usage
      :link-type: doc

      Prose overview of the conversion → registration → segmentation →
      stats pipeline and how the pieces fit together.

   .. grid-item-card:: Contributing
      :link: contributing
      :link-type: doc

      Dev environment, tests, lint/typing, and the correctness rules every
      change to this package must respect.

.. toctree::
   :maxdepth: 1
   :hidden:

   getting_started
   cli
   usage
   contributing
   package_structure
   notebooks/index
   reference/autoapi/index
   changelog


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
