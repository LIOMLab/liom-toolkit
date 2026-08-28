Example Notebooks
=================

The notebooks below demonstrate end-to-end usage of **LIOM Toolkit**.
Lightweight notebooks (conversion, mask saving, vessel density) execute
live during the documentation build; heavy notebooks (templating, vessel
segmentation, full vseg pipeline) render their stored outputs because
they require the ``[ai]`` or ``[antspy]`` extras which cannot run on the
ReadTheDocs build environment.

To run them locally, install all extras:

.. code-block:: bash

   uv sync --all-extras
   uv run jupyter notebook docs/source/notebooks/zarr_conversion.ipynb

.. toctree::
   :maxdepth: 1

   zarr_conversion
   save_masks
   segment_vessels
   templating
   vessel_density
   vseg_full
