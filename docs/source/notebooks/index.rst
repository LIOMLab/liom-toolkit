Example Notebooks
=================

The notebooks below demonstrate end-to-end usage of **LIOM Toolkit**.
Each notebook carries ``nbsphinx.execute = "never"`` metadata, so the
ReadTheDocs build renders the API-verified code without re-executing —
the notebooks reference lab data paths and heavy extras (``ants``,
``torch``) that are not available in the RTD build environment. To see
real outputs, run them locally with all extras installed.

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
