Example Notebooks
=================

The notebooks below demonstrate end-to-end usage of **LIOM Toolkit**.
They are committed **without stored outputs** (code-only) and are not
executed on ReadTheDocs — ``nbsphinx_execute = "never"`` is set globally
in ``conf.py`` so nbsphinx renders the API-verified source code as-is.
The notebooks reference lab data paths and heavy extras (``ants``,
``torch``) that are not available in the RTD build environment, so they
cannot be executed there. To see real outputs, run them locally with all
extras installed.

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
