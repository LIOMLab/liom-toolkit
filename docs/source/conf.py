"""Sphinx configuration for the LIOM Toolkit documentation."""

import sys
from datetime import datetime
from pathlib import Path

# Make the project package importable for autoapi (static analysis) +
# sphinx-argparse (imports _build_argument_parser() at build time).
# parents[2] because conf.py is at docs/source/conf.py (source/ subdir).
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# -- Project information ----------------------------------------------------
project = "LIOM Toolkit"
author = "LIOM Toolkit Developers"
copyright = f"{datetime.now().year}, LIOM Toolkit Developers"

# Pull version from installed package metadata when available.
try:
    from importlib.metadata import PackageNotFoundError, version as _get_version

    release = _get_version("liom-toolkit")
except ImportError:
    # importlib.metadata unavailable (should not happen on 3.12+, but
    # keep the guard for safety).
    release = "0.0.0"
except PackageNotFoundError:
    # Package not installed (e.g. building docs from a source checkout
    # without `uv sync`). Fall back to a placeholder version.
    release = "0.0.0"
version = ".".join(release.split(".")[:2])

# -- General configuration --------------------------------------------------
# NOTE: sphinx.ext.autosummary is intentionally DROPPED — autoapi owns API
# generation per D-01. Running both would double-render members.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "autoapi.extension",
    "sphinxarg.ext",
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# MyST: render the existing Markdown docs.
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "linkify",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 4

# -- autoapi (D-02) ---------------------------------------------------------
# Generate the API reference from the liom_toolkit package via static AST
# analysis (no import — safe with lazy-loaded heavy optional deps).
autoapi_type = "python"
autoapi_dirs = [str(ROOT / "liom_toolkit")]
autoapi_root = "reference/autoapi"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autoapi_ignore = ["*/tests/*"]
autoapi_keep_files = True
autoapi_add_toctree_entry = True

# Autodoc settings.
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# -- napoleon (numpy docstrings from Phase 7) -------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_attr_annotations = True
# Render "Attributes:" docstring sections as :ivar: fields instead of
# emitting a separate :py:attribute: directive for each — avoids the
# "duplicate object description" warnings when autoapi also documents
# the same class attributes from their type annotations.
napoleon_use_ivar = True

# -- intersphinx (verified targets) -----------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "skimage": ("https://scikit-image.org/docs/stable/", None),
    "SimpleITK": ("https://simpleitk.readthedocs.io/en/master/", None),
    "zarr": ("https://zarr.readthedocs.io/en/stable/", None),
    "ome-zarr": ("https://ome-zarr.readthedocs.io/en/latest/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "dask": ("https://docs.dask.org/en/stable/", None),
    "PIL": ("https://pillow.readthedocs.io/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# Nitpicky mode is kept off. autoapi-extracted API pages contain many
# informal type names from numpy-style docstrings (``ndarray``, ``optional``,
# ``array-like``, ...) that are not real Python xrefs and cannot be resolved
# without rewriting every docstring. Run ``sphinx-build -n`` ad-hoc when
# reviewing API docstring quality.
nitpicky = False

# -- nbsphinx (D-04, linum-basic pattern) -----------------------------------
# The notebooks ship WITHOUT stored outputs (every code cell has
# execution_count = None and no outputs) and are rendered as code-only on
# ReadTheDocs. They reference lab data paths and heavy extras (``ants``,
# ``torch``) that are not available in the RTD build environment
# (antspyx has no cp314 wheel), so they cannot be executed there. Setting
# ``nbsphinx_execute = "never"`` globally guarantees nbsphinx renders the
# API-verified source code without attempting execution — even if a future
# notebook is added without per-notebook ``nbsphinx.execute = "never"``
# metadata. To see real outputs, run the notebooks locally with all extras
# installed (see ``notebooks/index.rst``).
nbsphinx_execute = "never"
nbsphinx_kernel_name = "python3"
nbsphinx_allow_errors = False
nbsphinx_timeout = 120
nbsphinx_widgets_path = ""  # CDN widget JS for tqdm.auto bars

# -- HTML output (pydata-sphinx-theme, D-01) --------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "LIOM Toolkit"
html_static_path = ["_static"]

html_theme_options = {
    "github_url": "https://github.com/LIOMLab/liom-toolkit",
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "show_prev_next": True,
    "header_links_before_dropdown": 4,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/LIOMLab/liom-toolkit",
            "icon": "fa-brands fa-github",
        },
    ],
    "navbar_align": "left",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
}

# Hide the right "On this page" sidebar on narrative pages; autoapi pages
# still show it. Lets prose pages use the full width.
html_theme_options["secondary_sidebar_items"] = {
    "**": ["page-toc"],
    "index": [],
    "getting_started": [],
    "cli": [],
    "usage": [],
    "contributing": [],
    "package_structure": [],
    "notebooks/index": [],
}

html_context = {
    "github_user": "LIOMLab",
    "github_repo": "liom-toolkit",
    "github_version": "main",
    "doc_path": "docs/source",
}

# -- suppress_warnings (lazy-import safety, D-02) ---------------------------
# autoapi uses static AST analysis and never imports modules, but it can't
# resolve references to lazy-imported optional deps (antspyx/torch). Silence
# the resulting import-resolution noise. ref.python and
# misc.highlighting_failure tolerate ambiguous xrefs from numpy docstrings.
suppress_warnings = [
    "autoapi.python_import_resolution",
    "ref.python",
    "misc.highlighting_failure",
]

# -- sphinx-copybutton ------------------------------------------------------
# Strip prompt characters from copied code blocks.
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = False
