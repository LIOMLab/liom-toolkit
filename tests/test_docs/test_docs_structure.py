"""Behavioral tests for the Phase 08.1 Sphinx documentation structure.

These tests verify the 5 DOCS requirements (DOCS-01..DOCS-05) using fast,
stdlib-only assertions that run in every CI leg — the docs dependency group
(sphinx, nbsphinx, pydata-sphinx-theme, ...) is NOT installed in CI test legs,
so these tests must NEVER import sphinx or run sphinx-build.

conf.py is parsed via ``ast`` (NOT imported — it imports sphinx at module
top). pyproject.toml via stdlib ``tomllib``. .readthedocs.yaml via
``pytest.importorskip("yaml")`` (pyyaml is transitively available but
importorskip is the sanctioned pattern per AGENTS.md §5). Notebooks are
parsed as JSON via stdlib ``json`` and their code-cell sources are joined
into a single string for substring presence/absence checks — this verifies
the committed notebook CODE matches the modernized package API, the
regression we want to guard against.

Repo root is resolved via ``Path(__file__).resolve().parents[2]``:
tests/test_docs/test_docs_structure.py -> parents[2] = repo root.
"""

from __future__ import annotations

import ast
import json
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_SOURCE = REPO_ROOT / "docs" / "source"
CONF_PY = DOCS_SOURCE / "conf.py"


# --- helpers ---------------------------------------------------------------


def _conf_assignments() -> dict[str, ast.expr]:
    """Parse conf.py with ast and return a name -> assigned-value-expr map.

    Only top-level ``name = <literal>`` assignments are collected. conf.py
    is NOT imported (it would import sphinx, which is not in CI test legs).
    """
    tree = ast.parse(CONF_PY.read_text(encoding="utf-8"))
    assigns: dict[str, ast.expr] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                assigns[target.id] = node.value
    return assigns


def _conf_value(name: str):
    """Return the literal value assigned to ``name`` in conf.py, or None."""
    expr = _conf_assignments().get(name)
    if expr is None:
        return None
    return ast.literal_eval(expr)


def _conf_text() -> str:
    return CONF_PY.read_text(encoding="utf-8")


def _notebook_code(nb_path: Path) -> str:
    """Join all code-cell source lines of a .ipynb into one string."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    parts: list[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            src = cell.get("source", "")
            if isinstance(src, list):
                src = "".join(src)
            parts.append(src)
    return "\n".join(parts)


# --- DOCS-01: conf.py rewrite ----------------------------------------------


class TestConfPy:
    """DOCS-01: conf.py uses pydata-sphinx-theme + curated 11-extension stack."""

    def test_html_theme_is_pydata_not_rtd(self):
        assert _conf_value("html_theme") == "pydata_sphinx_theme"
        assert "sphinx_rtd_theme" not in _conf_text()

    def test_extensions_has_exactly_11_named_extensions(self):
        expected = {
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
        }
        exts = _conf_value("extensions")
        assert exts is not None, "extensions list not found in conf.py"
        assert set(exts) == expected, f"extensions mismatch; got {set(exts)!r}"

    def test_autosummary_not_in_extensions(self):
        exts = _conf_value("extensions") or []
        assert "sphinx.ext.autosummary" not in exts

    def test_napoleon_numpy_docstring_true(self):
        assert _conf_value("napoleon_numpy_docstring") is True

    def test_nbsphinx_execute_is_never(self):
        # conf.py sets nbsphinx_execute = "never" globally so nbsphinx renders
        # the API-verified notebook code without re-executing on RTD (the
        # source notebooks carry no stored outputs and lab data paths are not
        # available in the build env). Reverting to "auto" would make nbsphinx
        # try to execute every notebook and fail the build on missing data.
        val = _conf_value("nbsphinx_execute")
        assert val == "never", f"nbsphinx_execute must be 'never', got {val!r}"

    def test_root_uses_parents_2(self):
        text = _conf_text()
        assert "parents[2]" in text, "conf.py ROOT must use parents[2]"

    def test_intersphinx_mapping_has_11_targets_including_ome_zarr(self):
        mapping = _conf_value("intersphinx_mapping")
        assert mapping is not None, "intersphinx_mapping not found"
        assert isinstance(mapping, dict)
        assert "ome-zarr" in mapping, "ome-zarr intersphinx target missing"
        assert len(mapping) == 11, f"expected 11 intersphinx targets, got {len(mapping)}"

    def test_no_sphinxcontrib_apidoc(self):
        assert "sphinxcontrib.apidoc" not in _conf_text()


# --- DOCS-02: autoapi switch -----------------------------------------------


class TestAutoapiSwitch:
    """DOCS-02: API reference switched from sphinxcontrib-apidoc to autoapi."""

    def test_gitignore_ignores_autoapi_output(self):
        gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        assert "docs/source/reference/autoapi/" in gitignore

    def test_old_modules_rst_stub_removed(self):
        assert not (DOCS_SOURCE / "reference" / "modules.rst").exists()

    def test_conf_autoapi_keep_files_true(self):
        assert _conf_value("autoapi_keep_files") is True

    def test_conf_autoapi_dirs_points_at_liom_toolkit(self):
        # autoapi_dirs = [str(ROOT / "liom_toolkit")] — the value is an
        # ast.Call (str(...)), not a literal, so ast.literal_eval cannot
        # evaluate it. Inspect the assigned expression's string form
        # instead: the requirement is that autoapi_dirs points at the
        # liom_toolkit package.
        expr = _conf_assignments().get("autoapi_dirs")
        assert expr is not None, "autoapi_dirs not found in conf.py"
        rendered = ast.unparse(expr)
        assert "liom_toolkit" in rendered, (
            f"autoapi_dirs must point at liom_toolkit, got {rendered!r}"
        )


# --- DOCS-03: RTD config + dep relocation ----------------------------------


class TestRtdConfigAndDeps:
    """DOCS-03: native uv RTD build + PEP 735 docs dep group relocation."""

    def test_readthedocs_yaml_valid_and_uv_native(self):
        yaml = pytest.importorskip("yaml")
        rtd = (REPO_ROOT / ".readthedocs.yaml").read_text(encoding="utf-8")
        doc = yaml.safe_load(rtd)
        # build.method == uv (under python.install[0].method)
        install = doc["python"]["install"][0]
        assert install["method"] == "uv"
        assert install["command"] == "sync"
        assert install["groups"] == ["docs"]
        # python tool version
        assert doc["build"]["tools"]["python"] == "3.14"

    def test_docs_requirements_txt_deleted(self):
        assert not (REPO_ROOT / "docs" / "requirements.txt").exists()

    def test_pyproject_has_docs_dependency_group(self):
        with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
            data = tomllib.load(fh)
        groups = data.get("dependency-groups", {})
        assert "docs" in groups, "[dependency-groups].docs missing"
        docs_deps = groups["docs"]
        joined = " ".join(docs_deps)
        assert "sphinx-rtd-theme" not in joined
        assert "sphinxcontrib-apidoc" not in joined

    def test_pyproject_docs_group_has_required_deps(self):
        with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
            data = tomllib.load(fh)
        docs_deps = data["dependency-groups"]["docs"]

        def _names(specs: list[str]) -> set[str]:
            names = set()
            for spec in specs:
                # strip version specifiers and extras
                base = spec.split(";")[0].split("[")[0]
                for i, ch in enumerate(base):
                    if ch in "=<>!~ ":
                        base = base[:i]
                        break
                names.add(base.strip().lower().replace("_", "-"))
            return names

        names = _names(docs_deps)
        for required in (
            "sphinx",
            "pydata-sphinx-theme",
            "sphinx-autoapi",
            "sphinx-argparse",
            "nbsphinx",
        ):
            assert required in names, f"{required} missing from docs dep group"


# --- DOCS-04: narrative pages + CLI reference ------------------------------


NARRATIVE_PAGES = (
    "index.rst",
    "getting_started.rst",
    "cli.rst",
    "usage.md",
    "contributing.md",
    "package_structure.rst",
)

CLI_SCRIPT_MODULES = (
    "liom_convert_hdf5_to_zarr",
    "liom_create_mask",
    "liom_segment_2d",
    "liom_align_annotations",
    "liom_build_template",
    "liom_compute_slice_metrics",
    "liom_train_model",
)


class TestNarrativePagesAndCli:
    """DOCS-04: 6 narrative pages + 7 sphinx-argparse CLI blocks."""

    def test_all_narrative_pages_exist(self):
        for name in NARRATIVE_PAGES:
            assert (DOCS_SOURCE / name).is_file(), f"{name} missing"

    def test_cli_rst_has_exactly_7_argparse_directives(self):
        text = (DOCS_SOURCE / "cli.rst").read_text(encoding="utf-8")
        assert text.count(".. argparse::") == 7

    def test_cli_rst_uses_full_build_argument_parser_name(self):
        text = (DOCS_SOURCE / "cli.rst").read_text(encoding="utf-8")
        assert "_build_argument_parser" in text
        assert "_build_arg_parser" not in text

    def test_all_cli_scripts_define_build_argument_parser(self):
        import importlib

        for mod_name in CLI_SCRIPT_MODULES:
            mod = importlib.import_module(f"liom_toolkit.scripts.{mod_name}")
            fn = getattr(mod, "_build_argument_parser", None)
            assert callable(fn), f"{mod_name}._build_argument_parser is not callable"

    def test_index_rst_uses_grid_cards(self):
        text = (DOCS_SOURCE / "index.rst").read_text(encoding="utf-8")
        assert "grid:: 1 2 2 2" in text
        assert "grid-item-card::" in text

    def test_index_rst_toctree_links_all_7_subpages(self):
        text = (DOCS_SOURCE / "index.rst").read_text(encoding="utf-8")
        for entry in (
            "getting_started",
            "cli",
            "usage",
            "contributing",
            "package_structure",
            "notebooks/index",
            "reference/autoapi/index",
        ):
            assert entry in text, f"toctree entry {entry!r} missing from index.rst"


# --- DOCS-05: notebooks ingestion + API fixes ------------------------------


NOTEBOOKS = (
    "zarr_conversion",
    "save_masks",
    "segment_vessels",
    "templating",
    "vessel_density",
    "vseg_full",
)


class TestNotebooks:
    """DOCS-05: 6 notebooks ingested with nbsphinx.execute=never + API fixes."""

    def test_notebooks_index_toctree_lists_all_6_notebooks(self):
        text = (DOCS_SOURCE / "notebooks" / "index.rst").read_text(encoding="utf-8")
        # Extract just the toctree block (the file also contains a code-block
        # example with a .ipynb path, so checking the whole file for .ipynb
        # absence would be wrong). The toctree runs from ".. toctree::" to EOF
        # or the next directive at column 0.
        toctree_start = text.index(".. toctree::")
        toctree_block = text[toctree_start:]
        for nb in NOTEBOOKS:
            assert nb in toctree_block, f"{nb} missing from notebooks/index.rst toctree"
        # toctree entries must NOT carry .ipynb extensions (nbsphinx resolves
        # the bare stem to the .ipynb file).
        assert ".ipynb" not in toctree_block

    def test_all_6_notebook_files_exist(self):
        for nb in NOTEBOOKS:
            assert (DOCS_SOURCE / "notebooks" / f"{nb}.ipynb").is_file(), f"{nb}.ipynb missing"

    def test_each_notebook_has_nbsphinx_execute_never(self):
        for nb in NOTEBOOKS:
            path = DOCS_SOURCE / "notebooks" / f"{nb}.ipynb"
            meta = json.loads(path.read_text(encoding="utf-8")).get("metadata", {})
            assert meta.get("nbsphinx", {}).get("execute") == "never", (
                f"{nb}.ipynb metadata.nbsphinx.execute != 'never'"
            )

    def test_zarr_conversion_no_use_memmap(self):
        code = _notebook_code(DOCS_SOURCE / "notebooks" / "zarr_conversion.ipynb")
        assert "use_memmap" not in code
        assert "use_mem_map" not in code

    def test_segment_vessels_uses_extract_zarr_to_image(self):
        code = _notebook_code(DOCS_SOURCE / "notebooks" / "segment_vessels.ipynb")
        assert "extract_zarr_to_image" in code
        assert "extract_zarr_to_png" not in code

    def test_vseg_full_uses_prediction_module_import(self):
        code = _notebook_code(DOCS_SOURCE / "notebooks" / "vseg_full.ipynb")
        assert "vseg.prediction import predict_one" in code
        assert "vseg.predict_one import predict_one" not in code

    def test_vseg_full_no_make_dataset_make_train_val(self):
        code = _notebook_code(DOCS_SOURCE / "notebooks" / "vseg_full.ipynb")
        assert "make_dataset import make_train_val" not in code

    def test_vseg_full_has_wandb_entity_and_pretrained_artifact(self):
        code = _notebook_code(DOCS_SOURCE / "notebooks" / "vseg_full.ipynb")
        assert "wandb_entity" in code
        assert "pretrained_artifact" in code

    def test_templating_has_atlas_resolution(self):
        code = _notebook_code(DOCS_SOURCE / "notebooks" / "templating.ipynb")
        assert "atlas_resolution" in code
