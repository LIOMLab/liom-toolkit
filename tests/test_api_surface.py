"""Public API surface regression guard for the 6 subpackage ``__all__`` barrels.

The curated ``__all__`` in each subpackage ``__init__.py`` is the authoritative
public interface (SciPy API guide: ``__all__`` "authoritatively defines the
public interface"; "presence of underscores marks private, absence does NOT
mark public"). These tests lock that surface so future curation drift is
caught:

* every subpackage ``__init__.py`` defines ``__all__`` as a list of strings
* every name in ``__all__`` resolves to an importable attribute on the module
* no underscore-prefixed name survives in any ``__all__`` (a private name in
  the public surface is a contradiction)
* ``from <subpackage> import *`` exports exactly the names in ``__all__``

Parameterized over the 6 subpackages: conversion, registration, segmentation,
segmentation.vseg, utils, visualization. All 6 must import with core deps only
(no ants/torch/allensdk) -- these tests have no ``pytest.importorskip``.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import tomllib
from pathlib import Path

import pytest

# (dotted import path, subpackage short name for the test id)
SUBPACKAGES: list[tuple[str, str]] = [
    ("liom_toolkit.conversion", "conversion"),
    ("liom_toolkit.registration", "registration"),
    ("liom_toolkit.segmentation", "segmentation"),
    ("liom_toolkit.segmentation.vseg", "segmentation.vseg"),
    ("liom_toolkit.utils", "utils"),
    ("liom_toolkit.visualization", "visualization"),
]


def _load(dotted: str):
    """Import a subpackage and return the module object (core deps only)."""
    return importlib.import_module(dotted)


@pytest.mark.parametrize(("dotted", "name"), SUBPACKAGES, ids=[n for _, n in SUBPACKAGES])
def test_all_defined_in_every_init(dotted: str, name: str) -> None:
    """Every subpackage ``__init__.py`` defines ``__all__`` as a list of strings."""
    mod = _load(dotted)
    assert hasattr(mod, "__all__"), f"{dotted} must define __all__"
    all_list = mod.__all__
    assert isinstance(all_list, list), f"{dotted}.__all__ must be a list, got {type(all_list)}"
    assert len(all_list) > 0, f"{dotted}.__all__ must be non-empty (public surface)"
    for entry in all_list:
        assert isinstance(entry, str), (
            f"{dotted}.__all__ entries must be str, got {type(entry)}: {entry!r}"
        )


@pytest.mark.parametrize(("dotted", "name"), SUBPACKAGES, ids=[n for _, n in SUBPACKAGES])
def test_all_names_resolve(dotted: str, name: str) -> None:
    """Every name in ``__all__`` is an importable attribute on the subpackage."""
    mod = _load(dotted)
    for entry in mod.__all__:
        assert hasattr(mod, entry), (
            f"{dotted}.__all__ lists {entry!r} but getattr({dotted}, {entry!r}) "
            f"does not resolve -- the barrel import block is missing this name"
        )


@pytest.mark.parametrize(("dotted", "name"), SUBPACKAGES, ids=[n for _, n in SUBPACKAGES])
def test_no_underscore_in_all(dotted: str, name: str) -> None:
    """No underscore-prefixed name appears in any ``__all__``.

    A leading underscore marks a name private; re-exporting it in the public
    surface is a contradiction. Curation drops such names.
    """
    mod = _load(dotted)
    underscored = [entry for entry in mod.__all__ if entry.startswith("_")]
    assert not underscored, (
        f"{dotted}.__all__ contains underscore-prefixed (private) names: "
        f"{underscored} -- drop them from __all__"
    )


@pytest.mark.parametrize(("dotted", "name"), SUBPACKAGES, ids=[n for _, n in SUBPACKAGES])
def test_star_import_matches_all(dotted: str, name: str) -> None:
    """``from <subpackage> import *`` exports exactly the names in ``__all__``.

    With ``__all__`` defined, star-import brings in exactly those names (plus
    ``__builtins__`` injected by exec). Anything else is curation drift.
    """
    mod = _load(dotted)
    namespace: dict[str, object] = {}
    exec(f"from {dotted} import *", namespace)  # ruff: ignore[exec-builtin] -- star-import is the behavior under test
    # exec injects __builtins__; everything else must come from __all__.
    exported = {k for k in namespace if k != "__builtins__"}
    expected = set(mod.__all__)
    assert exported == expected, (
        f"{dotted}: star-import exports {sorted(exported)} but __all__ is "
        f"{sorted(expected)} -- mismatch (extra: {sorted(exported - expected)}, "
        f"missing: {sorted(expected - exported)})"
    )


# ---------------------------------------------------------------------------
# CLI entry-point resolution guard (CLOSE-02)
# ---------------------------------------------------------------------------

# The expected console-script roster is DERIVED from pyproject.toml
# [project.scripts] (config-as-data, AGENTS section 5) so the test can never
# lag the registry: adding a script to pyproject without a working main fails
# here, and removing one drops the expectation automatically. The
# entry_points guard protects the CLI contract WITHOUT growing the curated
# library __all__ surface -- the script targets are deliberately NOT
# re-exported by any subpackage __init__ (the entry_points guard and the
# __all__ guard are decoupled by design).

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _registered_liom_scripts() -> dict[str, str]:
    """Return the ``liom-*`` console scripts registered in pyproject.toml.

    Parses ``[project.scripts]`` via tomllib -- the pyproject table is the
    authoritative roster, so the expected set can never drift from the
    registered set.
    """
    scripts = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "scripts"
    ]
    return {name: value for name, value in scripts.items() if name.startswith("liom-")}


def test_installed_console_scripts_match_pyproject_roster() -> None:
    """Installed console_scripts match the pyproject ``liom-*`` roster exactly.

    Reads installed-package metadata via ``importlib.metadata.entry_points``
    and asserts the ``liom-*`` set equals the parsed ``[project.scripts]``
    set -- a missing entry point means the editable install was not
    refreshed after a pyproject edit, and an extra one means the registry
    drifted from the declared table.
    """
    expected = _registered_liom_scripts()
    eps = importlib.metadata.entry_points(group="console_scripts")
    found = {ep.name for ep in eps if ep.name.startswith("liom-")}
    assert found == set(expected), (
        f"console_scripts registry drift: pyproject declares "
        f"{sorted(expected)} but the installed package exposes "
        f"{sorted(found)} -- symmetric difference: "
        f"{sorted(set(expected) ^ found)} (re-run `uv sync` after editing "
        "[project.scripts])"
    )


def test_every_registered_script_resolves_to_callable_main() -> None:
    """Every pyproject ``liom-*`` script's ``module:attr`` resolves to a callable main.

    The convention (AGENTS section 6) is ``liom-<name> =
    "liom_toolkit.scripts.<module>:main"`` -- a script registered in
    pyproject without a working ``main`` fails here. Module imports stay
    core-safe by convention: each script lazy-imports heavy deps inside
    ``main()``, so resolving ``main`` never pulls torch/ants.
    """
    for name, target in _registered_liom_scripts().items():
        module_name, sep, attr = target.partition(":")
        assert sep and attr == "main", (
            f"{name} = {target!r} does not follow the '<module>:main' entry-point convention"
        )
        module = importlib.import_module(module_name)
        func = getattr(module, attr, None)
        assert callable(func), (
            f"{name} -> {target} is not callable (resolved to {type(func).__name__})"
        )


def test_every_registered_script_is_documented_in_cli_rst() -> None:
    """Each registered ``liom-*`` script name appears literally in docs/source/cli.rst.

    The CLI reference must cover the full registered roster -- a script
    added to pyproject without a cli.rst section fails here, so the docs
    can never lag the registry.
    """
    rst_text = (_REPO_ROOT / "docs" / "source" / "cli.rst").read_text(encoding="utf-8")
    missing = sorted(name for name in _registered_liom_scripts() if name not in rst_text)
    assert not missing, (
        f"cli.rst does not document registered scripts: {missing} -- add a "
        "section per script (the registry, the surface test, and the user "
        "docs all derive from pyproject.toml [project.scripts])"
    )
