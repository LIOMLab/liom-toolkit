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
from collections.abc import Callable

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

# The 7 console scripts registered in pyproject.toml [project.scripts]. The
# entry_points test guards the CLI contract (every registered name resolves to
# an importable callable) WITHOUT growing the curated library __all__ surface
# -- compute_slice_metrics and train_model are deliberately NOT re-exported by
# any subpackage __init__ (D-02 locked decision: the entry_points test and the
# __all__ guard are decoupled by design).
EXPECTED_SCRIPTS: set[str] = {
    "liom-align-annotations",
    "liom-build-template",
    "liom-compute-slice-metrics",
    "liom-convert-hdf5-to-zarr",
    "liom-create-mask",
    "liom-segment-2d",
    "liom-train-model",
}


def test_all_console_scripts_resolve_to_callables() -> None:
    """Every registered liom console script resolves to an importable callable.

    Reads installed-package metadata via ``importlib.metadata.entry_points``
    (the editable install exposes the [project.scripts] entries), filters to
    the 7 EXPECTED_SCRIPTS, asserts the set matches exactly (no curation
    drift -- a missing or extra entry point is a CLI contract violation), and
    asserts each ``ep.load()`` returns a callable. Runs on core-only CI: the
    metadata is read from the installed package, not from the optional deps.
    """
    eps = importlib.metadata.entry_points(group="console_scripts")
    liom_eps = {ep for ep in eps if ep.name in EXPECTED_SCRIPTS}
    found = {ep.name for ep in liom_eps}
    assert found == EXPECTED_SCRIPTS, (
        f"console_scripts curation drift: expected {sorted(EXPECTED_SCRIPTS)} "
        f"but found {sorted(found)} -- symmetric difference: "
        f"{sorted(EXPECTED_SCRIPTS ^ found)}"
    )
    for ep in liom_eps:
        func: Callable = ep.load()
        assert callable(func), (
            f"{ep.name} -> {ep.value} is not callable (ep.load() returned {type(func).__name__})"
        )
