"""Regression: every subpackage imports with ONLY core deps (no ants/torch/allensdk).

Run after every lazy-import move. Catches Pitfall 3 (lazy import not actually
deferred -- a transitive module-top import still pulls the heavy dep).

These tests have NO ``pytest.importorskip`` -- they MUST pass with core deps
only. If any of these fails on a core-only install, a module-top import
somewhere in the package re-introduced an eager optional-dep import.

The one exception is ``test_import_segmentation``, which uses
``pytest.importorskip("skimage")`` because segmentation genuinely requires
scikit-image/scipy/SimpleITK/cv2 at module scope (the honest ImportError
signal on an io-only install -- install ``liom-toolkit[seg]``).
"""

import importlib.util
import sys

import pytest


def test_import_liom_toolkit():
    """Top-level package imports with core deps only."""
    import liom_toolkit

    assert liom_toolkit


def test_import_utils():
    """liom_toolkit.utils imports with core deps only."""
    import liom_toolkit.utils

    assert liom_toolkit.utils


def test_import_conversion():
    """liom_toolkit.conversion imports with core deps only."""
    import liom_toolkit.conversion

    assert liom_toolkit.conversion


def test_import_registration():
    """liom_toolkit.registration imports with core deps only (no ants)."""
    import liom_toolkit.registration

    assert liom_toolkit.registration


def test_import_segmentation():
    """liom_toolkit.segmentation imports with core deps only (no torch)."""
    pytest.importorskip("skimage")
    import liom_toolkit.segmentation

    assert liom_toolkit.segmentation


def test_import_visualization():
    """liom_toolkit.visualization imports with core deps only."""
    import liom_toolkit.visualization

    assert liom_toolkit.visualization


# ---------------------------------------------------------------------------
# IO-only sentinel test (D-06)
#
# Masks the moved deps (skimage, simpleitk, scipy, cv2, pandas, requests) so
# any module-top eager import of them in the IO-clean subpackages raises
# ImportError. This runs on the existing --extra all CI legs at zero marginal
# cost and catches top-level eager-import regressions *even when the deps are
# installed*. The sentinel RAISES (not a MagicMock) -- a MagicMock sentinel
# gives a false green by returning a fake module instead of raising.
# ---------------------------------------------------------------------------

_MASKED_DEPS = ("skimage", "simpleitk", "scipy", "cv2", "pandas", "requests")


class _ImportRaisingLoader:
    """Loader that raises ImportError when the masked module is imported.

    Inserted into ``sys.meta_path`` by ``_SentinelFinder`` so that any
    ``import <masked_dep>`` at module top fires the finder, which returns a
    spec with this loader. Both ``create_module`` and ``exec_module`` raise
    so the ImportError surfaces regardless of which path the import system
    takes.
    """

    def create_module(self, spec):
        raise ImportError(f"sentinel: {spec.name} is masked (io-only test)")

    def exec_module(self, module):
        raise ImportError(f"sentinel: {module.__name__} is masked (io-only test)")


class _SentinelFinder:
    """Meta-path finder that returns an ImportRaisingLoader spec for masked deps.

    Only fires for the top-level package names in ``_MASKED_DEPS``. For any
    other name, returns ``None`` (falls through to the next finder). For
    submodules of a masked dep (e.g. ``skimage.measure``), the finder also
    fires because ``find_spec`` checks the top-level prefix.
    """

    def __init__(self, masked):
        self._masked = frozenset(masked)

    def find_spec(self, fullname, path=None, target=None):
        top = fullname.split(".")[0]
        if top in self._masked:
            return importlib.util.spec_from_loader(fullname, _ImportRaisingLoader())
        return None


@pytest.fixture
def io_only_sentinels():
    """Mask moved deps so module-top imports of them raise ImportError.

    Mirrors the conftest.py ``fake_ants`` / ``fake_torch`` restore-on-
    teardown pattern, but stricter: sentinels RAISE ``ImportError`` rather
    than returning a ``MagicMock``. The goal is to PROVE the IO-clean
    subpackages never eager-import the moved dep, not to fake its API.

    Teardown (the ``finally`` block) is critical -- without it, subsequent
    tests in the same xdist worker break with mysterious ImportError
    (Pitfall 4: cross-test leak). Each masked dep's original ``sys.modules``
    entry is saved on setup and restored (or popped if it was absent) on
    teardown.
    """
    saved = {name: sys.modules.get(name) for name in _MASKED_DEPS}
    finder = _SentinelFinder(_MASKED_DEPS)
    sys.meta_path.insert(0, finder)
    # Clear any already-imported masked entries so the finder fires on re-import.
    for name in _MASKED_DEPS:
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
        for name, orig in saved.items():
            if orig is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = orig


def test_io_only_imports_under_sentinel_masking(io_only_sentinels):
    """Under sentinel masking of moved deps, the IO-clean subpackages + entry points still import.

    Proves no eager top-import of skimage/simpleitk/scipy/cv2/pandas/requests
    leaks into the IO surface. The sentinel fixture inserts a meta-path
    finder that raises ImportError for those deps; if any IO-clean subpackage
    had a module-top ``import pandas`` (or similar), this test would fail
    with ImportError instead of passing.
    """
    import liom_toolkit
    import liom_toolkit.conversion
    import liom_toolkit.registration
    import liom_toolkit.utils
    import liom_toolkit.visualization
    from liom_toolkit.conversion import (
        convert_hdf5_to_zarr,
        convert_nifti_to_zarr,
        convert_nrrd_to_zarr,
        save_zarr,
    )

    assert liom_toolkit
    assert liom_toolkit.conversion
    assert liom_toolkit.registration
    assert liom_toolkit.utils
    assert liom_toolkit.visualization
    assert save_zarr
    assert convert_hdf5_to_zarr
    assert convert_nifti_to_zarr
    assert convert_nrrd_to_zarr
