"""Regression: every subpackage imports with ONLY core deps (no ants/torch/allensdk).

Run after every lazy-import move. Catches Pitfall 3 (lazy import not actually
deferred -- a transitive module-top import still pulls the heavy dep).

These tests have NO ``pytest.importorskip`` -- they MUST pass with core deps
only (that is the whole point of DEP-02). If any of these fails on a
core-only install, a module-top import somewhere in the package re-introduced
an eager optional-dep import.
"""


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
    import liom_toolkit.segmentation

    assert liom_toolkit.segmentation


def test_import_visualization():
    """liom_toolkit.visualization imports with core deps only."""
    import liom_toolkit.visualization

    assert liom_toolkit.visualization
