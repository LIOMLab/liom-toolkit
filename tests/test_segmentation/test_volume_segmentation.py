"""Numerical-equivalence regression tests for ``fill_holes_2d_3d``.

These tests gate the PERF-01a vectorization of ``fill_holes_2d_3d`` from the
inherited per-slice scipy ``binary_fill_holes`` cascade (one 3D pass + three
per-axis 2D slice passes + a final 3D pass = O(Z+Y+X) scipy calls) to a
single ``SimpleITK.BinaryFillhole(fullyConnected=True)`` call.

The reference (old) algorithm is inlined as ``_old_fill_holes_2d_3d`` so the
equivalence test is a true numerical-equivalence regression: it asserts
``array_equal`` (NOT ``allclose``) because hole-filling is a boolean topology
operation with no float intermediate — any divergence is a topology change,
not a rounding artifact.

Per AGENTS section 5, ``numpy``/``scipy``/``SimpleITK`` are NOT mocked -- the
tests exercise the real functions on small synthetic boolean volumes.
``volume_segmentation.py`` imports only core deps (numpy, scipy, SimpleITK),
so no ``pytest.importorskip`` gating is needed.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_fill_holes

from liom_toolkit.segmentation.volume_segmentation import fill_holes_2d_3d


def _old_fill_holes_2d_3d(mask: np.ndarray) -> np.ndarray:
    """The inherited per-slice scipy hole-filling cascade (reference algorithm).

    One 3D ``binary_fill_holes`` pass, then per-slice 2D passes along each of
    the three axes (X slices, Y slices, Z slices), then a final 3D pass. This
    is the O(Z+Y+X) implementation the vectorization replaces; inlined here
    verbatim so the equivalence test is a true regression, not a re-derivation.
    """
    mask = binary_fill_holes(mask)
    nx, ny, nz = mask.shape
    for x in range(nx):
        mask[x, :, :] = binary_fill_holes(mask[x, :, :])
    for y in range(ny):
        mask[:, y, :] = binary_fill_holes(mask[:, y, :])
    for z in range(nz):
        mask[:, :, z] = binary_fill_holes(mask[:, :, z])
    return binary_fill_holes(mask)


def test_fill_holes_equivalence():
    """The vectorized ``fill_holes_2d_3d`` must be ``array_equal`` to the old
    per-slice scipy cascade on a synthetic 3D volume with a known interior hole.

    Builds a 20x20x20 boolean volume that is True everywhere EXCEPT a 3x3x3
    interior hole at the center; both algorithms must fill the hole (return
    all True) AND produce identical boolean topology (``array_equal``).
    """
    vol = np.ones((20, 20, 20), dtype=bool)
    vol[8:11, 8:11, 8:11] = False  # 3x3x3 interior hole

    new_result = fill_holes_2d_3d(vol.copy())
    old_result = _old_fill_holes_2d_3d(vol.copy())

    # The hole must be filled by both.
    assert new_result.all(), "new fill_holes_2d_3d did not fill the interior hole"
    # Numerical equivalence: array_equal, NOT allclose (boolean topology).
    assert np.array_equal(new_result, old_result), (
        "fill_holes_2d_3d vectorization diverged from the old per-slice result"
    )


def test_fill_holes_empty_volume():
    """``fill_holes_2d_3d`` on an all-False volume returns all-False (no
    spurious holes filled — there is no foreground to enclose a hole)."""
    vol = np.zeros((12, 12, 12), dtype=bool)
    result = fill_holes_2d_3d(vol)
    assert not result.any()
    assert result.dtype == np.bool_


def test_fill_holes_full_volume():
    """``fill_holes_2d_3d`` on an all-True volume returns all-True (no holes
    to fill — the volume is already solid foreground)."""
    vol = np.ones((12, 12, 12), dtype=bool)
    result = fill_holes_2d_3d(vol)
    assert result.all()
    assert result.dtype == np.bool_


def test_fill_holes_no_hole():
    """``fill_holes_2d_3d`` on a volume with no interior holes returns
    ``array_equal`` to the input (a solid block with a missing corner is not
    an enclosed hole — it is open to the boundary)."""
    vol = np.ones((12, 12, 12), dtype=bool)
    vol[0, 0, 0] = False  # corner removed — open to boundary, not a hole
    result = fill_holes_2d_3d(vol.copy())
    assert np.array_equal(result, vol)


def test_fill_holes_returns_bool():
    """``fill_holes_2d_3d`` returns a boolean array (``np.bool_`` dtype) —
    the function signature promises ``NDArray[np.bool_]``."""
    vol = np.ones((8, 8, 8), dtype=bool)
    result = fill_holes_2d_3d(vol)
    assert result.dtype == np.bool_
