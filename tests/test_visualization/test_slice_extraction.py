"""Regression tests for ``liom_toolkit/visualization/slice_extraction.py``.

These tests close the Wave-0 test gap for the slice-extraction module and
guard two locked Phase-6 decisions:

* The slice-range clobber fix (D-10): ``extract_slices_from_zarr`` must assign
  each extracted 2D slice to exactly one z-slot of ``full_volume`` via
  single-index assignment ``full_volume[i, :, :] = image`` — NOT the
  slice-range ``full_volume[i:, :, :] = image`` that clobbered every slice
  from ``i`` onward with the current iteration's data (the canonical AGENTS
  section 2 silent-data-corruption anti-pattern).
* The hard rename (D-03, slice-rename portion): the two slice-extraction
  functions previously carried a misspelled ``form_zarr`` suffix in their
  public names; they are renamed to the correct ``from_zarr`` spelling
  (``extract_slices_from_zarr`` and ``extract_and_save_slices_from_zarr``)
  with no deprecation shim (clean break at the 1.0.0 boundary).

Per AGENTS section 5, ``numpy`` / ``dask`` / ``zarr`` / ``ome_zarr`` are NOT
mocked — the regression test exercises the real array-indexing path on a
small synthetic OME-Zarr volume built via ``save_zarr`` into ``tmp_path`` and
read back with ``ome_zarr.reader`` so ``node.data[0]`` is a real dask-backed
array.
"""

from __future__ import annotations

import numpy as np

from liom_toolkit.conversion.conversion import save_zarr
from liom_toolkit.utils.io import load_zarr
from liom_toolkit.visualization import (
    extract_and_save_slices_from_zarr,
    extract_slices_from_zarr,
)


def _make_distinct_plane_volume(depth: int = 8, height: int = 8, width: int = 8) -> np.ndarray:
    """Build a 3D uint16 volume where each z-plane holds a distinct constant.

    Plane ``z`` is filled with value ``z + 1`` (so plane 0 is all-1s, plane 1
    is all-2s, ...). This makes the slice-clobber bug unambiguous: under the
    buggy ``full_volume[i:, :, :] = image`` assignment every output slot ends
    up holding the LAST plane's value; under the fix each slot holds its own
    plane's value.
    """
    vol = np.zeros((depth, height, width), dtype=np.uint16)
    for z in range(depth):
        vol[z, :, :] = z + 1
    return vol


def test_extract_slices_from_zarr_each_slice_distinct(tmp_path):
    """Each output slice holds the distinct data read from its own z index.

    Builds a real synthetic OME-Zarr volume (plane ``z`` = constant ``z+1``)
    via ``save_zarr`` into ``tmp_path``, reads it back as an
    ``ome_zarr.reader.Node``, and calls ``extract_slices_from_zarr`` with
    ``start_z=4, num_slices=4``. The ``linspace(2, 6, 5, dtype=int)`` call
    yields ``image_zs = [2, 3, 4, 5, 6]`` so ``full_volume`` has shape
    ``(5, 8, 8)`` and output slice ``i`` must equal the constant
    ``image_zs[i] + 1``.

    This assertion FAILS on the buggy ``full_volume[i:, :, :] = image`` code
    because the slice-range assignment overwrites every slot from ``i``
    onward with the current iteration's data, so all slots end up holding the
    last plane's value (``image_zs[-1] + 1 = 7``).
    """
    vol = _make_distinct_plane_volume(depth=8, height=8, width=8)
    zpath = str(tmp_path / "distinct.zarr")
    save_zarr(vol, zpath, scales=(6.5, 6.5, 6.5), chunks=(8, 8, 8))

    nodes = load_zarr(zpath)
    node = nodes[0]

    start_z = 4
    num_slices = 4
    full_volume = extract_slices_from_zarr(node, start_z=start_z, num_slices=num_slices)

    image_zs = np.linspace(
        start_z - num_slices / 2, start_z + num_slices / 2, num_slices + 1, dtype=int
    )
    # Shape: one slot per linspace sample, plus the H/W of the volume planes.
    assert full_volume.shape == (len(image_zs), 8, 8)

    # Each output slice must hold its OWN plane's distinct constant value —
    # no slice is a copy of the last-written slice.
    for i, z in enumerate(image_zs):
        expected_value = int(z) + 1
        assert np.all(
            full_volume[i, :, :] == expected_value
        ), (
            f"output slice {i} (z={z}) should hold constant {expected_value}, "
            f"got {full_volume[i, 0, 0]} — slice clobber regression"
        )

    # Sanity: the slices are actually distinct from each other (no two output
    # slots hold the same constant). This is the strongest form of the
    # distinct-slice assertion and fails unambiguously under the clobber bug
    # where every slot holds the last plane's value.
    slice_values = [int(full_volume[i, 0, 0]) for i in range(full_volume.shape[0])]
    assert len(set(slice_values)) == len(slice_values), (
        f"output slices are not distinct — clobber bug present, "
        f"slice values = {slice_values}"
    )


def test_extract_slices_from_zarr_name_exposed():
    """The renamed functions are re-exported from ``liom_toolkit.visualization``.

    Proves the D-03 hard rename landed and the star-import in
    ``liom_toolkit/visualization/__init__.py`` re-exports the new names. This
    FAILS at import time before the rename because the names do not exist.
    """
    # Both names must be callable objects re-exported via the package surface.
    assert callable(extract_slices_from_zarr)
    assert callable(extract_and_save_slices_from_zarr)
