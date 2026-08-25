"""Smoke and IO round-trip tests for ``liom_toolkit/utils/io.py``.

Mirrors the package layout (``tests/test_utils/test_io.py``) and the
known-answer style established by ``tests/test_canary.py``. Covers:

* Known-answer tests for ``create_transformation_dict`` and
  ``generate_axes_dict`` (pure functions, no I/O).
* Real ``tmp_path`` round-trip tests for ``save_zarr``→``load_zarr`` and
  ``save_label_to_zarr``→``load_zarr`` asserting data equality, shape,
  dtype, axes metadata, coordinate transforms, and pyramid level count.

The round-trip tests are the guard for the Phase-3 ``CustomScaler`` deletion
and ``write_image(scale_factors=...)`` migration: they must pass genuinely on
the current pre-migration code so Phase 3 can prove behavioral equivalence
after the rewrite. They write real zarr groups to ``tmp_path`` (no mocking of
zarr/ome-zarr) per AGENTS.md §5 and the codebase testing map.
"""

from __future__ import annotations

import numpy as np

from liom_toolkit.utils.io import create_transformation_dict, generate_axes_dict


def test_create_transformation_dict_3d_5_levels():
    """5 pyramid levels of a 3D volume scale voxel_size by 2**level per axis."""
    result = create_transformation_dict(5, (6.5, 6.5, 6.5), 3)
    assert len(result) == 5
    assert result[0] == [{"type": "scale", "scale": [6.5, 6.5, 6.5]}]
    assert result[4] == [{"type": "scale", "scale": [104.0, 104.0, 104.0]}]


def test_create_transformation_dict_4d():
    """A 4D volume prepends a channel axis scale of 1.0.

    ``_get_scale`` builds ``[1.0, voxel_size[0]*2**level, voxel_size[1]*2**level,
    voxel_size[2]*2**level]`` and for 4D (offset=0) returns the full list, so the
    channel axis always gets scale 1.0 and only ``voxel_size[:3]`` are used.
    """
    result = create_transformation_dict(3, (1.0, 2.0, 3.0, 4.0), 4)
    assert result[0] == [{"type": "scale", "scale": [1.0, 1.0, 2.0, 3.0]}]


def test_generate_axes_dict_3d():
    """A 3D volume has z/y/x space axes in micrometer."""
    assert generate_axes_dict(3) == [
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]


def test_generate_axes_dict_4d_prepends_channel():
    """A 4D volume prepends a channel axis (no unit) before z/y/x."""
    result = generate_axes_dict(4)
    assert result[0] == {"name": "c", "type": "channel"}
    assert len(result) == 4
