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

from liom_toolkit.conversion.conversion import save_zarr
from liom_toolkit.utils.io import (
    create_transformation_dict,
    generate_axes_dict,
    generate_label_color_dict_mask,
    load_node_by_name,
    load_zarr,
    save_label_to_zarr,
)


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


def test_save_zarr_load_zarr_round_trip(tmp_path):
    """save_zarr -> load_zarr preserves data, shape, dtype, axes, transforms, levels.

    Uses the fresh-directory (non-overwrite) save_zarr path: ``os.mkdir`` creates
    the zarr directory, so the path must not exist beforehand. The CustomScaler
    default ``max_layer=4`` produces 5 pyramid levels (n_levels=5).
    """
    data = np.zeros((32, 32, 32), dtype=np.uint16)
    data[8:24, 8:24, 8:24] = 1000
    zpath = str(tmp_path / "vol.zarr")
    save_zarr(data, zpath, scales=(6.5, 6.5, 6.5), chunks=(32, 32, 32))

    nodes = load_zarr(zpath)
    img = nodes[0]

    # 5 pyramid levels (CustomScaler max_layer=4 -> levels 0..4)
    assert len(img.data) == 5
    # Level-0 data equality, shape, dtype
    assert np.array_equal(np.asarray(img.data[0]), data)
    assert img.data[0].shape == (32, 32, 32)
    assert img.data[0].dtype == np.uint16
    # Axes: ome-zarr 0.18 strips the 'unit' key on read; assert name/type only
    axes = img.metadata["axes"]
    assert [a["name"] for a in axes] == ["z", "y", "x"]
    assert [a["type"] for a in axes] == ["space", "space", "space"]
    # coordinateTransformations: one entry per pyramid level, each a scale
    ct = img.metadata["coordinateTransformations"]
    assert len(ct) == 5
    assert ct[0][0]["type"] == "scale"


def test_save_label_to_zarr_load_zarr_round_trip(tmp_path):
    """save_label_to_zarr -> load_zarr finds the label node by name with matching data.

    The image MUST be written via save_zarr FIRST: a label-only zarr has no
    ``ome`` multiscales metadata at the root, so the ome-zarr Reader returns 0
    nodes and ``load_node_by_name`` cannot find the label. Writing the image
    first gives the root group the multiscales metadata the Reader needs.
    """
    data = np.zeros((16, 16, 16), dtype=np.uint16)
    data[4:12, 4:12, 4:12] = 1000
    label = np.zeros((16, 16, 16), dtype=np.int8)
    label[4:12, 4:12, 4:12] = 1
    zpath = str(tmp_path / "vol.zarr")

    # Image first (per Pitfall 3): establishes root multiscales metadata.
    save_zarr(data, zpath, scales=(6.5, 6.5, 6.5), chunks=(16, 16, 16))
    save_label_to_zarr(
        label=label,
        zarr_file=zpath,
        color_dict=generate_label_color_dict_mask(),
        name="mask",
        scales=(6.5, 6.5, 6.5),
        chunks=(16, 16, 16),
        resolution_level=0,
    )

    nodes = load_zarr(zpath)
    mask_node = load_node_by_name(nodes, "mask")

    assert mask_node is not None
    assert len(mask_node.data) == 5
    assert np.array_equal(np.asarray(mask_node.data[0]), label)
    assert mask_node.data[0].dtype == np.int8


def test_generate_label_color_dict_mask_structure():
    """The mask color dict has 3 entries with label-values {0, 1, None}.

    The ``None`` label-value is invalid per the OME-NGFF label schema and
    triggers a non-fatal internal reader warning on load; this test
    characterizes the dict structure only and does not assert on the warning.
    """
    color_dict = generate_label_color_dict_mask()
    assert len(color_dict) == 3
    label_values = {entry["label-value"] for entry in color_dict}
    assert label_values == {0, 1, None}
    for entry in color_dict:
        assert "rgba" in entry
        assert len(entry["rgba"]) == 4
