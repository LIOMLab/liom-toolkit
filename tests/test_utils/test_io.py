"""Smoke and IO round-trip tests for ``liom_toolkit/utils/io.py``.

Mirrors the package layout (``tests/test_utils/test_io.py``) and the
known-answer style established by ``tests/test_canary.py``. Covers:

* Known-answer tests for ``generate_axes_dict`` (returns NGFF v0.5 full
  dict-form axes — ``[{"name":"z","type":"space","unit":...}, ...]`` with a
  channel axis prepended for 4D) and the ``validate_n_levels`` helper that
  clamps the requested pyramid level count to what the downsampled axes
  can actually support.
* Real ``tmp_path`` round-trip tests for ``save_zarr``→``load_zarr`` and
  ``save_label_to_zarr``→``load_zarr`` asserting data equality, shape,
  dtype, axes metadata, NGFF v0.5 ``ome.version`` metadata, anisotropic
  per-level ``coordinateTransformations`` (Z stays at base scale while
  Y/X grow cumulatively), and pyramid level count.

The round-trip tests are the guard for the ``CustomScaler`` deletion and
``write_image(scale_factors=..., method=..., scale=..., scaler=None)``
migration: they assert the new ome-zarr 0.18 NGFF v0.5 writer path produces
correct on-disk metadata. They write real zarr groups to ``tmp_path`` (no
mocking of zarr/ome-zarr) per AGENTS.md §5 and the codebase testing map.
"""

from __future__ import annotations

import numpy as np
import zarr

from liom_toolkit.conversion.conversion import save_zarr
from liom_toolkit.utils.io import (
    generate_axes_dict,
    generate_label_color_dict_mask,
    load_node_by_name,
    load_zarr,
    save_label_to_zarr,
    validate_n_levels,
)


def test_generate_axes_dict_3d():
    """A 3D volume has NGFF dict-form z/y/x axes with unit on each spatial axis."""
    assert generate_axes_dict(3) == [
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]


def test_generate_axes_dict_4d_prepends_channel():
    """A 4D volume prepends a channel axis (no unit key) before z/y/x."""
    assert generate_axes_dict(4) == [
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]


def test_generate_axes_dict_unit_param():
    """The ``unit`` param lands on each spatial axis; channel has no unit."""
    axes_3d = generate_axes_dict(3, unit="millimeter")
    assert all(ax["unit"] == "millimeter" for ax in axes_3d)
    axes_4d = generate_axes_dict(4, unit="millimeter")
    # Channel axis (index 0) carries only name + type — no "unit" key.
    assert "unit" not in axes_4d[0]
    assert axes_4d[0] == {"name": "c", "type": "channel"}
    assert all(ax["unit"] == "millimeter" for ax in axes_4d[1:])


def test_validate_n_levels_exact_boundary():
    """A 16³ volume supports exactly 4 levels (log2(16)=4) — no clamp."""
    assert validate_n_levels(4, (16, 16, 16), ["z", "y", "x"]) == 4


def test_validate_n_levels_clamps_to_downsampled_axes():
    """An 8³ volume can only support 3 levels (log2(8)=3 < requested 4)."""
    assert validate_n_levels(4, (8, 8, 8), ["z", "y", "x"]) == 3


def test_save_zarr_load_zarr_round_trip(tmp_path):
    """save_zarr -> load_zarr preserves data, shape, dtype, axes, NGFF v0.5
    metadata, anisotropic per-level coordinateTransformations, and level count.

    Uses the fresh-directory (non-overwrite) save_zarr path: ``os.mkdir`` creates
    the zarr directory, so the path must not exist beforehand. With
    ``validate_n_levels(4, (32,32,32), ["z","y","x"]) == 4`` the writer
    produces 5 total levels (base + 4 downsample levels). The dict-form
    ``scale_factors`` keeps Z anisotropic (Z scale stays at 6.5 across all
    levels) while Y/X grow cumulatively (6.5, 13, 26, 52, 104).
    """
    data = np.zeros((32, 32, 32), dtype=np.uint16)
    data[8:24, 8:24, 8:24] = 1000
    zpath = str(tmp_path / "vol.zarr")
    save_zarr(data, zpath, scales=(6.5, 6.5, 6.5), chunks=(32, 32, 32))

    nodes = load_zarr(zpath)
    img = nodes[0]

    # 5 pyramid levels (validate_n_levels(4, (32,32,32), ["z","y","x"]) == 4
    # -> base + 4 downsample levels)
    assert len(img.data) == 5
    # Level-0 data equality, shape, dtype
    assert np.array_equal(np.asarray(img.data[0]), data)
    assert img.data[0].shape == (32, 32, 32)
    assert img.data[0].dtype == np.uint16
    # Axes: ome-zarr 0.18 strips the 'unit' key on read; assert name/type only
    axes = img.metadata["axes"]
    assert [a["name"] for a in axes] == ["z", "y", "x"]
    assert [a["type"] for a in axes] == ["space", "space", "space"]

    # NGFF v0.5 schema push: root ome metadata version is "0.5".
    # The ome_zarr Reader's Node.metadata dict does not expose the raw "ome"
    # key (only "axes"/"name"/"coordinateTransformations"), so the version
    # check goes through zarr.open() directly. Use .get() — files from other
    # writers / older NGFF versions may not carry the "ome" root attr, and a
    # missing OPTIONAL metadata key is a legitimate state (returns None),
    # not an error.
    root = zarr.open(zpath, mode="r")
    ome = root.attrs.get("ome")
    assert ome is not None
    assert ome["version"] == "0.5"

    # coordinateTransformations: one list per pyramid level. The auto-derived
    # transforms come back as a list of dicts (Scale, then Translation for
    # levels > 0). Z scale stays at the base 6.5 across all levels
    # (anisotropic LSFM), while Y/X grow cumulatively by 2× per level.
    ct = img.metadata["coordinateTransformations"]
    assert len(ct) == 5
    # Level 0: a single Scale transform with the base voxel size.
    assert ct[0][0]["type"] == "scale"
    assert ct[0][0]["scale"] == [6.5, 6.5, 6.5]
    # Levels 1..4: Scale then Translation. Z stays at 6.5; Y/X double each level.
    expected_yx = [13.0, 26.0, 52.0, 104.0]
    for i in range(1, 5):
        scale_entry = ct[i][0]
        assert scale_entry["type"] == "scale"
        assert scale_entry["scale"][0] == 6.5  # Z anisotropic, stays at base
        assert scale_entry["scale"][1] == expected_yx[i - 1]
        assert scale_entry["scale"][2] == expected_yx[i - 1]
        # Translation entry present at every level > 0.
        assert ct[i][1]["type"] == "translation"


def test_save_label_to_zarr_load_zarr_round_trip(tmp_path):
    """save_label_to_zarr -> load_zarr finds the label node by name with
    matching data and NGFF v0.5 anisotropic per-level coordinateTransformations.

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
        unit="micrometer",
    )

    nodes = load_zarr(zpath)
    mask_node = load_node_by_name(nodes, "mask")

    assert mask_node is not None
    assert len(mask_node.data) == 5
    assert np.array_equal(np.asarray(mask_node.data[0]), label)
    assert mask_node.data[0].dtype == np.int8

    # NGFF v0.5 anisotropic per-level scales for the label node, mirroring the
    # image assertions: Z stays at base 6.5 across all levels while Y/X grow
    # cumulatively (13, 26, 52, 104).
    ct = mask_node.metadata["coordinateTransformations"]
    assert len(ct) == 5
    assert ct[0][0]["type"] == "scale"
    assert ct[0][0]["scale"] == [6.5, 6.5, 6.5]
    expected_yx = [13.0, 26.0, 52.0, 104.0]
    for i in range(1, 5):
        scale_entry = ct[i][0]
        assert scale_entry["type"] == "scale"
        assert scale_entry["scale"][0] == 6.5
        assert scale_entry["scale"][1] == expected_yx[i - 1]
        assert scale_entry["scale"][2] == expected_yx[i - 1]

    # NEAREST/no-interpolation guard (T-3-data-integrity): every downsampled
    # pyramid level of the label node must preserve the original integer label
    # value set. ``save_label_to_zarr`` writes with ``method=Methods.NEAREST``
    # so downsampled levels are nearest-neighbor resampled — a regression to a
    # linear/averaging method would silently produce fractional/interpolated
    # values at levels 1..N (the silent-wrong-data failure mode AGENTS.md §2
    # warns against). Use ``issubset({0, 1})`` (not equality) so the bright
    # block is allowed to downsample away at the coarsest levels while still
    # forbidding any fractional value. The dtype must remain integer at every
    # level — a float dtype would indicate averaging interpolation.
    original_values = {0, 1}
    for level in range(len(mask_node.data)):
        arr = np.asarray(mask_node.data[level])
        assert np.issubdtype(arr.dtype, np.integer), (
            f"label level {level} dtype {arr.dtype} is not integer — "
            "NEAREST resampling must preserve integer label dtype"
        )
        unique = set(int(v) for v in np.unique(arr))
        assert unique.issubset(original_values), (
            f"label level {level} unique values {unique} are not a subset of "
            f"the original label value set {original_values} — "
            "NEAREST resampling must not interpolate fractional label values"
        )


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
