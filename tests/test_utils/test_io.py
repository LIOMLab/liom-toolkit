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

from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr

from liom_toolkit.conversion.conversion import save_zarr
from liom_toolkit.utils.io import (
    _zip_work_dir,
    extract_zarr_to_image,
    finalise_zarr_to_zip,
    generate_axes_dict,
    generate_label_color_dict_mask,
    load_node_by_name,
    load_omero_channels,
    load_zarr,
    save_label_to_zarr,
    upgrade_ngff_v04_to_v05,
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
        unique = {int(v) for v in np.unique(arr)}
        assert unique.issubset(original_values), (
            f"label level {level} unique values {unique} are not a subset of "
            f"the original label value set {original_values} — "
            "NEAREST resampling must not interpolate fractional label values"
        )


def test_save_zarr_zip_write_round_trip(tmp_path):
    """save_zarr with a ``.zip`` path writes a single-file ZipStore OME-Zarr
    that load_zarr reads back with full data/metadata parity, and leaves no
    working directory behind.

    A ``.zip`` extension selects the single-file ZIP store: save_zarr writes
    the OME-Zarr to a working directory first (the ome_zarr writer's
    ``da.to_zarr`` delayed writes corrupt a ZipStore directly), packs it into
    the zip, and removes the directory. This test asserts only the ``.zip``
    file remains on disk and that the round-tripped image matches the
    directory-path round-trip on data, shape, dtype, axes, level count, and
    anisotropic per-level scales.
    """
    data = np.zeros((32, 32, 32), dtype=np.uint16)
    data[8:24, 8:24, 8:24] = 1000
    zpath = str(tmp_path / "vol.ome.zarr.zip")

    save_zarr(data, zpath, scales=(6.5, 6.5, 6.5), chunks=(32, 32, 32))

    # Only the single-file zip remains — the working directory is removed.
    assert Path(zpath).is_file()
    assert not Path(_zip_work_dir(zpath)).exists(), "working directory was not cleaned up"

    nodes = load_zarr(zpath)
    img = nodes[0]
    assert len(img.data) == 5
    assert np.array_equal(np.asarray(img.data[0]), data)
    assert img.data[0].shape == (32, 32, 32)
    assert img.data[0].dtype == np.uint16
    axes = img.metadata["axes"]
    assert [a["name"] for a in axes] == ["z", "y", "x"]
    ct = img.metadata["coordinateTransformations"]
    assert len(ct) == 5
    assert ct[0][0]["type"] == "scale"
    assert ct[0][0]["scale"] == [6.5, 6.5, 6.5]
    expected_yx = [13.0, 26.0, 52.0, 104.0]
    for i in range(1, 5):
        assert ct[i][0]["scale"][0] == 6.5
        assert ct[i][0]["scale"][1] == expected_yx[i - 1]
        assert ct[i][0]["scale"][2] == expected_yx[i - 1]


def test_save_label_to_zarr_zip_append_round_trip(tmp_path):
    """save_label_to_zarr with a ``.zip`` path appends a label into an existing
    image zip (unpack -> write label -> repack) with full data/metadata parity.

    This is the downstream-append use case: the microscope finalised an image
    to a zip, then registration/atlas adds a label. save_label_to_zarr unpacks
    the image zip into a working directory at save time, appends the label via
    the proven directory write path, repacks the directory into the zip, and
    removes the directory. load_zarr reads the zip directly (no unpack) for
    the in-memory steps. This test asserts only the ``.zip`` remains, the
    label node is found by name with matching data and integer dtype, and the
    NEAREST-resampling integer-value invariant holds at every pyramid level.
    """
    data = np.zeros((16, 16, 16), dtype=np.uint16)
    data[4:12, 4:12, 4:12] = 1000
    label = np.zeros((16, 16, 16), dtype=np.int8)
    label[4:12, 4:12, 4:12] = 1
    zpath = str(tmp_path / "vol.ome.zarr.zip")

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

    # Only the single-file zip remains after the repack.
    assert Path(zpath).is_file()
    assert not Path(_zip_work_dir(zpath)).exists(), "working directory was not cleaned up"

    nodes = load_zarr(zpath)
    mask_node = load_node_by_name(nodes, "mask")
    assert mask_node is not None
    assert len(mask_node.data) == 5
    assert np.array_equal(np.asarray(mask_node.data[0]), label)
    assert mask_node.data[0].dtype == np.int8

    # NEAREST-resampling integer invariant (mirrors the directory-path test).
    original_values = {0, 1}
    for level in range(len(mask_node.data)):
        arr = np.asarray(mask_node.data[level])
        assert np.issubdtype(arr.dtype, np.integer), (
            f"label level {level} dtype {arr.dtype} is not integer — "
            "NEAREST resampling must preserve integer label dtype"
        )
        unique = {int(v) for v in np.unique(arr)}
        assert unique.issubset(original_values), (
            f"label level {level} unique values {unique} are not a subset of "
            f"the original label value set {original_values} — "
            "NEAREST resampling must not interpolate fractional label values"
        )


def test_finalise_zarr_to_zip_round_trip(tmp_path):
    """finalise_zarr_to_zip packs a directory store (image + label) into a
    single-file ``.zip`` that load_zarr reads back with full data/metadata
    parity, and removes the source directory.

    The supported zip workflow is: write the image and any labels to a
    directory with save_zarr / save_label_to_zarr, then call
    finalise_zarr_to_zip to pack the finished directory into a zip
    (appending labels into an existing zip is not supported — the ome_zarr
    writer's ``da.to_zarr`` delayed writes corrupt a ZipStore on append).
    This test exercises that workflow end-to-end and asserts: only the
    ``.zip`` remains (source directory removed), the image node matches on
    data/shape/dtype/axes/levels/anisotropic scales, and the label node
    matches on data/integer dtype with the NEAREST-resampling integer-value
    invariant holding at every pyramid level.
    """
    data = np.zeros((16, 16, 16), dtype=np.uint16)
    data[4:12, 4:12, 4:12] = 1000
    label = np.zeros((16, 16, 16), dtype=np.int8)
    label[4:12, 4:12, 4:12] = 1
    dir_path = str(tmp_path / "vol.ome.zarr")

    # Write image + label to the directory (the working format).
    save_zarr(data, dir_path, scales=(6.5, 6.5, 6.5), chunks=(16, 16, 16))
    save_label_to_zarr(
        label=label,
        zarr_file=dir_path,
        color_dict=generate_label_color_dict_mask(),
        name="mask",
        scales=(6.5, 6.5, 6.5),
        chunks=(16, 16, 16),
        resolution_level=0,
        unit="micrometer",
    )

    zip_path = finalise_zarr_to_zip(dir_path)

    # The zip is at <dir>.zip and the source directory is removed.
    assert zip_path == dir_path + ".zip"
    assert Path(zip_path).is_file()
    assert not Path(dir_path).exists(), "source directory was not removed"

    nodes = load_zarr(zip_path)
    img = nodes[0]
    assert len(img.data) == 5
    assert np.array_equal(np.asarray(img.data[0]), data)
    assert img.data[0].shape == (16, 16, 16)
    assert img.data[0].dtype == np.uint16
    axes = img.metadata["axes"]
    assert [a["name"] for a in axes] == ["z", "y", "x"]
    ct = img.metadata["coordinateTransformations"]
    assert len(ct) == 5
    assert ct[0][0]["type"] == "scale"
    assert ct[0][0]["scale"] == [6.5, 6.5, 6.5]
    expected_yx = [13.0, 26.0, 52.0, 104.0]
    for i in range(1, 5):
        assert ct[i][0]["scale"][0] == 6.5
        assert ct[i][0]["scale"][1] == expected_yx[i - 1]
        assert ct[i][0]["scale"][2] == expected_yx[i - 1]

    mask_node = load_node_by_name(nodes, "mask")
    assert mask_node is not None
    assert len(mask_node.data) == 5
    assert np.array_equal(np.asarray(mask_node.data[0]), label)
    assert mask_node.data[0].dtype == np.int8

    # NEAREST-resampling integer invariant (mirrors the directory-path test).
    original_values = {0, 1}
    for level in range(len(mask_node.data)):
        arr = np.asarray(mask_node.data[level])
        assert np.issubdtype(arr.dtype, np.integer), (
            f"label level {level} dtype {arr.dtype} is not integer — "
            "NEAREST resampling must preserve integer label dtype"
        )
        unique = {int(v) for v in np.unique(arr)}
        assert unique.issubset(original_values), (
            f"label level {level} unique values {unique} are not a subset of "
            f"the original label value set {original_values} — "
            "NEAREST resampling must not interpolate fractional label values"
        )


def test_finalise_zarr_to_zip_missing_dir_raises(tmp_path):
    """finalise_zarr_to_zip raises FileNotFoundError for a missing directory.

    A missing input directory is an explicit failure (AGENTS.md §2: no silent
    wrong-data fallback), not a silent empty-zip write.
    """
    missing = str(tmp_path / "does_not_exist.ome.zarr")
    with pytest.raises(FileNotFoundError, match="zarr directory not found"):
        finalise_zarr_to_zip(missing)


def test_finalise_zarr_to_zip_keep_dir(tmp_path):
    """finalise_zarr_to_zip(remove_dir=False) leaves the source directory in
    place so further appends remain possible, while still producing the zip.
    """
    data = np.zeros((16, 16, 16), dtype=np.uint16)
    dir_path = str(tmp_path / "vol.ome.zarr")
    save_zarr(data, dir_path, scales=(6.5, 6.5, 6.5), chunks=(16, 16, 16))

    zip_path = finalise_zarr_to_zip(dir_path, remove_dir=False)

    assert Path(zip_path).is_file()
    assert Path(dir_path).is_dir(), "source directory was removed despite remove_dir=False"
    # The zip is readable and round-trips the image data.
    nodes = load_zarr(zip_path)
    assert np.array_equal(np.asarray(nodes[0].data[0]), data)


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


def _dir_zarr_to_zip(dir_path: str, zip_path: str) -> None:
    """Copy a directory zarr store into a single-file ZipStore.

    Reads every key (chunk + metadata file) from the source ``LocalStore``
    and writes it into a ``ZipStore`` via the async store API. This is the
    on-disk transformation the data-offload workflow applies to high-chunk-
    count stores (a 971k-file zarr becomes one .zip), and the round-trip
    tests below assert ``load_zarr`` reads the result identically to the
    directory original.
    """
    import asyncio

    from zarr.storage import LocalStore, ZipStore

    src = LocalStore(dir_path, read_only=True)
    dest = ZipStore(zip_path, mode="w", compression=0)

    async def _copy() -> None:
        keys = [k async for k in src.list_prefix("")]
        for k in keys:
            buf = await src.get(k)
            if buf is not None:
                await dest.set(k, buf)

    asyncio.run(_copy())
    dest.close()


def test_load_zarr_zip_store_image_round_trip(tmp_path):
    """load_zarr reads a single-file ZIP OME-Zarr identically to the dir store.

    A directory store written by ``save_zarr`` is copied into a ``.zarr.zip``
    ZipStore; ``load_zarr`` must return an image node with the same pyramid
    level count, per-level shape/dtype/data, axes, name, and per-level
    ``coordinateTransformations`` as the directory original. This is the
    transparency contract: callers pass a ``.zip`` path and get the same
    nodes back, even though ``ome_zarr.io.parse_url`` cannot route a ``.zip``
    path (it returns ``None`` for non-directory paths).
    """
    data = np.zeros((32, 32, 32), dtype=np.uint16)
    data[8:24, 8:24, 8:24] = 1000
    dir_path = str(tmp_path / "vol.zarr")
    zip_path = str(tmp_path / "vol.zarr.zip")
    save_zarr(data, dir_path, scales=(6.5, 6.5, 6.5), chunks=(32, 32, 32))
    _dir_zarr_to_zip(dir_path, zip_path)

    dir_nodes = load_zarr(dir_path)
    zip_nodes = load_zarr(zip_path)

    dir_img = load_node_by_name(dir_nodes, "image")
    zip_img = load_node_by_name(zip_nodes, "image")
    assert dir_img is not None and zip_img is not None

    assert len(zip_img.data) == len(dir_img.data) == 5
    for level in range(len(dir_img.data)):
        a = np.asarray(dir_img.data[level])
        b = np.asarray(zip_img.data[level])
        assert b.shape == a.shape
        assert b.dtype == a.dtype
        assert np.array_equal(b, a)

    assert zip_img.metadata["axes"] == dir_img.metadata["axes"]
    assert zip_img.metadata["name"] == dir_img.metadata["name"] == "image"
    assert (
        zip_img.metadata["coordinateTransformations"]
        == dir_img.metadata["coordinateTransformations"]
    )


def test_load_zarr_zip_store_label_round_trip(tmp_path):
    """load_zarr reads a ZIP store's label node identically to the dir store.

    The image is written first (establishing root multiscales metadata),
    then a ``mask`` label. After copying to a ZipStore, ``load_node_by_name``
    must find the ``mask`` node with matching per-level data, axes, and
    ``coordinateTransformations``. The label node's integer dtype and
    {0, 1} value subset are preserved (NEAREST resampling guard).
    """
    data = np.zeros((16, 16, 16), dtype=np.uint16)
    data[4:12, 4:12, 4:12] = 1000
    label = np.zeros((16, 16, 16), dtype=np.int8)
    label[4:12, 4:12, 4:12] = 1
    dir_path = str(tmp_path / "vol.zarr")
    zip_path = str(tmp_path / "vol.zarr.zip")
    save_zarr(data, dir_path, scales=(6.5, 6.5, 6.5), chunks=(16, 16, 16))
    save_label_to_zarr(
        label=label,
        zarr_file=dir_path,
        color_dict=generate_label_color_dict_mask(),
        name="mask",
        scales=(6.5, 6.5, 6.5),
        chunks=(16, 16, 16),
        resolution_level=0,
        unit="micrometer",
    )
    _dir_zarr_to_zip(dir_path, zip_path)

    zip_nodes = load_zarr(zip_path)
    mask_node = load_node_by_name(zip_nodes, "mask")

    assert mask_node is not None
    assert len(mask_node.data) == 5
    assert np.array_equal(np.asarray(mask_node.data[0]), label)
    assert mask_node.data[0].dtype == np.int8
    ct = mask_node.metadata["coordinateTransformations"]
    assert len(ct) == 5
    assert ct[0][0]["type"] == "scale"
    assert ct[0][0]["scale"] == [6.5, 6.5, 6.5]
    for level in range(len(mask_node.data)):
        arr = np.asarray(mask_node.data[level])
        assert np.issubdtype(arr.dtype, np.integer)
        unique = {int(v) for v in np.unique(arr)}
        assert unique.issubset({0, 1})


def test_load_zarr_zip_store_non_ome_raises(tmp_path):
    """A ZIP store without OME-Zarr multiscales metadata raises ValueError.

    A bare zarr group (no ``ome`` root attribute) packed into a zip is not a
    valid OME-Zarr file; ``load_zarr`` must raise rather than return an empty
    or wrong node list (the no-silent-wrong-data rule, AGENTS.md §2).
    """
    from zarr.storage import ZipStore

    zip_path = str(tmp_path / "not_ome.zarr.zip")
    store = ZipStore(zip_path, mode="w")
    g = zarr.open_group(store=store, mode="w")
    g.attrs["not_ome"] = True
    store.close()

    with pytest.raises(ValueError, match="multiscales metadata"):
        load_zarr(zip_path)


def _make_synthetic_zarr(path: str, shape=(4, 8, 8)) -> np.ndarray:
    """Write a tiny single-channel zarr volume and return the source array.

    The volume has a bright cube so per-slice PNGs are non-constant (the
    ``convert_to_png_for_saving`` constant-image branch returns all-zero,
    which would still round-trip but is a weaker assertion).
    """
    arr = np.zeros(shape, dtype=np.uint16)
    arr[1:3, 2:6, 2:6] = 1000
    save_zarr(arr, path, scales=(6.5, 6.5, 6.5), chunks=shape)
    return arr


def test_extract_zarr_to_image_tiff_round_trip(tmp_path):
    """extract_zarr_to_image(format='tiff') writes a single multi-page TIFF.

    The default format writes one multi-page TIFF (one page per Z slice) via
    ``tifffile.imwrite``; ``tifffile.imread`` reads it back with the expected
    shape. The previous name ``extract_zarr_to_png`` wrote per-slice PNGs
    only; the rename + ``format`` param adds the TIFF default with a PNG
    escape hatch.
    """
    zpath = str(tmp_path / "vol.zarr")
    _make_synthetic_zarr(zpath, shape=(4, 8, 8))
    target = str(tmp_path / "out_tiff")
    tiff_path = str(Path(target) / "extracted.tiff")

    extract_zarr_to_image(zpath, target, channel=0, format="tiff")

    assert Path(tiff_path).exists()
    back = tifffile.imread(tiff_path)
    # Multi-page TIFF: one page per Z slice -> shape (Z, Y, X).
    assert back.shape == (4, 8, 8)
    # The TIFF pages hold the normalized-to-uint8 slices (convert_to_png_for_saving
    # min-max normalizes each slice to [0, 255]). Assert the bright block is
    # present in the middle slices and absent at the edges.
    assert back[0].max() == 0  # slice 0 is all-zero (no bright block)
    assert back[1].max() == 255  # slice 1 has the bright block


def test_extract_zarr_to_image_png_escape_hatch(tmp_path):
    """extract_zarr_to_image(format='png') writes per-slice PNGs (escape hatch).

    The PNG escape hatch preserves PNG consumers after the TIFF default
    switch. One PNG per Z slice is written to the target directory.
    """
    zpath = str(tmp_path / "vol_png.zarr")
    _make_synthetic_zarr(zpath, shape=(4, 8, 8))
    target = str(tmp_path / "out_png")

    extract_zarr_to_image(zpath, target, channel=0, format="png")

    target_dir = Path(target)
    for z in range(4):
        assert (target_dir / f"{z}.png").exists()


def test_extract_zarr_to_image_unsupported_format_raises(tmp_path):
    """extract_zarr_to_image(format='unsupported') raises ValueError.

    An unsupported format string must raise ``ValueError`` (never silently
    fall back to a default or write no output — AGENTS §2 no-silent-wrong-data).
    """
    zpath = str(tmp_path / "vol_err.zarr")
    _make_synthetic_zarr(zpath, shape=(4, 8, 8))
    target = str(tmp_path / "out_err")

    with pytest.raises(ValueError):
        extract_zarr_to_image(zpath, target, channel=0, format="unsupported")


# ---------------------------------------------------------------------------
# extract_zarr_to_image PNG ThreadPoolExecutor parallelization (PERF-01c).
# The format="png" escape hatch writes one PNG per Z slice; the parallelization
# replaces the sequential for-loop with a ThreadPoolExecutor.map. Each slice z
# gets a unique filename {z}.png -- no clobber, no shared file.
# ---------------------------------------------------------------------------


def test_extract_zarr_to_image_png_parallel(tmp_path):
    """extract_zarr_to_image(format='png') writes all expected per-slice PNGs
    via ThreadPoolExecutor. The primary assertion is that every {z}.png file
    exists and is a valid readable image with the expected slice content."""
    zpath = str(tmp_path / "vol_par.zarr")
    src = _make_synthetic_zarr(zpath, shape=(6, 8, 8))
    target = str(tmp_path / "out_png_par")

    extract_zarr_to_image(zpath, target, channel=0, format="png")

    import imageio.v3 as iio

    target_dir = Path(target)
    png_files = sorted(target_dir.glob("*.png"), key=lambda p: int(p.stem))
    assert len(png_files) == src.shape[0], (
        f"expected {src.shape[0]} PNG files, got {len(png_files)}"
    )
    # Each file is named {z}.png and is readable.
    for z, p in enumerate(png_files):
        assert p.name == f"{z}.png"
        back = iio.imread(str(p))
        assert back.shape == (8, 8)  # Y, X


def test_extract_zarr_to_image_png_no_clobber(tmp_path):
    """Each slice z gets a unique filename {z}.png -- no two slices share a
    file. The number of unique PNG files equals volume.shape[0]."""
    zpath = str(tmp_path / "vol_clobber.zarr")
    _make_synthetic_zarr(zpath, shape=(5, 8, 8))
    target = str(tmp_path / "out_png_clobber")

    extract_zarr_to_image(zpath, target, channel=0, format="png")

    target_dir = Path(target)
    png_files = list(target_dir.glob("*.png"))
    names = {p.name for p in png_files}
    assert len(names) == 5, f"expected 5 unique PNG filenames, got {len(names)} (slice clobber)"
    assert names == {f"{z}.png" for z in range(5)}


def _make_v04_zarr(path: str, shape=(8, 8, 8)) -> np.ndarray:
    """Write a tiny NGFF v0.4-style store: ``multiscales`` at the group root
    (no ``ome`` wrapper), one multiscale entry with two levels.

    The current toolkit writer emits v0.5 (``ome.multiscales``); this helper
    builds the v0.4 shape directly so the upgrade path can be tested against
    the legacy on-disk format the lab's existing stores use.
    """
    arr = np.zeros(shape, dtype=np.uint16)
    arr[2:6, 2:6, 2:6] = 1000
    g = zarr.open_group(path, mode="w")
    g.create_array("0", data=arr)
    # Level 1: 2x downsample on Y/X (Z stays) -- a real but tiny second level.
    l1 = arr[:, ::2, ::2]
    g.create_array("1", data=l1)
    g.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": [
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            "datasets": [
                {
                    "coordinateTransformations": [{"type": "scale", "scale": [6.5, 6.5, 6.5]}],
                    "path": "0",
                },
                {
                    "coordinateTransformations": [{"type": "scale", "scale": [6.5, 13.0, 13.0]}],
                    "path": "1",
                },
            ],
            "name": "/",
        }
    ]
    return arr


def test_upgrade_ngff_v04_to_v05_wraps_metadata(tmp_path):
    """upgrade_ngff_v04_to_v05 moves root-level multiscales under an ``ome``
    key with ``version: "0.5"`` and drops the v0.4 root-level copy.

    The chunk data and array paths are untouched -- only the root group's
    metadata is rewritten -- so load_zarr reads the upgraded store back with
    the same data, shape, dtype, axes, and per-level scales.
    """
    zpath = str(tmp_path / "v04.ome.zarr")
    arr = _make_v04_zarr(zpath)

    upgraded = upgrade_ngff_v04_to_v05(zpath)
    assert upgraded is True

    g = zarr.open(zpath, mode="r")
    attrs = dict(g.attrs)
    assert "multiscales" not in attrs, "v0.4 root-level multiscales was not removed"
    ome = attrs["ome"]
    assert ome["version"] == "0.5"
    assert "multiscales" in ome
    assert len(ome["multiscales"]) == 1

    # load_zarr reads the upgraded store with data/metadata parity.
    nodes = load_zarr(zpath)
    img = nodes[0]
    assert len(img.data) == 2
    assert np.array_equal(np.asarray(img.data[0]), arr)
    assert img.data[0].shape == arr.shape
    assert img.data[0].dtype == np.uint16
    axes = img.metadata["axes"]
    assert [a["name"] for a in axes] == ["z", "y", "x"]
    ct = img.metadata["coordinateTransformations"]
    assert ct[0][0]["scale"] == [6.5, 6.5, 6.5]
    assert ct[1][0]["scale"] == [6.5, 13.0, 13.0]


def test_upgrade_ngff_v04_to_v05_idempotent(tmp_path):
    """A second upgrade call on an already-v0.5 store is a no-op and returns
    False (no error, no metadata churn)."""
    zpath = str(tmp_path / "v04.ome.zarr")
    _make_v04_zarr(zpath)
    upgrade_ngff_v04_to_v05(zpath)
    before = dict(zarr.open(zpath, mode="r").attrs)

    upgraded = upgrade_ngff_v04_to_v05(zpath)
    assert upgraded is False

    after = dict(zarr.open(zpath, mode="r").attrs)
    assert before == after


def test_upgrade_ngff_v04_to_v05_errors(tmp_path):
    """upgrade_ngff_v04_to_v05 raises FileNotFoundError for a missing dir and
    ValueError for a non-OME store (no v0.4 multiscales, no v0.5 ome)."""
    missing = str(tmp_path / "nope.ome.zarr")
    with pytest.raises(FileNotFoundError, match="zarr directory not found"):
        upgrade_ngff_v04_to_v05(missing)

    not_ome = str(tmp_path / "not_ome.zarr")
    zarr.open_group(not_ome, mode="w")  # empty group, no OME metadata
    with pytest.raises(ValueError, match="not an OME-Zarr store"):
        upgrade_ngff_v04_to_v05(not_ome)


def test_load_zarr_reads_v04_zip(tmp_path):
    """load_zarr reads a v0.4 store packed into a zip -- the zip reader
    handles root-level ``multiscales`` (v0.4) as well as ``ome.multiscales``
    (v0.5), so a v0.4 store can be finalised to zip and read back without an
    upgrade. This is the microscope-finalises-then-downstream-reads path for
    legacy v0.4 stores."""
    zdir = str(tmp_path / "v04.ome.zarr")
    arr = _make_v04_zarr(zdir)
    zip_path = finalise_zarr_to_zip(zdir)
    assert not Path(zdir).exists()

    nodes = load_zarr(zip_path)
    img = nodes[0]
    assert len(img.data) == 2
    assert np.array_equal(np.asarray(img.data[0]), arr)
    axes = img.metadata["axes"]
    assert [a["name"] for a in axes] == ["z", "y", "x"]
    ct = img.metadata["coordinateTransformations"]
    assert ct[0][0]["scale"] == [6.5, 6.5, 6.5]
    assert ct[1][0]["scale"] == [6.5, 13.0, 13.0]


def _make_v04_zarr_with_label(path: str, shape=(8, 8, 8)) -> tuple[np.ndarray, np.ndarray]:
    """Write a tiny NGFF v0.4-style store WITH a ``mask`` label.

    Mirrors :func:`_make_v04_zarr` for the image, then adds a ``labels/mask``
    subgroup carrying v0.4 root-level ``multiscales`` + ``image-label`` (no
    ``ome`` wrapper). The ``labels`` group carries a root-level ``labels``
    attribute -- a list of ``{"label": "mask"}`` dicts -- which is the v0.4
    labels-group convention the zip reader must consult (``ome.labels`` is
    absent on a v0.4 labels group). Returns the image and label arrays.
    """
    arr = _make_v04_zarr(path, shape=shape)
    g = zarr.open_group(path, mode="r+")
    labels_group = g.require_group("labels")
    mask_group = labels_group.require_group("mask")
    label = np.zeros(shape, dtype=np.int8)
    label[2:6, 2:6, 2:6] = 1
    mask_group.create_array("0", data=label)
    # Level 1: 2x downsample on Y/X (Z stays), nearest-neighbour.
    l1 = label[:, ::2, ::2]
    mask_group.create_array("1", data=l1)
    mask_group.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": [
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            "datasets": [
                {
                    "coordinateTransformations": [{"type": "scale", "scale": [6.5, 6.5, 6.5]}],
                    "path": "0",
                },
                {
                    "coordinateTransformations": [{"type": "scale", "scale": [6.5, 13.0, 13.0]}],
                    "path": "1",
                },
            ],
            "name": "mask",
        }
    ]
    mask_group.attrs["image-label"] = {
        "colors": [
            {"label-value": 0, "rgba": [0, 0, 0, 0]},
            {"label-value": 1, "rgba": [250, 0, 0, 255]},
        ],
        "source": {"image": "../../"},
        "visible": True,
    }
    # v0.4 labels-group convention: a root-level "labels" list of
    # {"label": name} dicts (no "ome" wrapper on the labels group itself).
    labels_group.attrs["labels"] = [{"label": "mask"}]
    return arr, label


def test_load_zarr_reads_v04_zip_with_label(tmp_path):
    """load_zarr reads a v0.4 store WITH a label packed into a zip.

    Regression guard: the zip reader's label discovery previously only
    consulted the v0.5 ``ome.labels`` list, which is absent on a v0.4 labels
    group, so the ``mask`` label was silently dropped on read. The fix
    discovers label names from the v0.4 labels-group ``labels`` attribute
    (list of ``{"label": name}`` dicts) and/or the labels-group child group
    keys. This test writes a v0.4 store with a ``mask`` label, finalises it
    to a zip, and asserts the label node is found by name and round-trips.
    """
    zdir = str(tmp_path / "v04_label.ome.zarr")
    arr, label = _make_v04_zarr_with_label(zdir)
    zip_path = finalise_zarr_to_zip(zdir)
    assert not Path(zdir).exists()

    nodes = load_zarr(zip_path)
    # Image node is still present.
    img = nodes[0]
    assert np.array_equal(np.asarray(img.data[0]), arr)

    mask_node = load_node_by_name(nodes, "mask")
    assert mask_node is not None, "v0.4 label 'mask' was silently dropped on zip read"
    assert len(mask_node.data) == 2
    assert np.array_equal(np.asarray(mask_node.data[0]), label)
    assert mask_node.data[0].dtype == np.int8
    axes = mask_node.metadata["axes"]
    assert [a["name"] for a in axes] == ["z", "y", "x"]
    ct = mask_node.metadata["coordinateTransformations"]
    assert ct[0][0]["scale"] == [6.5, 6.5, 6.5]
    assert ct[1][0]["scale"] == [6.5, 13.0, 13.0]


def test_load_omero_channels_reads_zip(tmp_path):
    """load_omero_channels reads omero channels from a ``.zip`` OME-Zarr.

    Regression guard: load_omero_channels previously opened the store via
    ``zarr.open``, which routes a ``.zip`` path to ``LocalStore`` (a
    directory store) and raises ``GroupNotFoundError`` -- it did not use
    the ``ZipStore`` dispatch that ``load_zarr`` uses. So a caller who
    finalised an OME-Zarr to ``.zip`` and then called
    ``load_omero_channels(zip_path)`` got a hard failure instead of the
    channel metadata. The fix dispatches on ``_is_zip_zarr`` and opens the
    zip via ``ZipStore``.
    """
    zdir = str(tmp_path / "omero.ome.zarr")
    arr = np.zeros((8, 8, 8), dtype=np.uint16)
    g = zarr.open_group(zdir, mode="w")
    g.create_array("0", data=arr)
    omero_channels = [
        {"label": "GFP", "color": "00FF00", "active": True, "wavelength": 488},
    ]
    g.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "axes": [
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "datasets": [
                    {
                        "coordinateTransformations": [{"type": "scale", "scale": [6.5, 6.5, 6.5]}],
                        "path": "0",
                    }
                ],
                "name": "/",
            }
        ],
        "omero": {"channels": omero_channels},
    }

    zip_path = finalise_zarr_to_zip(zdir)
    assert not Path(zdir).exists()

    channels = load_omero_channels(zip_path)
    assert channels is not None
    assert len(channels) == 1
    assert channels[0]["label"] == "GFP"
    assert channels[0]["color"] == "00FF00"


def test_load_zarr_zip_label_without_multiscales_raises(tmp_path):
    """A label group carrying ``image-label`` but no ``multiscales`` raises
    ValueError with the label name, not a bare KeyError/IndexError.

    A malformed but on-disk-real store (e.g. a label group whose
    multiscales metadata was stripped or never written) must fail
    explicitly and actionable (AGENTS.md §2: no silent wrong-data
    fallback, and assert is not validation) -- a raw ``KeyError:
    'multiscales'`` with no file context is not actionable.
    """
    zdir = str(tmp_path / "bad_label.ome.zarr")
    arr = np.zeros((8, 8, 8), dtype=np.uint16)
    g = zarr.open_group(zdir, mode="w")
    g.create_array("0", data=arr)
    g.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "axes": [
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "datasets": [
                    {
                        "coordinateTransformations": [{"type": "scale", "scale": [6.5, 6.5, 6.5]}],
                        "path": "0",
                    }
                ],
                "name": "/",
            }
        ],
    }
    # Malformed label group: image-label present, multiscales absent.
    labels_group = g.require_group("labels")
    labels_group.attrs["ome"] = {"version": "0.5", "labels": ["mask"]}
    mask_group = labels_group.require_group("mask")
    mask_group.attrs["ome"] = {"version": "0.5", "image-label": {"visible": True}}

    zip_path = finalise_zarr_to_zip(zdir)
    with pytest.raises(ValueError, match="label group 'mask' has no 'multiscales'"):
        load_zarr(zip_path)


def test_load_node_by_name_on_zip_uses_visible(tmp_path):
    """load_node_by_name on a zip node reads ``.visible`` (a property backed
    by the name-mangled ``_Node__visible`` set in ``Node.__init__``). The
    ``_ZipNode`` subclass bypasses ``Node.__init__`` and must set
    ``_Node__visible`` explicitly, or ``load_node_by_name`` raises
    ``AttributeError: '_ZipNode' object has no attribute '_Node__visible'``.
    """
    data = np.zeros((8, 8, 8), dtype=np.uint16)
    label = np.zeros((8, 8, 8), dtype=np.int8)
    label[2:6, 2:6, 2:6] = 1
    dir_path = str(tmp_path / "vol.ome.zarr")
    save_zarr(data, dir_path, scales=(6.5, 6.5, 6.5), chunks=(8, 8, 8))
    save_label_to_zarr(
        label=label,
        zarr_file=dir_path,
        color_dict=generate_label_color_dict_mask(),
        name="mask",
        scales=(6.5, 6.5, 6.5),
        chunks=(8, 8, 8),
        resolution_level=0,
        unit="micrometer",
    )
    zip_path = finalise_zarr_to_zip(dir_path)

    nodes = load_zarr(zip_path)
    # load_node_by_name reads .visible on each node -- this would raise
    # AttributeError if _ZipNode did not set _Node__visible.
    mask_node = load_node_by_name(nodes, "mask")
    assert mask_node is not None
    assert np.array_equal(np.asarray(mask_node.data[0]), label)
    # The image node is nodes[0] (named "/"); confirm .visible is readable
    # on it too (the property backed by _Node__visible).
    assert nodes[0].visible is True
    assert np.array_equal(np.asarray(nodes[0].data[0]), data)
