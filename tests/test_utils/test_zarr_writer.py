"""Tests for ``liom_toolkit/utils/zarr_writer.py``.

Covers the streaming OME-Zarr writer (NGFF v0.5), the analysis target-
resolution writer, and the two pure-logic helpers that live in the same
module (``create_directory`` and ``create_transformation_dict``).

Mirrors the established ``tests/test_utils/test_io.py`` round-trip pattern:
real zarr groups written to ``tmp_path``, no mocking of zarr/dask/numpy
(AGENTS.md §5). Pure-logic helpers are tested by direct call + assert on
small synthetic inputs.

The streaming round-trip tests are the guard for the three divergences from
the linumpy pattern (see the ``zarr_writer`` module docstring):

* channel-axis scale = 1.0 (linumpy's bug was 0.0),
* anisotropic Y/X-only downsample (Z stays at base),
* AnalysisOmeZarrWriter keeps raw L0 (linumpy replaces it destructively).
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import zarr

from liom_toolkit.utils.io import load_omero_channels, load_zarr
from liom_toolkit.utils.zarr_writer import (
    AnalysisOmeZarrWriter,
    OmeZarrWriter,
    create_directory,
    create_transformation_dict,
)


# ---------------------------------------------------------------------------
# create_directory — pure-logic / filesystem tests (use tmp_path, real dirs)
# ---------------------------------------------------------------------------


def test_create_directory_creates(tmp_path):
    """create_directory on a fresh subdir returns the path and it exists."""
    target = tmp_path / "fresh"
    result = create_directory(target, overwrite=False)
    assert result == target
    assert target.exists()
    assert target.is_dir()


def test_create_directory_existing_no_overwrite_raises(tmp_path):
    """Pre-existing dir + overwrite=False raises FileExistsError."""
    target = tmp_path / "exists"
    target.mkdir()
    with pytest.raises(FileExistsError):
        create_directory(target, overwrite=False)


def test_create_directory_existing_overwrite_replaces(tmp_path):
    """overwrite=True removes the existing dir (with a sentinel file) and
    recreates an empty one."""
    target = tmp_path / "exists"
    target.mkdir()
    (target / "sentinel.txt").write_text("old")
    create_directory(target, overwrite=True)
    assert target.exists()
    assert target.is_dir()
    # Sentinel file is gone — the dir was replaced, not merged into.
    assert not (target / "sentinel.txt").exists()


def test_create_directory_symlink_overwrite_unlinks(tmp_path):
    """overwrite=True on a symlink unlinks the link (target untouched) and
    creates a real directory in its place."""
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    (real_dir / "keep.txt").write_text("target contents")
    link = tmp_path / "link"
    os.symlink(real_dir, link)

    create_directory(link, overwrite=True)

    # The symlink is gone; a real directory exists in its place.
    assert link.exists()
    assert link.is_dir()
    assert not link.is_symlink()
    # The original target directory is untouched.
    assert real_dir.exists()
    assert (real_dir / "keep.txt").read_text() == "target contents"


# ---------------------------------------------------------------------------
# create_transformation_dict — pure-logic tests (no I/O)
# ---------------------------------------------------------------------------


def test_create_transformation_dict_4d_channel_scale_one():
    """4D: channel scale == 1.0, Z constant, Y/X cumulative by 2**i."""
    transforms = create_transformation_dict(
        n_levels=4, voxel_size=(6.5, 6.5, 6.5), ndims=4, downscale_factor=2
    )
    assert len(transforms) == 4
    for i, entry in enumerate(transforms):
        assert entry == [{"type": "scale", "scale": entry[0]["scale"]}]
        scale = entry[0]["scale"]
        assert len(scale) == 4
        assert scale[0] == 1.0  # channel
        assert scale[1] == 6.5  # Z constant
        assert scale[2] == pytest.approx(6.5 * 2**i)  # Y cumulative
        assert scale[3] == pytest.approx(6.5 * 2**i)  # X cumulative


def test_create_transformation_dict_3d_no_channel():
    """3D: scale lists have 3 elements [z, y*2**i, x*2**i], no channel."""
    transforms = create_transformation_dict(
        n_levels=3, voxel_size=(6.5, 6.5, 6.5), ndims=3, downscale_factor=2
    )
    assert len(transforms) == 3
    for i, entry in enumerate(transforms):
        scale = entry[0]["scale"]
        assert len(scale) == 3
        assert scale[0] == 6.5  # Z
        assert scale[1] == pytest.approx(6.5 * 2**i)  # Y
        assert scale[2] == pytest.approx(6.5 * 2**i)  # X


def test_create_transformation_dict_downscale_factor():
    """downscale_factor=3 -> Y/X = 6.5 * 3**i (not 2**i)."""
    transforms = create_transformation_dict(
        n_levels=3, voxel_size=(6.5, 6.5, 6.5), ndims=4, downscale_factor=3
    )
    for i, entry in enumerate(transforms):
        scale = entry[0]["scale"]
        assert scale[0] == 1.0
        assert scale[1] == 6.5  # Z stays at base regardless of factor
        assert scale[2] == pytest.approx(6.5 * 3**i)
        assert scale[3] == pytest.approx(6.5 * 3**i)


# ---------------------------------------------------------------------------
# OmeZarrWriter — streaming round-trip tests (real zarr in tmp_path)
# ---------------------------------------------------------------------------


def _root_attrs(zpath: str) -> dict | None:
    """Read the raw root ``ome`` attrs via zarr.open (the ome-zarr Reader
    Node.metadata does not expose the raw ``ome`` key).

    Tolerates a missing ``ome`` key (returns ``None``) — ome-zarr files from
    elsewhere, older NGFF versions, or files written without channel
    metadata may not carry the ``ome`` root attr. A missing OPTIONAL
    metadata key is a legitimate state, not an error (AGENTS.md §2 inverted
    on the read side: surface as None, not a crash).
    """
    return zarr.open(zpath, mode="r").attrs.get("ome")


def test_omezarrwriter_streaming_3d_roundtrip(tmp_path):
    """3D streaming: write 4 planes via writer[z,:,:]=frame, finalize with
    n_levels=2; load_zarr -> L0 data == frames, 3 levels, per-level scale
    == [6.5, 6.5*2**i, 6.5*2**i], ome.version == '0.5'."""
    zpath = str(tmp_path / "vol3d.zarr")
    writer = OmeZarrWriter(
        store_path=zpath,
        shape=(4, 16, 16),
        chunk_shape=(1, 16, 16),
        dtype=np.uint16,
        overwrite=True,
        downscale_factor=2,
        unit="micrometer",
    )
    frames = [np.full((16, 16), 100 * (z + 1), dtype=np.uint16) for z in range(4)]
    for z, frame in enumerate(frames):
        writer[z, :, :] = frame
    writer.finalize(res=(6.5, 6.5, 6.5), n_levels=2)

    nodes = load_zarr(zpath)
    img = nodes[0]
    expected_l0 = np.stack(frames, axis=0)
    assert len(img.data) == 3  # L0 + 2
    assert np.array_equal(np.asarray(img.data[0]), expected_l0)
    assert img.data[0].shape == (4, 16, 16)
    assert img.data[0].dtype == np.uint16

    ct = img.metadata["coordinateTransformations"]
    assert len(ct) == 3
    for i in range(3):
        scale = ct[i][0]["scale"]
        assert scale[0] == 6.5  # Z constant
        assert scale[1] == pytest.approx(6.5 * 2**i)
        assert scale[2] == pytest.approx(6.5 * 2**i)

    assert _root_attrs(zpath)["version"] == "0.5"


def test_omezarrwriter_streaming_4d_multichannel_omero(tmp_path):
    """4D multichannel + omero: write frames per channel/plane, finalize
    with omero_channels for 2 channels; load_zarr -> L0 data == frames,
    omero.channels at root.attrs['ome']['omero']['channels'] with 2 entries,
    per-level scale[0]==1.0 (channel), scale[1]==6.5 (Z), Y/X cumulative."""
    zpath = str(tmp_path / "vol4d.zarr")
    writer = OmeZarrWriter(
        store_path=zpath,
        shape=(2, 4, 16, 16),
        chunk_shape=(1, 1, 16, 16),
        dtype=np.uint16,
        overwrite=True,
        downscale_factor=2,
        unit="micrometer",
    )
    # Distinct frame per (channel, plane): value = 100*(c+1) + z
    frames = {}
    for c in range(2):
        for z in range(4):
            frames[(c, z)] = np.full((16, 16), 100 * (c + 1) + z, dtype=np.uint16)
            writer[c, z, :, :] = frames[(c, z)]
    omero_channels = [
        {
            "label": "555 nm",
            "color": "00FF00",
            "active": True,
            "wavelength": 555,
            "window": {"min": 0, "max": 65535, "start": 0, "end": 65535},
        },
        {
            "label": "647 nm",
            "color": "FF0000",
            "active": True,
            "wavelength": 647,
            "window": {"min": 0, "max": 65535, "start": 0, "end": 65535},
        },
    ]
    writer.finalize(res=(6.5, 6.5, 6.5), n_levels=4, omero_channels=omero_channels)

    nodes = load_zarr(zpath)
    img = nodes[0]
    expected_l0 = np.stack(
        [np.stack([frames[(c, z)] for z in range(4)], axis=0) for c in range(2)], axis=0
    )
    assert len(img.data) == 5  # L0 + 4
    assert np.array_equal(np.asarray(img.data[0]), expected_l0)

    ct = img.metadata["coordinateTransformations"]
    assert len(ct) == 5
    for i in range(5):
        scale = ct[i][0]["scale"]
        assert scale[0] == 1.0  # channel
        assert scale[1] == 6.5  # Z constant
        assert scale[2] == pytest.approx(6.5 * 2**i)
        assert scale[3] == pytest.approx(6.5 * 2**i)

    ome = _root_attrs(zpath)
    assert ome is not None
    assert ome["version"] == "0.5"
    # Read omero.channels via the safe production helper (returns None when
    # ome/omero/channels is absent — guards against KeyError on files from
    # other writers / older NGFF versions / files written without omero).
    channels = load_omero_channels(zpath)
    assert channels is not None
    assert len(channels) == 2
    assert channels[0]["label"] == "555 nm"
    assert channels[0]["color"] == "00FF00"
    assert channels[1]["label"] == "647 nm"
    assert channels[1]["color"] == "FF0000"


def test_omezarrwriter_anisotropic_z(tmp_path):
    """res=(10, 6.5, 6.5) (Z thicker than Y/X) -> per-level scale Z stays
    10, Y/X grow from 6.5 cumulatively."""
    zpath = str(tmp_path / "aniso.zarr")
    writer = OmeZarrWriter(
        store_path=zpath,
        shape=(2, 4, 16, 16),
        chunk_shape=(1, 1, 16, 16),
        dtype=np.uint16,
        overwrite=True,
        downscale_factor=2,
        unit="micrometer",
    )
    writer[:, :, :, :] = 0
    writer.finalize(res=(10.0, 6.5, 6.5), n_levels=2)

    nodes = load_zarr(zpath)
    img = nodes[0]
    ct = img.metadata["coordinateTransformations"]
    assert len(ct) == 3
    for i in range(3):
        scale = ct[i][0]["scale"]
        assert scale[0] == 1.0  # channel
        assert scale[1] == 10.0  # Z anisotropic, stays at base
        assert scale[2] == pytest.approx(6.5 * 2**i)
        assert scale[3] == pytest.approx(6.5 * 2**i)


def test_omezarrwriter_overwrite_false_raises(tmp_path):
    """Pre-existing dir + overwrite=False -> FileExistsError (no silent clobber)."""
    zpath = str(tmp_path / "exists.zarr")
    os.mkdir(zpath)  # pre-create
    with pytest.raises(FileExistsError):
        OmeZarrWriter(
            store_path=zpath,
            shape=(4, 16, 16),
            chunk_shape=(1, 16, 16),
            dtype=np.uint16,
            overwrite=False,
        )


def test_omezarrwriter_validate_n_levels_clamp(tmp_path):
    """shape=(2,4,8,8) (Y/X=8 -> max 3 levels), request n_levels=4 -> only
    4 levels total (L0 + 3), no crash."""
    zpath = str(tmp_path / "clamp.zarr")
    writer = OmeZarrWriter(
        store_path=zpath,
        shape=(2, 4, 8, 8),
        chunk_shape=(1, 1, 8, 8),
        dtype=np.uint16,
        overwrite=True,
        downscale_factor=2,
        unit="micrometer",
    )
    writer[:, :, :, :] = 0
    writer.finalize(res=(6.5, 6.5, 6.5), n_levels=4)

    nodes = load_zarr(zpath)
    img = nodes[0]
    # log2(8) = 3 -> validate_n_levels clamps 4 -> 3 -> 4 total levels (L0+3).
    assert len(img.data) == 4


# ---------------------------------------------------------------------------
# AnalysisOmeZarrWriter — round-trip tests (real zarr in tmp_path)
# ---------------------------------------------------------------------------


def test_analysisomezarrwriter_keeps_raw_l0(tmp_path):
    """L0 == raw frames at scale (1.0,6.5,6.5,6.5); 5 levels total (L0 + 4
    targets); L1..L4 scales == (1.0,10,10,10),(1.0,25,25,25),(1.0,50,50,50),
    (1.0,100,100,100); each target level shape computed from sf=target/base
    per dim with max(1,...) floor."""
    zpath = str(tmp_path / "analysis.zarr")
    writer = AnalysisOmeZarrWriter(
        store_path=zpath,
        shape=(1, 8, 32, 32),
        chunk_shape=(1, 1, 32, 32),
        dtype=np.uint16,
        overwrite=True,
        downscale_factor=2,
        unit="micrometer",
    )
    raw = np.zeros((1, 8, 32, 32), dtype=np.uint16)
    for z in range(8):
        raw[0, z, :, :] = 50 * (z + 1)
        writer[0, z, :, :] = raw[0, z, :, :]
    writer.finalize_with_resolutions(
        base_res=(6.5, 6.5, 6.5),
        target_resolutions_um=(10, 25, 50, 100),
        make_isotropic=True,
    )

    nodes = load_zarr(zpath)
    img = nodes[0]
    # L0 untouched.
    assert np.array_equal(np.asarray(img.data[0]), raw)
    # 5 levels total: L0 + 4 targets.
    assert len(img.data) == 5

    ct = img.metadata["coordinateTransformations"]
    assert len(ct) == 5
    # L0 scale.
    assert ct[0][0]["scale"] == [1.0, 6.5, 6.5, 6.5]
    # L1..L4 scales (isotropic targets).
    expected_targets = [10.0, 25.0, 50.0, 100.0]
    for i, target in enumerate(expected_targets, start=1):
        scale = ct[i][0]["scale"]
        assert scale[0] == 1.0
        assert scale[1] == pytest.approx(target)
        assert scale[2] == pytest.approx(target)
        assert scale[3] == pytest.approx(target)

    # Each target level shape: (1, max(1,int(8/sf_z)), max(1,int(32/sf_y)),
    # max(1,int(32/sf_x))) with sf_d = target/base_res_d (isotropic).
    for i, target in enumerate(expected_targets, start=1):
        sf_z = target / 6.5
        sf_y = target / 6.5
        sf_x = target / 6.5
        expected_shape = (
            1,
            max(1, int(8 / sf_z)),
            max(1, int(32 / sf_y)),
            max(1, int(32 / sf_x)),
        )
        assert tuple(img.data[i].shape) == expected_shape


def test_analysisomezarrwriter_drops_sub_base_targets(tmp_path):
    """base_res=(6.5,6.5,6.5), targets=(5,10,25) -> 5 um dropped (5 < 6.5),
    only 10 and 25 levels appended -> 3 levels total (L0 + 2)."""
    zpath = str(tmp_path / "drop.zarr")
    writer = AnalysisOmeZarrWriter(
        store_path=zpath,
        shape=(1, 8, 32, 32),
        chunk_shape=(1, 1, 32, 32),
        dtype=np.uint16,
        overwrite=True,
        downscale_factor=2,
        unit="micrometer",
    )
    writer[:, :, :, :] = 0
    writer.finalize_with_resolutions(
        base_res=(6.5, 6.5, 6.5),
        target_resolutions_um=(5, 10, 25),
        make_isotropic=True,
    )

    nodes = load_zarr(zpath)
    img = nodes[0]
    # L0 + 2 valid targets (10, 25) — 5 um dropped.
    assert len(img.data) == 3
    ct = img.metadata["coordinateTransformations"]
    assert ct[1][0]["scale"][1] == pytest.approx(10.0)
    assert ct[2][0]["scale"][1] == pytest.approx(25.0)


def test_analysisomezarrwriter_anisotropic_actual_voxel(tmp_path):
    """make_isotropic=False, base_res=(1.5,10,10), target=25 -> sf=25/1.5
    uniformly; scale dict == (1.0, 1.5*sf, 10*sf, 10*sf) ≈ (1.0, 25, 166.67,
    166.67) — ACTUAL voxel, NOT (1.0,25,25,25)."""
    zpath = str(tmp_path / "aniso_analysis.zarr")
    writer = AnalysisOmeZarrWriter(
        store_path=zpath,
        shape=(1, 8, 32, 32),
        chunk_shape=(1, 1, 32, 32),
        dtype=np.uint16,
        overwrite=True,
        downscale_factor=2,
        unit="micrometer",
    )
    writer[:, :, :, :] = 0
    writer.finalize_with_resolutions(
        base_res=(1.5, 10.0, 10.0),
        target_resolutions_um=(25,),
        make_isotropic=False,
    )

    nodes = load_zarr(zpath)
    img = nodes[0]
    assert len(img.data) == 2  # L0 + 1 target
    ct = img.metadata["coordinateTransformations"]
    # L0 scale = (1.0, 1.5, 10, 10) — raw base.
    assert ct[0][0]["scale"] == [1.0, 1.5, 10.0, 10.0]
    # L1 scale = ACTUAL voxel: sf = 25 / min(1.5, 10, 10) = 25/1.5 ≈ 16.667
    sf = 25.0 / 1.5
    scale = ct[1][0]["scale"]
    assert scale[0] == 1.0
    assert scale[1] == pytest.approx(1.5 * sf)  # ≈ 25.0
    assert scale[2] == pytest.approx(10.0 * sf)  # ≈ 166.67
    assert scale[3] == pytest.approx(10.0 * sf)  # ≈ 166.67
    # Sanity: the wrong-data failure mode would be (1.0, 25, 25, 25). Make
    # sure Y/X are NOT 25.
    assert scale[2] != pytest.approx(25.0)
