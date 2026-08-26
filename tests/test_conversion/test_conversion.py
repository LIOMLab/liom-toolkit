"""Conversion-correctness tests for ``liom_toolkit/conversion/conversion.py``.

Mirrors the package layout (``tests/test_conversion/test_conversion.py``) and
covers the three format-conversion entry points plus the ``save_zarr``
overwrite behavior:

* ``convert_hdf5_to_zarr`` — real HDF5→Zarr round-trip with the Dask client
  mocked per the established MagicMock pattern (the orchestration dependency
  is mocked; ``h5py``/``dask.array``/``zarr`` are real and unmocked, per
  AGENTS.md §5). The ``load_hdf5`` 4D-rechunk ``ValueError`` (a characterized
  pre-migration bug — ``da.rechunk(data, chunks=(128, 128, 128))`` hardcoded
  for 3D data but ``da.stack`` always produces 4D) was previously dodged via
  a test-local ``monkeypatch.setattr`` that reimplemented ``load_hdf5`` with
  4D-aware rechunk. That workaround was removed once the real function's
  rechunk was made 4D-aware (the rechunk now picks ``(1, 128, 128, 128)`` for
  4D stacked data).
* ``convert_hdf5_to_zarr(use_memmap=True)`` — the default-arg path hits the
  dead ``os.remove("temp.dat")`` on a file never written and raises
  ``FileNotFoundError``. Characterized via ``xfail(strict=True,
  raises=FileNotFoundError)`` so the Phase-6 fix forces marker-removal.
* ``convert_nifti_to_zarr`` / ``convert_nrrd_to_zarr`` — real
  ``nib.load`` / ``nrrd.read`` round-trips (no Dask client involved).
* ``save_zarr`` overwrite — both ``overwrite=False`` (fresh directory) and
  ``overwrite=True`` (second call into an existing directory) must succeed;
  the second call overwrites the first via the symlink-aware
  ``create_directory(overwrite=True)`` helper, and the overwritten data
  round-trips via ``load_zarr``.

The Dask-mock pattern (``patch("liom_toolkit.conversion.conversion.dask_client_manager")``)
patches at the import site inside ``conversion.py`` — NOT at
``liom_toolkit.utils.dask_client.dask_client_manager`` — because that is where
``load_hdf5`` looks up the manager at call time.
"""

from __future__ import annotations

import h5py
import nibabel as nib
import nrrd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from liom_toolkit.conversion.conversion import (
    convert_hdf5_to_zarr,
    convert_nifti_to_zarr,
    convert_nrrd_to_zarr,
    save_zarr,
)
from liom_toolkit.utils.io import load_zarr


def _make_synthetic_hdf5(path, n_channels=1, shape=(16, 16, 16)):
    """Write a tiny HDF5 file with ``n_channels`` datasets named ``channel_{i}``.

    Each channel holds ``arr * i + 100`` where ``arr`` is a uint16 volume with
    a bright cube at ``[4:12, 4:12, 4:12]``. Returns the base ``arr`` so the
    caller can compute the expected stacked volume.
    """
    arr = np.zeros(shape, dtype=np.uint16)
    arr[4:12, 4:12, 4:12] = 1000
    with h5py.File(path, "w") as f:
        for i in range(n_channels):
            f.create_dataset(f"channel_{i}", data=arr * i + 100)
    return arr


def _make_dask_mock():
    """Build a MagicMock dask client whose submit/gather/persist are pass-through.

    ``submit`` calls the given function for real (so ``da.from_array`` produces
    a real dask array) and wraps the result in a MagicMock future whose
    ``.result()`` returns that real result. ``gather`` unwraps the future via
    ``fut.result()``. ``persist`` is pass-through. This is the established
    TESTING.md / D-04 pattern — mock the orchestration dependency, not the
    compute deps.
    """
    mock_client = MagicMock()

    def _submit(fn, *a, **k):
        fut = MagicMock()
        fut.result.return_value = fn(*a, **k)  # call da.from_array for real
        return fut

    mock_client.submit.side_effect = _submit
    mock_client.gather.side_effect = lambda fut: fut.result()  # unwrap -> dask array
    mock_client.persist.side_effect = lambda x: x  # pass-through
    return mock_client


def test_convert_hdf5_to_zarr_round_trip(tmp_path):
    """convert_hdf5_to_zarr(use_memmap=False) round-trips a 3-channel HDF5.

    The Dask client is mocked (D-04); the real ``load_hdf5`` is exercised
    (its rechunk is now 4D-aware). Asserts level-0 data equality, shape,
    dtype, pyramid level count, and the submit→gather→persist call sequence
    on the mock (one submit+gather pair per channel, one persist after
    rechunk).
    """
    arr = _make_synthetic_hdf5(str(tmp_path / "synth.h5"), n_channels=3)
    mock_client = _make_dask_mock()
    zpath = str(tmp_path / "out.zarr")

    with patch("liom_toolkit.conversion.conversion.dask_client_manager") as mgr:
        mgr.get_client.return_value = mock_client
        convert_hdf5_to_zarr(
            str(tmp_path / "synth.h5"),
            zpath,
            use_memmap=False,
            scales=(6.5, 6.5, 6.5),
            chunks=(16, 16, 16),
        )

    nodes = load_zarr(zpath)
    img = nodes[0]
    expected = np.stack([arr * 0 + 100, arr * 1 + 100, arr * 2 + 100], axis=0)
    assert np.array_equal(np.asarray(img.data[0]), expected)
    assert img.data[0].shape == (3, 16, 16, 16)
    assert img.data[0].dtype == np.uint16
    assert len(img.data) == 5  # 5 pyramid levels (CustomScaler max_layer=4)

    # Mock call-sequence assertion (D-04): characterizes today's load_hdf5
    # orchestration — one submit+gather pair per channel, one persist after
    # rechunk. If load_hdf5's internal orchestration changes, these counts
    # need updating.
    assert mock_client.submit.call_count == 3
    assert mock_client.gather.call_count == 3
    assert mock_client.persist.call_count == 1


@pytest.mark.xfail(
    strict=True,
    raises=FileNotFoundError,
    reason="BUG-01: use_memmap dead code — os.remove('temp.dat') on a file never written",
)
def test_convert_hdf5_use_memmap_true_raises(tmp_path, monkeypatch):
    """convert_hdf5_to_zarr(use_memmap=True) raises FileNotFoundError.

    The default-arg ``use_memmap=True`` path ends with
    ``os.remove("temp.dat")`` on a file that was never written (the
    ``map_file``/``map_file`` memmap code is dead — ``temp.dat`` is never
    created). Characterized via strict xfail so the Phase-6 fix (removing the
    dead code) forces marker-removal: an xpass becomes a hard failure.
    """
    # chdir into tmp_path so the relative "temp.dat" in os.remove resolves into
    # the clean temp directory — without this, a stray temp.dat in the pytest
    # CWD would be deleted and the test would XPASS (a hard failure under
    # strict=True) instead of characterizing the FileNotFoundError bug.
    monkeypatch.chdir(tmp_path)

    arr = np.zeros((16, 16, 16), dtype=np.uint16)
    arr[4:12, 4:12, 4:12] = 1000
    h5path = str(tmp_path / "synth.h5")
    with h5py.File(h5path, "w") as f:
        f.create_dataset("channel_0", data=arr)

    mock_client = _make_dask_mock()
    with patch("liom_toolkit.conversion.conversion.dask_client_manager") as mgr:
        mgr.get_client.return_value = mock_client
        convert_hdf5_to_zarr(
            h5path,
            str(tmp_path / "out.zarr"),
            use_memmap=True,
            scales=(6.5, 6.5, 6.5),
            chunks=(16, 16, 16),
        )


def test_convert_nifti_to_zarr_round_trip(tmp_path):
    """convert_nifti_to_zarr round-trips a real NIfTI via nib.load.

    No Dask client is involved — ``convert_nifti_to_zarr`` uses
    ``da.from_array(nib.load(...).get_fdata())`` directly. Note the production
    signature typo ``chucks=`` (not ``chunks=``) is matched verbatim.
    """
    arr = np.zeros((16, 16, 16), dtype=np.uint16)
    arr[4:12, 4:12, 4:12] = 1000
    npath = str(tmp_path / "synth.nii.gz")
    nib.save(nib.Nifti1Image(arr, affine=np.eye(4)), npath)

    zpath = str(tmp_path / "nifti.zarr")
    convert_nifti_to_zarr(npath, zpath, scales=(6.5, 6.5, 6.5), chucks=(16, 16, 16))

    nodes = load_zarr(zpath)
    assert np.array_equal(np.asarray(nodes[0].data[0]), arr)
    assert len(nodes[0].data) == 5
    # NOTE: nibabel.get_fdata() returns float64, so the level-0 dtype is
    # float64 (not uint16). This test asserts data-value equality only and
    # intentionally does NOT assert dtype == np.uint16 for the NIfTI path.


def test_convert_nrrd_to_zarr_round_trip(tmp_path):
    """convert_nrrd_to_zarr round-trips a real NRRD via nrrd.read.

    NRRD preserves the uint16 dtype (unlike NIfTI's float64 get_fdata), so
    both data equality AND dtype are asserted. Note the production signature
    typo ``chucks=`` is matched verbatim.
    """
    arr = np.zeros((16, 16, 16), dtype=np.uint16)
    arr[4:12, 4:12, 4:12] = 1000
    npath = str(tmp_path / "synth.nrrd")
    nrrd.write(npath, arr)

    zpath = str(tmp_path / "nrrd.zarr")
    convert_nrrd_to_zarr(npath, zpath, scales=(6.5, 6.5, 6.5), chucks=(16, 16, 16))

    nodes = load_zarr(zpath)
    assert np.array_equal(np.asarray(nodes[0].data[0]), arr)
    assert nodes[0].data[0].dtype == np.uint16
    assert len(nodes[0].data) == 5


@pytest.mark.parametrize("overwrite", [False, True])
def test_save_zarr_overwrite(tmp_path, overwrite):
    """save_zarr fresh-directory and overwrite paths both succeed.

    ``save_zarr`` uses the symlink-aware ``create_directory`` helper from
    ``utils/zarr_writer.py`` with ``overwrite=True``: a second call into an
    existing zarr store directory ``rmtree``'s the store then recreates it
    before the zarr write proceeds (zarr stores are directories with
    subdirectories, so ``shutil.rmtree`` is the correct clearing primitive,
    not ``os.remove`` which only handles flat files). Both the
    ``overwrite=False`` (fresh directory) and ``overwrite=True`` (second call
    into an existing directory) branches must succeed without
    ``FileExistsError``; the overwritten data must round-trip via
    ``load_zarr`` and match the second write, not the first.
    """
    data = np.zeros((16, 16, 16), dtype=np.uint16)
    zpath = str(tmp_path / "vol.zarr")
    save_zarr(data, zpath, scales=(6.5, 6.5, 6.5), chunks=(16, 16, 16))
    if overwrite:
        # Second call into the existing directory must succeed (overwrite-safe
        # via create_directory(overwrite=True)). Write a distinct volume so
        # the round-trip assertion can confirm the second write replaced the
        # first rather than silently leaving stale data.
        overwritten = np.full((16, 16, 16), 7, dtype=np.uint16)
        save_zarr(overwritten, zpath, scales=(6.5, 6.5, 6.5), chunks=(16, 16, 16))
        nodes = load_zarr(zpath)
        assert np.array_equal(np.asarray(nodes[0].data[0]), overwritten)
