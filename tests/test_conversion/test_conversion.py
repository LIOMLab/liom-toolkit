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
* ``convert_hdf5_to_zarr`` no longer accepts a ``use_memmap`` parameter — the
  dead ``os.remove("temp.dat")`` path that crashed with ``FileNotFoundError``
  has been removed (clean break, no deprecation shim). The default call now
  succeeds and writes a valid zarr; passing ``use_memmap=`` raises
  ``TypeError``. The CLI ``--use_memmap`` flag is gone too.
* ``convert_nifti_to_zarr`` / ``convert_nrrd_to_zarr`` — real
  ``nib.load`` / ``nrrd.read`` round-trips (no Dask client involved). The
  misspelled ``chucks=`` parameter has been hard-renamed to ``chunks=``;
  passing ``chucks=`` now raises ``TypeError``.
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

from unittest.mock import MagicMock, patch

import h5py
import nibabel as nib
import nrrd
import numpy as np
import pytest

from liom_toolkit.conversion.conversion import (
    convert_hdf5_to_zarr,
    convert_nifti_to_zarr,
    convert_nrrd_to_zarr,
    load_hdf5,
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
    """convert_hdf5_to_zarr round-trips a 3-channel HDF5.

    The Dask client is mocked; the real ``load_hdf5`` is exercised (its
    rechunk is now 4D-aware). Asserts level-0 data equality, shape, dtype,
    pyramid level count, and the submit→gather→persist call sequence on the
    mock (one submit+gather pair per channel, one persist after rechunk).
    """
    arr = _make_synthetic_hdf5(str(tmp_path / "synth.h5"), n_channels=3)
    mock_client = _make_dask_mock()
    zpath = str(tmp_path / "out.zarr")

    with patch("liom_toolkit.conversion.conversion.dask_client_manager") as mgr:
        mgr.get_client.return_value = mock_client
        convert_hdf5_to_zarr(
            str(tmp_path / "synth.h5"),
            zpath,
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


def test_convert_hdf5_to_zarr_no_use_memmap_succeeds(tmp_path):
    """convert_hdf5_to_zarr (no use_memmap kwarg) succeeds and writes a valid zarr.

    The dead ``use_memmap``/``map_file`` path (which crashed with
    ``FileNotFoundError`` on ``os.remove("temp.dat")`` against a file never
    written) has been removed. The default call now writes the zarr and
    returns normally; the level-0 data has the expected shape and a non-zero
    block. This test fails RED against the current source because the
    function still has ``use_memmap: bool = True`` as a default and still
    calls ``os.remove("temp.dat")`` → ``FileNotFoundError`` on the default
    path.
    """
    arr = np.zeros((16, 16, 16), dtype=np.uint16)
    arr[4:12, 4:12, 4:12] = 1000
    h5path = str(tmp_path / "synth.h5")
    with h5py.File(h5path, "w") as f:
        f.create_dataset("channel_0", data=arr)

    mock_client = _make_dask_mock()
    out = str(tmp_path / "out.zarr")
    with patch("liom_toolkit.conversion.conversion.dask_client_manager") as mgr:
        mgr.get_client.return_value = mock_client
        convert_hdf5_to_zarr(
            h5path,
            out,
            scales=(6.5, 6.5, 6.5),
            chunks=(16, 16, 16),
        )

    nodes = load_zarr(out)
    assert np.asarray(nodes[0].data[0]).shape == (1, 16, 16, 16)
    assert np.asarray(nodes[0].data[0]).any()  # non-zero block present


def test_convert_nifti_to_zarr_round_trip(tmp_path):
    """convert_nifti_to_zarr round-trips a real NIfTI via nib.load.

    No Dask client is involved — ``convert_nifti_to_zarr`` loads the NIfTI
    data array directly via ``np.asanyarray(nib.load(...).dataobj)``, which
    preserves the stored dtype (uint16 stays uint16 — no float64 upcast).
    """
    arr = np.zeros((16, 16, 16), dtype=np.uint16)
    arr[4:12, 4:12, 4:12] = 1000
    npath = str(tmp_path / "synth.nii.gz")
    nib.save(nib.Nifti1Image(arr, affine=np.eye(4)), npath)

    zpath = str(tmp_path / "nifti.zarr")
    convert_nifti_to_zarr(npath, zpath, scales=(6.5, 6.5, 6.5), chunks=(16, 16, 16))

    nodes = load_zarr(zpath)
    assert np.array_equal(np.asarray(nodes[0].data[0]), arr)
    assert len(nodes[0].data) == 5
    # The dtype is preserved from the stored NIfTI (uint16 stays uint16) —
    # np.asanyarray(ni_img.dataobj) does NOT upcast to float64 the way
    # get_fdata() does.
    assert nodes[0].data[0].dtype == np.uint16


def test_convert_nifti_to_zarr_preserves_dtype(tmp_path):
    """convert_nifti_to_zarr preserves the input NIfTI's stored dtype.

    A uint16 NIfTI must produce a uint16 zarr — NOT float64. The previous
    ``get_fdata()`` call upcast every integer dtype to float64 (4x storage
    inflation + dtype loss); ``np.asanyarray(ni_img.dataobj)`` preserves the
    stored dtype exactly. Fails RED against the current source because the
    function still calls ``get_fdata()``.
    """
    arr = np.zeros((16, 16, 16), dtype=np.uint16)
    arr[4:12, 4:12, 4:12] = 1000
    npath = str(tmp_path / "synth_uint16.nii.gz")
    nib.save(nib.Nifti1Image(arr, affine=np.eye(4)), npath)

    zpath = str(tmp_path / "nifti_uint16.zarr")
    convert_nifti_to_zarr(npath, zpath, scales=(6.5, 6.5, 6.5), chunks=(16, 16, 16))

    nodes = load_zarr(zpath)
    assert nodes[0].data[0].dtype == np.uint16


def test_convert_nifti_to_zarr_no_float64_upcast(tmp_path):
    """convert_nifti_to_zarr does NOT upcast integer NIfTI to float64.

    Regression guard for the ``get_fdata()`` upcast: a uint16 input must
    produce a uint16 zarr, never float64. Fails RED against the current
    source because ``get_fdata()`` always returns float64.
    """
    arr = np.zeros((16, 16, 16), dtype=np.uint16)
    arr[4:12, 4:12, 4:12] = 1000
    npath = str(tmp_path / "synth_no_upcast.nii.gz")
    nib.save(nib.Nifti1Image(arr, affine=np.eye(4)), npath)

    zpath = str(tmp_path / "nifti_no_upcast.zarr")
    convert_nifti_to_zarr(npath, zpath, scales=(6.5, 6.5, 6.5), chunks=(16, 16, 16))

    nodes = load_zarr(zpath)
    assert nodes[0].data[0].dtype != np.float64


def test_convert_nrrd_to_zarr_round_trip(tmp_path):
    """convert_nrrd_to_zarr round-trips a real NRRD via nrrd.read.

    NRRD preserves the uint16 dtype (unlike NIfTI's float64 get_fdata), so
    both data equality AND dtype are asserted.
    """
    arr = np.zeros((16, 16, 16), dtype=np.uint16)
    arr[4:12, 4:12, 4:12] = 1000
    npath = str(tmp_path / "synth.nrrd")
    nrrd.write(npath, arr)

    zpath = str(tmp_path / "nrrd.zarr")
    convert_nrrd_to_zarr(npath, zpath, scales=(6.5, 6.5, 6.5), chunks=(16, 16, 16))

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


def test_convert_nifti_to_zarr_rejects_chucks_kwarg(tmp_path):
    """convert_nifti_to_zarr rejects the misspelled ``chucks=`` kwarg.

    The parameter has been hard-renamed ``chucks``→``chunks`` (clean break,
    no deprecation shim). Passing the old spelling as a keyword must raise
    ``TypeError`` rather than being silently accepted or aliased. Fails RED
    against the current source because the signature still spells the
    parameter ``chucks=`` (no TypeError raised).
    """
    arr = np.zeros((16, 16, 16), dtype=np.uint16)
    arr[4:12, 4:12, 4:12] = 1000
    npath = str(tmp_path / "synth.nii.gz")
    nib.save(nib.Nifti1Image(arr, affine=np.eye(4)), npath)

    with pytest.raises(TypeError):
        convert_nifti_to_zarr(
            npath,
            str(tmp_path / "out.zarr"),
            scales=(6.5, 6.5, 6.5),
            chucks=(16, 16, 16),
        )


def test_convert_hdf5_to_zarr_rejects_use_memmap_kwarg(tmp_path):
    """convert_hdf5_to_zarr rejects the removed ``use_memmap=`` kwarg.

    The dead ``use_memmap``/``map_file`` path has been removed (clean break,
    no deprecation shim). Passing ``use_memmap=`` must raise ``TypeError``
    rather than being silently ignored. Fails RED against the current source
    because the signature still accepts ``use_memmap=``.
    """
    arr = np.zeros((16, 16, 16), dtype=np.uint16)
    arr[4:12, 4:12, 4:12] = 1000
    h5path = str(tmp_path / "synth.h5")
    with h5py.File(h5path, "w") as f:
        f.create_dataset("channel_0", data=arr)

    mock_client = _make_dask_mock()
    with patch("liom_toolkit.conversion.conversion.dask_client_manager") as mgr:
        mgr.get_client.return_value = mock_client
        with pytest.raises(TypeError):
            convert_hdf5_to_zarr(
                h5path,
                str(tmp_path / "out.zarr"),
                use_memmap=True,
                scales=(6.5, 6.5, 6.5),
                chunks=(16, 16, 16),
            )


def test_cli_help_has_no_use_memmap_flag():
    """The HDF5→Zarr CLI parser does not register a ``--use_memmap`` flag.

    The dead ``--use_memmap`` CLI flag has been removed alongside the
    library parameter. Inspecting the parser's registered actions must find
    no option whose ``option_strings`` contain ``--use_memmap``. Fails RED
    against the current source because the parser still registers the flag.
    """
    from liom_toolkit.scripts.liom_convert_hdf5_to_zarr import _build_argument_parser

    parser = _build_argument_parser()
    for action in parser._actions:
        assert "--use_memmap" not in action.option_strings


def test_load_hdf5_no_file_descriptor_leak(tmp_path):
    """Repeated load_hdf5 calls do not leak OS file descriptors.

    This is a GREEN-verify regression test for the prior-phase
    ``with h5py.File(hdf5_file, "r") as f:`` context-manager fix in
    ``load_hdf5``: the context manager must release the OS file descriptor
    on every exit path (success or exception), so 50 repeated calls on the
    same HDF5 file must leave the process open-fd count unchanged. A failure
    here means the prior fix regressed (a real bug to investigate), not that
    this test drives new behavior.

    The fd-count measurement uses ``psutil.Process().num_fds()``; if psutil
    is not installed the test is skipped (no weaker fallback that could
    silently pass against a regressed fix — per the no-silent-wrong-data
    rule applied to tests). The Dask client is mocked via the existing
    ``_make_dask_mock`` helper (orchestration mock only); ``h5py``,
    ``dask.array``, and ``numpy`` are real and unmocked, per AGENTS.md §5.
    Each iteration rebinds the result (``_ = load_hdf5(...)``) so no
    cross-iteration reference pins the dask array and masks a genuine leak.
    """
    psutil = pytest.importorskip("psutil")  # fd-count mechanism; skip if absent

    h5path = str(tmp_path / "leak.h5")
    _make_synthetic_hdf5(h5path, n_channels=1, shape=(8, 8, 8))
    mock_client = _make_dask_mock()

    fd_before = psutil.Process().num_fds()
    with patch("liom_toolkit.conversion.conversion.dask_client_manager") as mgr:
        mgr.get_client.return_value = mock_client
        for _ in range(50):
            # Rebind each iteration so no cross-iteration reference pins the
            # returned dask array and hides a real fd leak in the count.
            _ = load_hdf5(h5path)
    fd_after = psutil.Process().num_fds()

    assert fd_after == fd_before, (
        f"load_hdf5 leaked OS file descriptors: {fd_before} before -> "
        f"{fd_after} after 50 repeated calls (the with h5py.File context "
        "manager must release the fd on every exit path)"
    )
