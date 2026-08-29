"""In-process ``main()`` smoke for the ``liom-convert-hdf5-to-zarr`` CLI.

This is the D-01 TRACER slice: prove the in-process ``main()`` + real-domain-callee
pattern end-to-end on the simplest CLI (core-deps-only, no lazy-import guard)
before adding the 5 expansion smokes.

The smoke builds a tiny real HDF5 in ``tmp_path`` (h5py is a core dep), invokes
``liom_convert_hdf5_to_zarr.main()`` in-process with ``sys.argv`` set via
``monkeypatch`` (no subprocess), and asserts the real zarr store exists at the
output path. The real ``convert_hdf5_to_zarr`` domain function runs against the
real HDF5 fixture -- a kwarg-name typo in ``main()``'s call to
``convert_hdf5_to_zarr`` raises ``TypeError`` against the real signature
(Pitfall 1: do NOT mock the domain callee, that would absorb the typo).

Per D-01: no ``pytest.importorskip`` (smokes run on core-deps-only CI by
design); no ``sys.modules`` mock (this CLI has no lazy-import guard). The Dask
orchestration client is mocked via the established
``patch("...dask_client_manager")`` pattern (orchestration, not the domain
callee -- mocking it does not absorb a ``convert_hdf5_to_zarr`` kwarg typo).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest


def _make_dask_mock() -> MagicMock:
    """Build a MagicMock dask client whose submit/gather/persist are pass-through.

    Mirrors ``tests/test_conversion/test_conversion.py:_make_dask_mock``: the
    orchestration dependency is mocked while ``h5py``/``dask.array``/``zarr``
    run real (AGENTS section 5).
    """
    mock_client = MagicMock()

    def _submit(fn, *a, **k):
        fut = MagicMock()
        fut.result.return_value = fn(*a, **k)
        return fut

    mock_client.submit.side_effect = _submit
    mock_client.gather.side_effect = lambda fut: fut.result()
    mock_client.persist.side_effect = lambda x: x
    return mock_client


def test_liom_convert_hdf5_to_zarr_main_smoke(tmp_path: Path, monkeypatch) -> None:
    """``main()`` reaches the real ``convert_hdf5_to_zarr`` and writes a zarr store.

    Builds a tiny real HDF5 (single ``channel_0`` dataset, 8x8x8 float32),
    invokes ``main()`` in-process, and asserts the output zarr store exists.
    A kwarg-name typo in ``main()``'s call to ``convert_hdf5_to_zarr`` raises
    ``TypeError`` against the real signature before any zarr is written.
    """
    hdf5_in = tmp_path / "in.h5"
    with h5py.File(str(hdf5_in), "w") as f:
        f.create_dataset("channel_0", data=np.zeros((8, 8, 8), dtype=np.float32))

    zarr_out = tmp_path / "out.zarr"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "liom-convert-hdf5-to-zarr",
            str(hdf5_in),
            str(zarr_out),
            "--scales",
            "1.0",
            "1.0",
            "1.0",
            "--chunks",
            "8",
            "8",
            "8",
        ],
    )

    from liom_toolkit.scripts.liom_convert_hdf5_to_zarr import main

    # Mock the Dask orchestration client (NOT the domain callee) so the smoke
    # does not spin up a real LocalCluster inside pytest-xdist. The real
    # convert_hdf5_to_zarr runs against the real HDF5 fixture.
    with patch("liom_toolkit.conversion.conversion.dask_client_manager") as mgr:
        mgr.get_client.return_value = _make_dask_mock()
        main()

    assert zarr_out.exists(), (
        f"main() did not write the zarr store at {zarr_out} -- the real "
        "convert_hdf5_to_zarr domain callee was not reached"
    )


def test_liom_convert_hdf5_to_zarr_missing_input_exits_2(tmp_path: Path, monkeypatch) -> None:
    """A nonexistent input_file path exits 2 with a clear CLI error.

    Without a file-existence check at the argparse boundary, ``main()`` reaches
    ``convert_hdf5_to_zarr`` which opens the HDF5 file via h5py and raises a
    raw third-party traceback instead of the argparse-style exit code 2. This
    regression test pins the D-01 file-existence mitigation: the check
    surfaces the missing path via ``parser.error`` before any Dask client
    setup.
    """
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "liom-convert-hdf5-to-zarr",
            str(tmp_path / "nope.h5"),
            str(tmp_path / "out.zarr"),
        ],
    )

    from liom_toolkit.scripts.liom_convert_hdf5_to_zarr import main

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 2, (
        "main() should exit 2 via parser.error on a nonexistent input_file, "
        f"got exit code {exc.value.code}"
    )
