"""In-process ``main()`` smoke for the ``liom-create-mask`` CLI.

D-01 expansion slice (core-deps-only CLI, no lazy-import guard). Builds a tiny
real OME-Zarr store in ``tmp_path`` via ``save_zarr``, invokes
``liom_create_mask.main()`` in-process with ``sys.argv`` set via
``monkeypatch`` (no subprocess), and asserts the real
``create_and_write_mask`` domain callee writes the ``labels/mask`` group into
the same zarr store. A kwarg-name typo in ``main()``'s call to
``create_and_write_mask`` raises ``TypeError`` against the real signature
(Pitfall 1: do NOT mock the domain callee).

Per D-01: no ``pytest.importorskip``; no ``sys.modules`` heavy-dep mock
(SimpleITK is a core dep). The Dask orchestration client is NOT involved here
(``create_and_write_mask`` does not use the Dask client), so no dask mock is
needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import zarr

from liom_toolkit.conversion.conversion import save_zarr


def test_liom_create_mask_main_smoke(tmp_path: Path, monkeypatch) -> None:
    """``main()`` reaches the real ``create_and_write_mask`` and writes the labels/mask group.

    Builds a tiny real OME-Zarr (16x16x16 uint16 with a bright cube so
    ``segment_3d`` produces a non-trivial watershed mask), invokes ``main()``
    in-process, and asserts the ``labels/mask`` subgroup exists in the zarr
    store after the call. A kwarg-name typo in ``main()``'s call to
    ``create_and_write_mask`` raises ``TypeError`` against the real signature
    before any labels group is written.
    """
    arr = np.zeros((16, 16, 16), dtype=np.uint16)
    arr[4:12, 4:12, 4:12] = 1000
    zarr_in = tmp_path / "in.zarr"
    save_zarr(arr, str(zarr_in), scales=(1.0, 1.0, 1.0), chunks=(16, 16, 16))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "liom-create-mask",
            str(zarr_in),
            "--scales",
            "1.0",
            "1.0",
            "1.0",
            "--chunks",
            "16",
            "16",
            "16",
        ],
    )

    from liom_toolkit.scripts.liom_create_mask import main

    main()

    root = zarr.open(str(zarr_in), mode="r")
    assert "labels" in root, "main() did not create the labels group"
    assert "mask" in root["labels"], (
        "main() did not write the labels/mask subgroup -- the real "
        "create_and_write_mask domain callee was not reached"
    )
