"""In-process ``main()`` smoke + edge-case tests for the ``liom-create-mask`` CLI.

D-01 expansion slice (core-deps-only CLI, no lazy-import guard). The smoke
test builds a tiny real OME-Zarr store in ``tmp_path`` via ``save_zarr``,
invokes ``liom_create_mask.main()`` in-process with ``sys.argv`` set via
``monkeypatch`` (no subprocess), and asserts the real
``create_and_write_mask`` domain callee writes the ``labels/mask`` group into
the same zarr store. A kwarg-name typo in ``main()``'s call to
``create_and_write_mask`` raises ``TypeError`` against the real signature
(Pitfall 1: do NOT mock the domain callee).

The edge-case test asserts the D-01 file-existence validator: a nonexistent
``input_file`` must exit 2 with a clear message instead of a raw zarr
traceback. The help smoke asserts the D-02 hyphenated ``--fill-holes`` flag
(argparse auto-derives ``dest``, so ``main()`` reads ``args.fill_holes``
unchanged through the rename).

Per D-01: no ``pytest.importorskip``; no ``sys.modules`` heavy-dep mock
(SimpleITK is a core dep). The Dask orchestration client is NOT involved here
(``create_and_write_mask`` does not use the Dask client), so no dask mock is
needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
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


def test_liom_create_mask_missing_input_exits_2(tmp_path: Path, monkeypatch) -> None:
    """A nonexistent input_file exits 2 with a clear message, not a raw zarr traceback.

    Without a file-existence check at the argparse boundary, ``main()`` reaches
    ``create_and_write_mask`` which raises a confusing zarr traceback for a CLI
    user. The validator must surface the offending path in the exit-2 message.
    """
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "liom-create-mask",
            str(tmp_path / "nope.zarr"),
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

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


def test_liom_create_mask_help_exits_0() -> None:
    """--help shows the hyphenated --fill-holes flag (D-02), not the snake_case form.

    argparse auto-derives ``dest`` (hyphen to underscore), so ``main()`` reads
    ``args.fill_holes`` unchanged through the rename. This test guards against
    a regression that reverts the flag string to snake_case.
    """
    from liom_toolkit.scripts.liom_create_mask import _build_argument_parser

    out = _build_argument_parser().format_help()
    assert "--fill-holes" in out, "expected hyphenated --fill-holes flag in help"
    assert "--log-level" in out, "shared --log-level flag missing from help"
    assert "--resume" in out, "shared --resume flag missing from help"
