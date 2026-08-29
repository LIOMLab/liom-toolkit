"""Smoke + in-process ``main()`` test for the ``liom-compute-slice-metrics`` CLI.

Exercises:
* ``--help`` exits 0 and contains the 4 shared flags + the curated stats flags
  (output_dir, image, --voxel-size).
* ``main()`` reaches the real ``compute_slice_metrics`` domain callee and
  writes ``regions.xlsx`` (and ``regions.png``) via the real openpyxl path
  (D-03 coordination -- this smoke also exercises the real ``df.to_excel``
  call, no monkeypatch). A kwarg-name typo in ``main()``'s call to
  ``compute_slice_metrics`` raises ``TypeError`` against the real signature.

Per D-01: no ``pytest.importorskip``; no ``sys.modules`` heavy-dep mock
(core-deps-only CLI). Per AGENTS section 5, ``imageio``/``numpy``/``pandas``/
``openpyxl`` are NOT mocked -- the smoke writes real TIFFs and reads the real
xlsx output.

The ``--help`` smoke check invokes the script's ``_build_argument_parser()``
in-process and formats its help text, rather than spawning a ``uv run``
subprocess (avoids the per-invocation venv-reconcile + interpreter-startup
cost).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest


def test_liom_compute_slice_metrics_help_exits_0() -> None:
    """liom-compute-slice-metrics --help contains shared + curated flags."""
    from liom_toolkit.scripts.liom_compute_slice_metrics import (
        _build_argument_parser,
    )

    out = _build_argument_parser().format_help()
    for flag in ("--log-level", "--resume", "--dask-scheduler", "--n-workers"):
        assert flag in out, f"liom-compute-slice-metrics --help missing {flag}"
    for flag in ("output_dir", "image", "--voxel-size"):
        assert flag in out, f"liom-compute-slice-metrics --help missing {flag}"


def test_liom_compute_slice_metrics_main_smoke(tmp_path: Path, monkeypatch) -> None:
    """``main()`` reaches the real ``compute_slice_metrics`` and writes regions.xlsx.

    Builds 4 tiny real TIFFs (mask, vessel_mask, region_map, vessel_exclude)
    using the synthetic arrays from ``test_stats.py`` (mask=ones, vessel_mask
    with a bright block, region_map=ones, vessel_exclude=ones), invokes
    ``main()`` in-process, and asserts ``regions.xlsx`` exists in the output
    directory. The real ``df.to_excel`` -> openpyxl path runs (no monkeypatch),
    coordinating with D-03. A kwarg-name typo in ``main()``'s call to
    ``compute_slice_metrics`` raises ``TypeError`` against the real signature.
    """
    mask = np.ones((30, 30), dtype=np.uint8)
    vessel_mask = np.zeros((30, 30), dtype=np.uint8)
    vessel_mask[10:20, 10:20] = 1
    region_map = np.ones((30, 30), dtype=np.uint8)
    vessel_exclude = np.ones((30, 30), dtype=np.uint8)

    mask_path = tmp_path / "mask.tif"
    vessel_path = tmp_path / "vessel.tif"
    region_path = tmp_path / "region.tif"
    exclude_path = tmp_path / "exclude.tif"
    iio.imwrite(str(mask_path), mask)
    iio.imwrite(str(vessel_path), vessel_mask)
    iio.imwrite(str(region_path), region_map)
    iio.imwrite(str(exclude_path), vessel_exclude)

    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "liom-compute-slice-metrics",
            str(out_dir),
            "label",
            str(mask_path),
            str(vessel_path),
            str(region_path),
            str(exclude_path),
            "--voxel-size",
            "0.65",
        ],
    )

    from liom_toolkit.scripts.liom_compute_slice_metrics import main

    main()

    assert os.path.isfile(str(out_dir / "regions.xlsx")), (
        "main() did not write regions.xlsx -- the real compute_slice_metrics "
        "domain callee (and its df.to_excel -> openpyxl path) was not reached"
    )
    assert os.path.isfile(str(out_dir / "regions.png")), (
        "main() did not write regions.png -- the real compute_slice_metrics "
        "domain callee was not reached"
    )


def test_liom_compute_slice_metrics_missing_input_exits_2(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """A nonexistent mask_file path exits 2 with a clear CLI error.

    Without a file-existence check at the argparse boundary, ``main()`` reaches
    ``imageio.imread`` and raises ``FileNotFoundError`` (a raw third-party
    traceback) instead of the argparse-style exit code 2. This regression test
    pins the D-01 file-existence mitigation: the check loops over the 4 input
    positionals and surfaces the first missing one via ``parser.error`` before
    any image is loaded.
    """
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "liom-compute-slice-metrics",
            str(tmp_path / "out"),
            "label",
            str(tmp_path / "nope.tif"),
            str(tmp_path / "nope_vessel.tif"),
            str(tmp_path / "nope_region.tif"),
            str(tmp_path / "nope_exclude.tif"),
            "--voxel-size",
            "0.65",
        ],
    )

    from liom_toolkit.scripts.liom_compute_slice_metrics import main

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 2, (
        "main() should exit 2 via parser.error on a nonexistent mask_file, "
        f"got exit code {exc.value.code}"
    )
    captured = capsys.readouterr()
    assert "does not exist" in captured.err
    assert str(tmp_path / "nope.tif") in captured.err
