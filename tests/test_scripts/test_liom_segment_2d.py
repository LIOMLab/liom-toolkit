"""Smoke test for the liom-segment-2d CLI (the TRACER slice).

Exercises the end-to-end path: argparse → basicConfig → segment_2d_image →
output files written. Uses a small synthetic 2D image saved as a TIFF in
``tmp_path`` and invokes ``liom_segment_2d.main()`` in-process with
``sys.argv`` set via ``monkeypatch`` (no subprocess) — the same pattern used
by ``test_liom_create_mask.py``. This avoids the ``uv run`` subprocess
overhead (~3s for venv-reconcile + interpreter startup) while exercising the
real ``main()`` → ``segment_2d_image`` path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np


def test_liom_segment_2d_smoke(tmp_path: Path, monkeypatch) -> None:
    """liom-segment-2d on a synthetic 2D image writes the expected output files."""
    # Build a small synthetic 2D image with a bright square on a dark
    # background — the same bimodal pattern used by the bimodal_2d fixture,
    # but constructed inline so this test stays self-contained.
    img = np.zeros((64, 64), dtype=np.uint8)
    img[16:48, 16:48] = 200
    input_file = tmp_path / "input.tif"
    iio.imwrite(str(input_file), img)

    output_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "liom-segment-2d",
            str(input_file),
            str(output_dir),
            "--log-level",
            "INFO",
        ],
    )

    from liom_toolkit.scripts.liom_segment_2d import main

    main()

    # segment_2d_image writes {name}_mask.tif and {name}_vessel_mask.tif
    # where name defaults to the input file stem.
    assert (output_dir / "input_mask.tif").exists()
    assert (output_dir / "input_vessel_mask.tif").exists()
