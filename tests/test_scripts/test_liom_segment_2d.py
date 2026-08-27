"""Smoke test for the liom-segment-2d CLI (the TRACER slice).

Exercises the end-to-end path: argparse → basicConfig → segment_2d_image →
output files written. Uses a small synthetic 2D image saved as a TIFF in
``tmp_path`` and invokes the console script via ``uv run`` so the installed
entry point is exercised (not just the in-process function).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np


def test_liom_segment_2d_smoke(tmp_path: Path) -> None:
    """liom-segment-2d on a synthetic 2D image writes the expected output files."""
    # Build a small synthetic 2D image with a bright square on a dark
    # background — the same bimodal pattern used by the bimodal_2d fixture,
    # but constructed inline so this test stays self-contained.
    img = np.zeros((64, 64), dtype=np.uint8)
    img[16:48, 16:48] = 200
    input_file = tmp_path / "input.tif"
    iio.imwrite(str(input_file), img)

    output_dir = tmp_path / "out"
    result = subprocess.run(
        [
            "uv",
            "run",
            "liom-segment-2d",
            str(input_file),
            str(output_dir),
            "--log-level",
            "INFO",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"liom-segment-2d failed: rc={result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    # segment_2d_image writes {name}_mask.tif and {name}_vessel_mask.tif
    # where name defaults to the input file stem.
    assert (output_dir / "input_mask.tif").exists()
    assert (output_dir / "input_vessel_mask.tif").exists()
