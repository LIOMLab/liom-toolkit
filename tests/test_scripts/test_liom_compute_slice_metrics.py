"""Smoke test for the ``liom-compute-slice-metrics`` CLI.

Exercises ``--help`` exits 0 and contains the 4 shared flags + the curated
stats flags (output_dir, image, --voxel_size).
"""

from __future__ import annotations

import subprocess


def test_liom_compute_slice_metrics_help_exits_0() -> None:
    """liom-compute-slice-metrics --help exits 0 with shared + curated flags."""
    result = subprocess.run(
        ["uv", "run", "liom-compute-slice-metrics", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"liom-compute-slice-metrics --help failed: rc={result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    out = result.stdout + result.stderr
    for flag in ("--log-level", "--resume", "--dask_scheduler", "--n_workers"):
        assert flag in out, f"liom-compute-slice-metrics --help missing {flag}"
    for flag in ("output_dir", "image", "--voxel_size"):
        assert flag in out, f"liom-compute-slice-metrics --help missing {flag}"
