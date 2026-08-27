"""Smoke test for the ``liom-build-template`` CLI.

Exercises ``--help`` exits 0 and contains the 4 shared flags + the curated
template-building flags (output_file, zarr_files, brain_names,
--resolution_level, --template_resolution, --iterations, --atlas_resolution).
"""

from __future__ import annotations

import subprocess


def test_liom_build_template_help_exits_0() -> None:
    """liom-build-template --help exits 0 with shared + curated flags."""
    result = subprocess.run(
        ["uv", "run", "liom-build-template", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"liom-build-template --help failed: rc={result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    out = result.stdout + result.stderr
    for flag in ("--log-level", "--resume", "--dask_scheduler", "--n_workers"):
        assert flag in out, f"liom-build-template --help missing {flag}"
    for flag in ("output_file", "zarr_files", "brain_names"):
        assert flag in out, f"liom-build-template --help missing {flag}"
    for flag in (
        "--resolution_level",
        "--template_resolution",
        "--iterations",
        "--atlas_resolution",
    ):
        assert flag in out, f"liom-build-template --help missing {flag}"
