"""Smoke + ImportError-surfacing tests for the ``liom-align-annotations`` CLI.

Exercises:
* ``--help`` exits 0 and contains the 4 shared flags + the curated
  registration flags (target_volume, mask, template, atlas, data_dir,
  --resolution, --rigid_type, --deformable_type).
* The CLI surfaces a clear ``ImportError`` mentioning ``antspy`` when the
  ``ants`` extra is not importable (lazy-import guard pattern).
"""

from __future__ import annotations

import subprocess
import sys


def test_liom_align_annotations_help_exits_0() -> None:
    """liom-align-annotations --help exits 0 with shared + curated flags."""
    result = subprocess.run(
        ["uv", "run", "liom-align-annotations", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"liom-align-annotations --help failed: rc={result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    out = result.stdout + result.stderr
    for flag in ("--log-level", "--resume", "--dask_scheduler", "--n_workers"):
        assert flag in out, f"liom-align-annotations --help missing {flag}"
    for flag in ("target_volume", "mask", "template", "atlas", "data_dir"):
        assert flag in out, f"liom-align-annotations --help missing {flag}"
    for flag in ("--resolution", "--rigid_type", "--deformable_type"):
        assert flag in out, f"liom-align-annotations --help missing {flag}"


def test_importerror_surfacing_align_annotations(monkeypatch) -> None:
    """liom-align-annotations surfaces a clear ImportError mentioning 'antspy'
    when ants is not importable (mock ants import to raise ImportError)."""
    # Make `import ants` raise ImportError by poisoning sys.modules.
    monkeypatch.setitem(sys.modules, "ants", None)
    # Provide required positional args so parse_args succeeds; the ImportError
    # fires in main()'s lazy-import guard before any image loading.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "liom-align-annotations",
            "tv.nii",
            "mask.nii",
            "tmpl.nii",
            "atlas.nii",
            "data_dir",
        ],
    )

    import pytest

    from liom_toolkit.scripts.liom_align_annotations import main

    with pytest.raises(ImportError, match="antspy"):
        main()
