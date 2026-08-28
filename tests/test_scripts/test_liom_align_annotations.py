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
from unittest.mock import patch


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


def test_liom_align_annotations_main_smoke(tmp_path, fake_ants, monkeypatch) -> None:
    """``main()`` reaches the real ``align_annotations_to_volume`` with the expected kwargs.

    D-01 expansion slice for an ants-gated CLI. The ``ants`` lazy-import guard
    is satisfied by the ``fake_ants`` fixture (leaf-only ``sys.modules``
    injection); ``ants.image_read`` returns a MagicMock image (configured in
    ``fake_ants``) so the four image loads in ``main()`` do not crash. The
    domain callee is spied via ``patch`` on the imported name so the test does
    not need real NIfTI inputs. The spy's ``call_args`` kwargs are asserted
    against the verified kwarg map. A kwarg-name typo in ``main()``'s call to
    ``align_annotations_to_volume`` raises ``TypeError`` at the ``main()``
    call site before the spy is invoked.
    """
    data_dir = str(tmp_path / "data")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "liom-align-annotations",
            "tv.nii",
            "mask.nii",
            "tmpl.nii",
            "atlas.nii",
            data_dir,
            "--resolution",
            "25",
            "--rigid_type",
            "Similarity",
            "--deformable_type",
            "SyN",
        ],
    )

    from liom_toolkit.scripts.liom_align_annotations import main

    with patch("liom_toolkit.registration.align_annotations_to_volume") as spy:
        main()

    assert spy.called, (
        "main() did not call align_annotations_to_volume -- the domain callee was not reached"
    )
    kwargs = spy.call_args.kwargs
    assert kwargs["data_dir"] == data_dir
    assert kwargs["resolution"] == 25
    assert kwargs["rigid_type"] == "Similarity"
    assert kwargs["deformable_type"] == "SyN"
    # The four image args are the MagicMock image returned by fake_ants.image_read
    # (fake_ants returns the same mock for every image_read call).
    assert kwargs["target_volume"] is not None
    assert kwargs["mask"] is not None
    assert kwargs["template"] is not None
    assert kwargs["atlas"] is not None
