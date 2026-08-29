"""Smoke + ImportError-surfacing tests for the ``liom-align-annotations`` CLI.

Exercises:
* ``--help`` exits 0 and contains the 4 shared flags + the curated
  registration flags (target_volume, mask, template, atlas, data_dir,
  --resolution, --rigid-type, --deformable-type).
* The CLI surfaces a clear ``ImportError`` mentioning ``antspy`` when the
  ``ants`` extra is not importable (lazy-import guard pattern).
* ``--resolution`` rejects values outside [10, 25, 50, 100] with exit 2
  (argparse ``choices=`` constraint at the boundary).
* A nonexistent input path exits 2 with an actionable message
  (file-existence check before the ants lazy-import guard).

The ``--help`` smoke check invokes the script's ``_build_argument_parser()``
in-process and formats its help text, rather than spawning a ``uv run``
subprocess (avoids the per-invocation venv-reconcile + interpreter-startup
cost).
"""

from __future__ import annotations

import sys
from unittest.mock import patch


def test_liom_align_annotations_help_exits_0() -> None:
    """liom-align-annotations --help contains shared + curated flags."""
    from liom_toolkit.scripts.liom_align_annotations import _build_argument_parser

    out = _build_argument_parser().format_help()
    for flag in ("--log-level", "--resume", "--dask-scheduler", "--n-workers"):
        assert flag in out, f"liom-align-annotations --help missing {flag}"
    for flag in ("target_volume", "mask", "template", "atlas", "data_dir"):
        assert flag in out, f"liom-align-annotations --help missing {flag}"
    for flag in ("--resolution", "--rigid-type", "--deformable-type"):
        assert flag in out, f"liom-align-annotations --help missing {flag}"


def test_importerror_surfacing_align_annotations(tmp_path, monkeypatch) -> None:
    """liom-align-annotations surfaces a clear ImportError mentioning 'antspy'
    when ants is not importable (mock ants import to raise ImportError).

    The four image positionals + data_dir are materialized in ``tmp_path`` so
    the argparse-boundary file-existence check passes before the ants
    lazy-import guard fires -- the ImportError is the behaviour under test
    here, not the path check (covered by
    ``test_liom_align_annotations_missing_input_exits_2``).
    """
    # Make `import ants` raise ImportError by poisoning sys.modules.
    monkeypatch.setitem(sys.modules, "ants", None)
    # Materialize the input paths so the file-existence check passes; the
    # ImportError fires in main()'s lazy-import guard after the path check
    # and before any image loading.
    tv = tmp_path / "tv.nii"
    mask = tmp_path / "mask.nii"
    tmpl = tmp_path / "tmpl.nii"
    atlas = tmp_path / "atlas.nii"
    data_dir = tmp_path / "data"
    for p in (tv, mask, tmpl, atlas):
        p.touch()
    data_dir.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "liom-align-annotations",
            str(tv),
            str(mask),
            str(tmpl),
            str(atlas),
            str(data_dir),
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

    The four image positionals + data_dir are materialized in ``tmp_path`` so
    the argparse-boundary file-existence check passes before the spy fires
    (the critical D-01/D-04 coupling). The argv uses the hyphenated
    ``--rigid-type`` / ``--deformable-type`` flag strings; argparse
    auto-derives ``dest`` (hyphen to underscore), so ``main()`` reads
    ``args.rigid_type`` / ``args.deformable_type`` unchanged and the spy
    kwargs assertions stay identical.
    """
    tv = tmp_path / "tv.nii"
    mask = tmp_path / "mask.nii"
    tmpl = tmp_path / "tmpl.nii"
    atlas = tmp_path / "atlas.nii"
    data_dir = tmp_path / "data"
    # Touch the image paths + mkdir data_dir so the file-existence check in
    # main() passes before the spy fires.
    for p in (tv, mask, tmpl, atlas):
        p.touch()
    data_dir.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "liom-align-annotations",
            str(tv),
            str(mask),
            str(tmpl),
            str(atlas),
            str(data_dir),
            "--resolution",
            "25",
            "--rigid-type",
            "Similarity",
            "--deformable-type",
            "SyN",
        ],
    )

    from liom_toolkit.scripts.liom_align_annotations import main

    # The fake_ants fixture returns the same MagicMock image for every
    # image_read call. Capture it so the assertions below verify IDENTITY
    # (the exact mock passed by main()), not just non-None -- a bare
    # `is not None` check would pass even if main() passed the wrong
    # variable (e.g. target_volume=mask).
    fake_image = fake_ants.image_read.return_value

    with patch("liom_toolkit.registration.align_annotations_to_volume") as spy:
        main()

    assert spy.called, (
        "main() did not call align_annotations_to_volume -- the domain callee was not reached"
    )
    kwargs = spy.call_args.kwargs
    assert kwargs["data_dir"] == str(data_dir)
    assert kwargs["resolution"] == 25
    assert kwargs["rigid_type"] == "Similarity"
    assert kwargs["deformable_type"] == "SyN"
    # The four image args are the MagicMock image returned by fake_ants.image_read
    # (fake_ants returns the same mock for every image_read call). Assert
    # identity so a regression where main() passes the wrong variable
    # (e.g. target_volume=mask) is caught.
    assert kwargs["target_volume"] is fake_image
    assert kwargs["mask"] is fake_image
    assert kwargs["template"] is fake_image
    assert kwargs["atlas"] is fake_image


def test_liom_align_annotations_resolution_choices_reject_invalid() -> None:
    """``--resolution`` rejects values outside [10, 25, 50, 100] with exit 2.

    Regression test for the silent wrong-atlas-alignment footgun: an invalid
    ``--resolution`` value (e.g. 30) previously flowed into
    ``align_annotations_to_volume`` and silently produced a wrong atlas
    alignment (the help text said "must be 10/25/50/100" but no constraint
    was enforced). The ``choices=[10, 25, 50, 100]`` constraint at the
    argparse boundary exits 2 before the domain call. Drives the parser
    in-process (no subprocess); argparse raises ``SystemExit(2)`` on a
    bad choice.
    """
    import pytest

    from liom_toolkit.scripts.liom_align_annotations import _build_argument_parser

    with pytest.raises(SystemExit) as excinfo:
        _build_argument_parser().parse_args(
            ["tv.nii", "mask.nii", "tmpl.nii", "atlas.nii", "data_dir", "--resolution", "30"]
        )
    assert excinfo.value.code == 2


def test_liom_align_annotations_missing_input_exits_2(
    tmp_path, fake_ants, monkeypatch, capsys
) -> None:
    """A nonexistent ``target_volume`` path exits 2 with an actionable message.

    The file-existence check runs at the argparse boundary (before the ants
    lazy-import guard) so a bad input path produces a clear CLI error
    (``input file does not exist: <path>``, argparse exit 2) instead of a
    raw ``ants.image_read`` traceback deep in the domain call. Uses
    ``parser.error`` (not ``assert``) per the project's validation rule
    (AGENTS section 2). The other positionals are materialized in
    ``tmp_path`` so only ``target_volume`` is missing -- the check loops in
    positional order and surfaces the first missing path.
    """
    # Materialize the other positionals so only target_volume is missing.
    mask = tmp_path / "mask.nii"
    tmpl = tmp_path / "tmpl.nii"
    atlas = tmp_path / "atlas.nii"
    data_dir = tmp_path / "data"
    for p in (mask, tmpl, atlas):
        p.touch()
    data_dir.mkdir()
    missing_target = str(tmp_path / "nope_target.nii")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "liom-align-annotations",
            missing_target,
            str(mask),
            str(tmpl),
            str(atlas),
            str(data_dir),
            "--resolution",
            "25",
        ],
    )

    import pytest

    from liom_toolkit.scripts.liom_align_annotations import main

    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "does not exist" in captured.err
    assert missing_target in captured.err
