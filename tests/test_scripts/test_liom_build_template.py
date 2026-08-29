"""Smoke + in-process ``main()`` test for the ``liom-build-template`` CLI.

Exercises:
* ``--help`` exits 0 and contains the 4 shared flags + the curated
  template-building flags (output_file, --zarr-files, --brain-names,
  --resolution-level, --template-resolution, --iterations, --atlas-resolution).
* ``main()`` reaches the real ``build_template_for_resolution`` domain callee
  (D-01 expansion slice for an ants-gated CLI). The ``ants`` lazy-import guard
  is satisfied via the ``fake_ants`` fixture (leaf-only ``sys.modules``
  injection); the domain callee is spied via ``unittest.mock.patch`` on the
  imported name so the test does not have to build real OME-Zarr brains for a
  full groupwise registration. The spy still catches a ``main()`` kwarg-name
  typo: the typo raises ``TypeError`` at the ``main()`` call site BEFORE the
  spy is invoked (per RESEARCH "Critical D-01 nuance"). The spy's
  ``call_args`` kwargs are asserted against the verified kwarg map.

Per D-01: no ``pytest.importorskip``; the ``ants`` leaf is mocked via
``fake_ants`` (``sys.modules`` injection with mandatory teardown), NOT the
liom domain callee in a way that absorbs arbitrary kwargs.

The ``--help`` smoke check invokes the script's ``_build_argument_parser()``
in-process and formats its help text, rather than spawning a ``uv run``
subprocess (avoids the per-invocation venv-reconcile + interpreter-startup
cost).
"""

from __future__ import annotations

import sys
from unittest.mock import patch


def test_liom_build_template_help_exits_0() -> None:
    """liom-build-template --help contains shared + curated flags."""
    from liom_toolkit.scripts.liom_build_template import _build_argument_parser

    out = _build_argument_parser().format_help()
    for flag in ("--log-level", "--resume", "--dask-scheduler", "--n-workers"):
        assert flag in out, f"liom-build-template --help missing {flag}"
    for flag in ("output_file", "--zarr-files", "--brain-names"):
        assert flag in out, f"liom-build-template --help missing {flag}"
    for flag in (
        "--resolution-level",
        "--template-resolution",
        "--iterations",
        "--atlas-resolution",
    ):
        assert flag in out, f"liom-build-template --help missing {flag}"


def test_liom_build_template_main_smoke(tmp_path, fake_ants, monkeypatch) -> None:
    """``main()`` reaches the real ``build_template_for_resolution`` with the expected kwargs.

    The ``ants`` lazy-import guard is satisfied by the ``fake_ants`` fixture
    (leaf-only ``sys.modules`` injection). The domain callee is spied via
    ``patch`` on the imported name so the test does not need to build real
    OME-Zarr brains for a full groupwise registration. The spy's ``call_args``
    kwargs are asserted against the verified kwarg map. A kwarg-name typo in
    ``main()``'s call to ``build_template_for_resolution`` raises
    ``TypeError`` at the ``main()`` call site before the spy is invoked.
    """
    out_file = str(tmp_path / "template.nii")
    zarr1 = str(tmp_path / "brain1.zarr")
    # Touch the zarr path so the file-existence check in main() passes.
    with open(zarr1, "w"):
        pass
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "liom-build-template",
            out_file,
            "--zarr-files",
            zarr1,
            "--brain-names",
            "brain1",
            "--resolution-level",
            "0",
            "--template-resolution",
            "25",
            "--atlas-resolution",
            "25",
            "--iterations",
            "1",
        ],
    )

    from liom_toolkit.scripts.liom_build_template import main

    with patch("liom_toolkit.registration.build_template_for_resolution") as spy:
        main()

    assert spy.called, (
        "main() did not call build_template_for_resolution -- the domain callee was not reached"
    )
    kwargs = spy.call_args.kwargs
    assert kwargs["output_file"] == out_file
    assert kwargs["zarr_files"] == [zarr1]
    assert kwargs["brain_names"] == ["brain1"]
    assert kwargs["resolution_level"] == 0
    assert kwargs["template_resolution"] == 25
    assert kwargs["atlas_resolution"] == 25
    assert kwargs["iterations"] == 1
    assert kwargs["resume"] is False


def test_liom_build_template_two_zarr_two_brains_no_misassignment(
    tmp_path, fake_ants, monkeypatch
) -> None:
    """``liom-build-template`` assigns 2 zarr files + 2 brain names correctly.

    Regression test for the silent mis-assignment footgun: with the old
    consecutive ``nargs="+"`` positionals, ``out.nii b1.zarr b2.zarr brain1
    brain2`` resolved to ``zarr_files=['b1.zarr','b2.zarr','brain1']`` and
    ``brain_names=['brain2']`` — a plausible-shaped-but-wrong result that
    silently propagated into the groupwise registration. The repeatable
    ``--zarr-files`` / ``--brain-names`` options make the assignment
    unambiguous. The domain callee is spied so the test does not need real
    OME-Zarr brains; the spy's ``call_args`` kwargs are asserted against the
    exact lists passed on the command line.
    """
    out_file = str(tmp_path / "template.nii")
    zarr1 = str(tmp_path / "brain1.zarr")
    zarr2 = str(tmp_path / "brain2.zarr")
    # Touch the zarr paths so the file-existence check in main() passes.
    for p in (zarr1, zarr2):
        with open(p, "w"):
            pass
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "liom-build-template",
            out_file,
            "--zarr-files",
            zarr1,
            zarr2,
            "--brain-names",
            "brain1",
            "brain2",
            "--iterations",
            "1",
        ],
    )

    from liom_toolkit.scripts.liom_build_template import main

    with patch("liom_toolkit.registration.build_template_for_resolution") as spy:
        main()

    assert spy.called, (
        "main() did not call build_template_for_resolution -- the domain callee was not reached"
    )
    kwargs = spy.call_args.kwargs
    assert kwargs["zarr_files"] == [zarr1, zarr2]
    assert kwargs["brain_names"] == ["brain1", "brain2"]


def test_liom_build_template_nonexistent_zarr_exits_2(tmp_path, monkeypatch, capsys) -> None:
    """A nonexistent ``--zarr-files`` path exits 2 with an actionable message.

    The file-existence check runs at the argparse boundary (before the ants
    lazy-import guard) so a bad input path produces a clear CLI error
    ``input file does not exist: <path>`` (argparse exit 2) instead of a raw
    ``ants.image_read`` / ome-zarr traceback deep in the domain call. Uses
    ``parser.error`` (not ``assert``) per the project's validation rule.
    """
    out_file = str(tmp_path / "template.nii")
    missing_zarr = str(tmp_path / "this_zarr_path_does_not_exist.zarr")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "liom-build-template",
            out_file,
            "--zarr-files",
            missing_zarr,
            "--brain-names",
            "brain1",
            "--iterations",
            "1",
        ],
    )

    import pytest

    from liom_toolkit.scripts.liom_build_template import main

    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "input file does not exist" in captured.err
    assert missing_zarr in captured.err


def test_liom_build_template_mismatched_lengths_exits_2(tmp_path, monkeypatch, capsys) -> None:
    """Mismatched --zarr-files / --brain-names lengths exit 2 at the CLI boundary.

    Both options are ``nargs="+"`` so argparse cannot enforce equal length.
    Without the parity check, 2 zarr files + 3 brain names silently drops the
    extra name, and 3 zarr files + 2 brain names raises a confusing
    ``IndexError`` deep in the registration loop. The check fires after the
    file-existence loop, so the zarr paths must exist on disk first.
    """
    out_file = str(tmp_path / "template.nii")
    zarr1 = str(tmp_path / "brain1.zarr")
    zarr2 = str(tmp_path / "brain2.zarr")
    # Touch the zarr paths so the file-existence check passes first; the
    # parity check then fires on the 2-vs-3 length mismatch.
    for p in (zarr1, zarr2):
        with open(p, "w"):
            pass
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "liom-build-template",
            out_file,
            "--zarr-files",
            zarr1,
            zarr2,
            "--brain-names",
            "brain1",
            "brain2",
            "brain3",
            "--iterations",
            "1",
        ],
    )

    import pytest

    from liom_toolkit.scripts.liom_build_template import main

    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "same number of values" in captured.err
