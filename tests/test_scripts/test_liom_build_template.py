"""Smoke + in-process ``main()`` test for the ``liom-build-template`` CLI.

Exercises:
* ``--help`` exits 0 and contains the 4 shared flags + the curated
  template-building flags (output_file, zarr_files, brain_names,
  --resolution_level, --template_resolution, --iterations, --atlas_resolution).
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
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "liom-build-template",
            out_file,
            zarr1,
            "brain1",
            "--resolution_level",
            "0",
            "--template_resolution",
            "25",
            "--atlas_resolution",
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
