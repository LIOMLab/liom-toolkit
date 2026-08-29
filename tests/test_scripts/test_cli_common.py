"""Tests for the shared CLI parent parser (``liom_toolkit/scripts/_common.py``).

Covers the cross-cutting flag contract consumed by every CLI via
``parents=[build_common_parser()]``: ``--dask-scheduler``, ``--n-workers``,
``--log-level``, ``--resume``. Also smoke-tests the existing retrofitted CLIs'
``--help`` to confirm the shared flags are present.

The ``--help`` smoke checks invoke each script's ``_build_argument_parser()``
in-process and format its help text, rather than spawning a ``uv run``
subprocess. This avoids the per-invocation venv-reconcile + interpreter-startup
cost (~2s each) while exercising the exact same parser construction the CLI
uses at runtime.
"""

from __future__ import annotations

from liom_toolkit.scripts._common import build_common_parser


def test_build_common_parser_has_flags() -> None:
    """build_common_parser() returns a parser with the 4 cross-cutting flags."""
    parser = build_common_parser()
    args = parser.parse_args(
        [
            "--dask-scheduler",
            "127.0.0.1:8786",
            "--n-workers",
            "4",
            "--log-level",
            "DEBUG",
            "--resume",
        ]
    )
    assert args.dask_scheduler == "127.0.0.1:8786"
    assert args.n_workers == 4
    assert args.log_level == "DEBUG"
    assert args.resume is True


def test_common_parser_add_help_false() -> None:
    """build_common_parser().add_help is False (required for parent parsers)."""
    parser = build_common_parser()
    assert parser.add_help is False


def test_log_level_choices() -> None:
    """--log-level accepts DEBUG/INFO/WARNING/ERROR and rejects other values."""
    parser = build_common_parser()
    for level in ("DEBUG", "INFO", "WARNING", "ERROR"):
        args = parser.parse_args(["--log-level", level])
        assert args.log_level == level
    # argparse rejects invalid choices by raising SystemExit (error exit).
    import pytest

    with pytest.raises(SystemExit):
        parser.parse_args(["--log-level", "TRACE"])


def test_resume_default_false() -> None:
    """--resume default is False; store_true sets it True."""
    parser = build_common_parser()
    args_default = parser.parse_args([])
    assert args_default.resume is False
    args_set = parser.parse_args(["--resume"])
    assert args_set.resume is True


def test_n_workers_default_none() -> None:
    """--n-workers defaults to None (local cluster decides)."""
    parser = build_common_parser()
    args = parser.parse_args([])
    assert args.n_workers is None


def test_dask_scheduler_default_none() -> None:
    """--dask-scheduler defaults to None (local cluster)."""
    parser = build_common_parser()
    args = parser.parse_args([])
    assert args.dask_scheduler is None


def _help_text(module_name: str) -> str:
    """Build the script's argparse parser in-process and return its --help text.

    Imports ``liom_toolkit.scripts.<module_name>``, calls its
    ``_build_argument_parser()``, and returns ``parser.format_help()`` — the
    exact text ``--help`` would print without the subprocess overhead.
    """
    import importlib

    mod = importlib.import_module(f"liom_toolkit.scripts.{module_name}")
    parser = mod._build_argument_parser()
    return parser.format_help()


def test_liom_segment_2d_help_exits_0() -> None:
    """liom-segment-2d --help contains the 4 shared flags."""
    out = _help_text("liom_segment_2d")
    for flag in ("--log-level", "--resume", "--dask-scheduler", "--n-workers"):
        assert flag in out, f"liom-segment-2d --help missing {flag}"


def test_existing_clis_have_common_flags() -> None:
    """liom-convert-hdf5-to-zarr and liom-create-mask --help contain the shared flags."""
    for module_name in ("liom_convert_hdf5_to_zarr", "liom_create_mask"):
        out = _help_text(module_name)
        assert "--log-level" in out, f"{module_name} --help missing --log-level"
        assert "--resume" in out, f"{module_name} --help missing --resume"
