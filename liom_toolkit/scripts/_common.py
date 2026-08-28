"""Shared parent parser for cross-cutting CLI flags.

Every LIOM Toolkit CLI consumes these via ``parents=[build_common_parser()]``
in its own ``_build_argument_parser()``. The shared flags are:

* ``--dask_scheduler`` — remote Dask scheduler address (default: local cluster)
* ``--n_workers`` — Dask local-cluster worker count (default: scheduler decides)
* ``--log-level`` — logging level for the CLI's ``basicConfig`` (default: INFO)
* ``--resume`` — resume from checkpoint if available (default: False)
"""

from __future__ import annotations

import argparse


def build_common_parser() -> argparse.ArgumentParser:
    """Build the shared parent parser for cross-cutting CLI flags.

    Use via ``parents=[build_common_parser()]`` in each CLI's
    ``_build_argument_parser()``. ``add_help=False`` is set so the child
    parser's ``--help`` wins (argparse parent-parser convention).

    Returns
    -------
    argparse.ArgumentParser
        The configured parent parser (call ``parse_args()`` on the child
        parser, not on this one directly).
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--dask_scheduler",
        type=str,
        default=None,
        help="Network address of the dask scheduler (default: local cluster)",
    )
    p.add_argument(
        "--n_workers",
        type=int,
        default=None,
        help="Number of Dask local-cluster workers (default: min(cpu_count()-1, 8))",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from checkpoint if available (default: False)",
    )
    return p
