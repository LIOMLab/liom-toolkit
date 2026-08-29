#!/usr/bin/env python3
"""CLI: build a groupwise registration template for a set of OME-Zarr brains."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from liom_toolkit.scripts._common import build_common_parser


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the template-building CLI.

    Returns
    -------
    argparse.ArgumentParser
        The configured argument parser (call ``parse_args()`` on it).
    """
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
        parents=[build_common_parser()],
    )
    p.add_argument("output_file", help="Path where the output template is saved (NIfTI/NRRD)")
    p.add_argument(
        "--zarr-files",
        nargs="+",
        required=True,
        help="Paths to the input OME-Zarr brain files",
    )
    p.add_argument(
        "--brain-names",
        nargs="+",
        required=True,
        help="Brain names for saving pre-registered images",
    )
    p.add_argument(
        "--resolution-level",
        type=int,
        default=3,
        help="OME-Zarr resolution level to load (default=%(default)s)",
    )
    p.add_argument(
        "--template-resolution",
        type=int,
        default=50,
        help="Template resolution in micron (default=%(default)s)",
    )
    p.add_argument(
        "--atlas-resolution",
        type=int,
        default=None,
        help="Allen atlas resolution in micron (default: template_resolution)",
    )
    p.add_argument(
        "--iterations",
        type=int,
        default=15,
        help="Number of groupwise registration iterations (default=%(default)s)",
    )
    return p


def main() -> None:
    """Build a registration template from a set of OME-Zarr brains.

    Parses CLI arguments, configures logging via ``basicConfig`` on the root
    logger, optionally connects to a remote Dask scheduler when
    ``--dask-scheduler`` is given, and delegates to
    :func:`liom_toolkit.registration.build_template_for_resolution`.

    Note
    ----
    The ``--resume`` flag is threaded through to
    ``build_template_for_resolution``: completed template-build iterations
    (whose rolling-latest template NIfTI artifact validates) are skipped and
    the build continues from the first incomplete iteration.

    Raises
    ------
    ImportError
        If ANTsPy is not installed (re-raised with an actionable message).
    """
    parser = _build_argument_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s %(name)s: %(message)s",
    )

    # File-existence check at the argparse boundary: a nonexistent input
    # path produces a clear CLI error (exit 2) instead of a raw
    # ants.image_read / ome-zarr traceback deep in the domain call. Runs
    # before the ants lazy-import guard so the error surfaces regardless of
    # whether the extra is installed.
    for zarr_path in args.zarr_files:
        if not Path(zarr_path).exists():
            parser.error(f"input file does not exist: {zarr_path}")

    # Parity check: --zarr-files and --brain-names are both repeatable
    # nargs="+" options, so argparse cannot enforce that they have the same
    # length. A mismatch would either raise a confusing IndexError deep in
    # the registration loop (more zarr files than brain names) or silently
    # drop the extra brain name (more brain names than zarr files). Surface
    # it at the CLI boundary with an actionable exit-2 message instead.
    if len(args.zarr_files) != len(args.brain_names):
        parser.error(
            f"--zarr-files and --brain-names must have the same number of values, "
            f"got {len(args.zarr_files)} zarr file(s) and "
            f"{len(args.brain_names)} brain name(s)"
        )

    try:
        import ants  # ruff: ignore[unused-import] — guard surfaces a clear ImportError before the domain call
    except ImportError as e:
        raise ImportError(
            "Please install ANTsPy (antspy extra) to use the registration CLI of the LIOM toolkit."
        ) from e

    if args.dask_scheduler:
        from liom_toolkit.utils import dask_client_manager

        dask_client_manager.set_client(args.dask_scheduler, n_workers=args.n_workers)

    from liom_toolkit.registration import build_template_for_resolution

    build_template_for_resolution(
        output_file=args.output_file,
        zarr_files=args.zarr_files,
        brain_names=args.brain_names,
        resolution_level=args.resolution_level,
        template_resolution=args.template_resolution,
        atlas_resolution=args.atlas_resolution,
        iterations=args.iterations,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
