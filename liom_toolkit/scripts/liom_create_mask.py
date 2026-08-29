#!/usr/bin/env python3
"""CLI: create a brain mask for an OME-Zarr volume and write it to disk."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from liom_toolkit.scripts._common import build_common_parser
from liom_toolkit.utils import create_and_write_mask


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the mask-creation CLI.

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
    p.add_argument("input_file", help="Full path to the input zarr file")
    p.add_argument(
        "--scales",
        type=float,
        nargs=3,
        default=(6.5, 6.5, 6.5),
        help="Scales (voxel size) for the Zarr dataset (default=%(default)s)",
    )
    p.add_argument(
        "--chunks",
        type=int,
        nargs=3,
        default=(128, 128, 128),
        help="Chunk size for the Zarr dataset (default=%(default)s)",
    )
    p.add_argument(
        "--fill-holes", action="store_true", help="Fill holes in the mask (default: False)"
    )
    return p


def main() -> None:
    """Create a brain mask for an OME-Zarr volume and write it to the labels group.

    Parses CLI arguments, configures logging via ``basicConfig`` on the root
    logger, optionally connects to a remote Dask scheduler when
    ``--dask_scheduler`` is given, and delegates the mask creation + write to
    :func:`liom_toolkit.utils.create_and_write_mask`.

    Note
    ----
    The ``--resume`` flag is accepted by the shared parent parser but ignored
    by this CLI: ``create_and_write_mask`` is a single-step transform with no
    checkpointable stages. The flag is honoured by the multi-stage pipeline
    CLIs that consume the same parent parser.
    """
    parser = _build_argument_parser()
    args = parser.parse_args()

    # Boundary validator (D-01): surface a nonexistent input_file at the
    # argparse boundary with an actionable exit-2 message instead of letting
    # it flow into create_and_write_mask and produce a confusing zarr
    # traceback. parser.error exits 2 — never use assert for validation (it
    # is stripped under python -O).
    if not Path(args.input_file).exists():
        parser.error(f"input file does not exist: {args.input_file}")

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.dask_scheduler:
        from liom_toolkit.utils import dask_client_manager

        # Initialize Dask client if a scheduler address is provided
        dask_client_manager.set_client(args.dask_scheduler, n_workers=args.n_workers)

    create_and_write_mask(
        zarr_file=args.input_file,
        scales=args.scales,
        chunks=args.chunks,
        fill_holes=args.fill_holes,
    )


if __name__ == "__main__":
    main()
