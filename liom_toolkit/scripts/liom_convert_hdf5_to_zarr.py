#!/usr/bin/env python3
"""CLI: convert an HDF5 volume to an OME-Zarr store."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from liom_toolkit.conversion import convert_hdf5_to_zarr
from liom_toolkit.scripts._common import build_common_parser


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the HDF5-to-Zarr conversion CLI.

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
    p.add_argument("input_file", help="Full path to the input HDF5 file")
    p.add_argument("output_file", help="Full path to the output Zarr file")
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
    return p


def main() -> None:
    """Convert an HDF5 volume to OME-Zarr.

    Parses CLI arguments, configures logging via ``basicConfig`` on the root
    logger, optionally connects to a remote Dask scheduler when
    ``--dask_scheduler`` is given, and delegates the conversion to
    :func:`liom_toolkit.conversion.convert_hdf5_to_zarr`.

    Note
    ----
    The ``--resume`` flag is accepted by the shared parent parser but ignored
    by this CLI: ``convert_hdf5_to_zarr`` is a single-step conversion with no
    checkpointable stages. The flag is honoured by the multi-stage pipeline
    CLIs that consume the same parent parser.
    """
    parser = _build_argument_parser()
    args = parser.parse_args()

    # Validate the input_file positional at the argparse boundary so a
    # nonexistent path exits 2 with an actionable message naming the offending
    # file, instead of leaking a raw h5py traceback. parser.error exits 2 —
    # never use assert for validation (it is stripped under python -O).
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

    # Convert the HDF5 file to Zarr format
    convert_hdf5_to_zarr(
        hdf5_file=args.input_file,
        zarr_file=args.output_file,
        scales=args.scales,
        chunks=args.chunks,
    )


if __name__ == "__main__":
    main()
