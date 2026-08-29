#!/usr/bin/env python3
"""CLI: compute per-region vessel morphometric statistics for a brain slice."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from liom_toolkit.scripts._common import build_common_parser


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the slice-metrics CLI.

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
    p.add_argument("output_dir", help="Directory to save the output metrics to")
    p.add_argument(
        "image",
        help="Label (filename/identifier) of the brain slice, used in the output DataFrame",
    )
    p.add_argument("mask_file", help="Path to the tissue mask image (TIFF/PNG)")
    p.add_argument("vessel_mask_file", help="Path to the vessel mask image (TIFF/PNG)")
    p.add_argument("region_map_file", help="Path to the region map image (TIFF/PNG)")
    p.add_argument("vessel_exclude_file", help="Path to the vessel-exclude mask image (TIFF/PNG)")
    p.add_argument(
        "--voxel-size",
        type=float,
        default=0.65,
        help="Voxel size in micron (default=%(default)s)",
    )
    return p


def main() -> None:
    """Compute per-region vessel metrics for a brain slice.

    Parses CLI arguments, configures logging via ``basicConfig`` on the root
    logger, loads the four mask/region arrays from their file paths, and
    delegates to
    :func:`liom_toolkit.segmentation.stats.compute_slice_metrics`.

    Note
    ----
    The ``--resume`` flag is accepted by the shared parent parser but ignored
    by this CLI: ``compute_slice_metrics`` is a single-step computation with
    no checkpointable stages.
    """
    parser = _build_argument_parser()
    args = parser.parse_args()

    # Validate the 4 input-file positionals at the argparse boundary so a
    # nonexistent path exits 2 with an actionable message naming the offending
    # file, instead of leaking a raw imageio traceback. output_dir is created
    # by the domain callee and image is a label string, so neither is checked
    # here. parser.error exits 2 — never use assert for validation (it is
    # stripped under python -O).
    for input_attr in (
        "mask_file",
        "vessel_mask_file",
        "region_map_file",
        "vessel_exclude_file",
    ):
        input_path = Path(getattr(args, input_attr))
        if not input_path.exists():
            parser.error(f"input file does not exist: {input_path}")

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.dask_scheduler:
        from liom_toolkit.utils import dask_client_manager

        dask_client_manager.set_client(args.dask_scheduler, n_workers=args.n_workers)

    import imageio.v3 as iio
    import numpy as np

    from liom_toolkit.segmentation.stats import compute_slice_metrics

    mask = np.asarray(iio.imread(args.mask_file))
    vessel_mask = np.asarray(iio.imread(args.vessel_mask_file))
    region_map = np.asarray(iio.imread(args.region_map_file))
    vessel_exclude = np.asarray(iio.imread(args.vessel_exclude_file))

    compute_slice_metrics(
        output_dir=args.output_dir,
        image=args.image,
        mask=mask,
        vessel_mask=vessel_mask,
        region_map=region_map,
        vessel_exclude=vessel_exclude,
        voxel_size=args.voxel_size,
    )


if __name__ == "__main__":
    main()
