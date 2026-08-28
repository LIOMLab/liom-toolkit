#!/usr/bin/env python3
"""CLI: 2D Frangi + threshold vessel segmentation of a single image."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from liom_toolkit.scripts._common import build_common_parser
from liom_toolkit.segmentation import segment_2d_image


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the 2D segmentation CLI.

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
    p.add_argument("input_file", help="Full path to the input 2D image (TIFF/PNG/etc.)")
    p.add_argument(
        "output_file",
        help="Full path to the output directory (segmentation masks are written here)",
    )
    p.add_argument(
        "--name",
        type=str,
        default=None,
        help="Base name for the output files (default: input file stem)",
    )
    p.add_argument(
        "--frangi-sigma-range",
        type=int,
        nargs=3,
        default=(2, 16, 2),
        help="Frangi filter sigma range (start, stop, step) (default=%(default)s)",
    )
    p.add_argument(
        "--frangi-black-ridges",
        action="store_true",
        help="Detect black ridges instead of bright ridges (default: False)",
    )
    p.add_argument(
        "--local-threshold",
        action="store_true",
        help="Use Sauvola local thresholding instead of Li global thresholding (default: False)",
    )
    p.add_argument(
        "--local-threshold-size",
        type=int,
        default=15,
        help="Window size for Sauvola local thresholding, must be odd (default=%(default)s)",
    )
    return p


def main() -> None:
    """Segment a 2D image and write the vessel + tissue masks to disk.

    Parses CLI arguments, configures logging via ``basicConfig`` on the root
    logger, and delegates to
    :func:`liom_toolkit.segmentation.segment_2d_image`.

    Note
    ----
    The ``--resume`` flag is accepted by the shared parent parser but ignored
    by this CLI: ``segment_2d_image`` is a single-step transform with no
    checkpointable stages. The flag is honoured by the multi-stage pipeline
    CLIs that consume the same parent parser.
    """
    parser = _build_argument_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s %(name)s: %(message)s",
    )

    input_path = Path(args.input_file)
    name = args.name if args.name is not None else input_path.stem
    image = np.asarray(iio.imread(str(input_path)), dtype=np.float64)

    segment_2d_image(
        output_dir=args.output_file,
        image=image,
        name=name,
        frangi_sigma_range=tuple(args.frangi_sigma_range),
        frangi_black_ridges=args.frangi_black_ridges,
        local_threshold=args.local_threshold,
        local_threshold_size=args.local_threshold_size,
    )


if __name__ == "__main__":
    main()
