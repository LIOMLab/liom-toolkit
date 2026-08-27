#!/usr/bin/env python3
"""CLI: align annotations to a target volume."""

from __future__ import annotations

import argparse
import logging

from liom_toolkit.scripts._common import build_common_parser


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the annotation-alignment CLI.

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
    p.add_argument("target_volume", help="Path to the target volume image (NIfTI/NRRD)")
    p.add_argument("mask", help="Path to the mask image (NIfTI/NRRD)")
    p.add_argument("template", help="Path to the template image (NIfTI/NRRD)")
    p.add_argument("atlas", help="Path to the atlas/annotation image (NIfTI/NRRD)")
    p.add_argument("data_dir", help="Directory for intermediary files")
    p.add_argument(
        "--resolution",
        type=int,
        default=25,
        help="Atlas resolution in micron, must be 10/25/50/100 (default=%(default)s)",
    )
    p.add_argument(
        "--rigid_type",
        type=str,
        default="Similarity",
        help="Rigid registration type (default=%(default)s)",
    )
    p.add_argument(
        "--deformable_type",
        type=str,
        default="SyN",
        help="Deformable registration type (default=%(default)s)",
    )
    return p


def main() -> None:
    """Align an annotation to a target volume.

    Parses CLI arguments, configures logging via ``basicConfig`` on the root
    logger, optionally connects to a remote Dask scheduler when
    ``--dask_scheduler`` is given, loads the four input images as ANTsImages,
    and delegates to
    :func:`liom_toolkit.registration.align_annotations_to_volume`.

    Note
    ----
    The ``--resume`` flag is accepted by the shared parent parser but ignored
    by this CLI: ``align_annotations_to_volume`` is a single registration step
    with no checkpointable stages. The flag is honoured by the multi-stage
    pipeline CLIs that consume the same parent parser.

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

    try:
        import ants
    except ImportError as e:
        raise ImportError(
            "Please install ANTsPy (antspy extra) to use the registration CLI "
            "of the LIOM toolkit."
        ) from e

    if args.dask_scheduler:
        from liom_toolkit.utils import dask_client_manager

        dask_client_manager.set_client(args.dask_scheduler, n_workers=args.n_workers)

    from liom_toolkit.registration import align_annotations_to_volume

    target_volume = ants.image_read(args.target_volume)
    mask = ants.image_read(args.mask)
    template = ants.image_read(args.template)
    atlas = ants.image_read(args.atlas)

    align_annotations_to_volume(
        target_volume=target_volume,
        mask=mask,
        template=template,
        atlas=atlas,
        data_dir=args.data_dir,
        resolution=args.resolution,
        rigid_type=args.rigid_type,
        deformable_type=args.deformable_type,
    )


if __name__ == "__main__":
    main()
