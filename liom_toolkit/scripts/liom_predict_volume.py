#!/usr/bin/env python3
"""CLI: predict a vessel-segmentation mask for a whole OME-Zarr volume.

Thin wrapper over the nnU-Net v2 inference path: constructs an
``NnUnetV2Model`` from a trained-model directory (the output of
``nnUNetv2_train``), wraps the input store in an ``OmeZarrDataset``, and
calls ``predict_volume`` — which reads the volume, resolves the voxel
spacing, runs sliding-window inference with Gaussian overlap blending and
fold ensembling, and writes a ``(Z, Y, X)`` uint8 ``{0, 255}`` mask zarr.

Spacing precedence: an explicit ``--spacing SZ SY SX`` (microns, in z,y,x
axis order) always wins; when omitted, the spacing is read from the
input's NGFF ``coordinateTransformations`` metadata by axis name. If
neither is available the call fails rather than assuming isotropic
spacing — a silently wrong spacing mis-resamples the inference and
produces a plausible-shaped-but-wrong mask.

The ``--model-dir`` must be a TRUSTED ``nnUNetv2_train`` output directory
(containing ``dataset.json``, ``plans.json``, and
``fold_<n>/<checkpoint>``): nnU-Net loads the checkpoint upstream with
``weights_only=False``, so a model dir is code-execution-grade input —
only point this at checkpoints you trained or fully trust.

All paths are parameterized (CLI args) -- no hardcoded lab paths. All
heavy dependencies (``torch``/``nnunetv2`` via ``NnUnetV2Model``,
``OmeZarrDataset``, ``predict_volume``) are imported INSIDE ``main()``
after cheap argument validation, so this module imports cleanly with
only the core deps installed and ``--help`` is instant; the ``[ai]``
extra is required at call time.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

from liom_toolkit.scripts._common import build_common_parser

logger = logging.getLogger(__name__)


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the volume-prediction CLI.

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
    p.add_argument(
        "input",
        help="Path to the input OME-Zarr volume. A '.zip'/'.ozx' extension "
        "reads a single-file ZIP store; any other path reads a directory "
        "store.",
    )
    p.add_argument(
        "--model-dir",
        required=True,
        help="Path to the trained nnU-Net model directory (an nnUNetv2_train "
        "output containing dataset.json, plans.json, and "
        "fold_<n>/<checkpoint>). Must be a trusted artifact: the checkpoint "
        "is loaded upstream with weights_only=False.",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Path of the output mask zarr to create. Must not already "
        "exist -- the nnU-Net path refuses to overwrite (FileExistsError).",
    )
    p.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=None,
        metavar=("SZ", "SY", "SX"),
        help="Voxel spacing in microns, z y x order (e.g. '--spacing 6.5 6.5 "
        "6.5'). When omitted, spacing is read from the input's NGFF "
        "coordinateTransformations metadata; when neither is available the "
        "prediction fails rather than assuming isotropic spacing.",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device for inference (e.g. 'cuda', 'cuda:0', 'cpu'). "
        "When omitted, resolves to cuda if a GPU is available, else cpu.",
    )
    p.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Channel index to predict on when the input store is 4D "
        "(c, z, y, x) (default: %(default)s)",
    )
    p.add_argument(
        "--z-chunk-size",
        type=int,
        default=None,
        help="Optional Z-slab depth for bounded-memory inference: the volume "
        "is predicted in slabs of this many slices instead of one "
        "whole-volume call (default: whole volume)",
    )
    p.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help="Fold indices to ensemble (e.g. '--folds 0 1'). When omitted, "
        "every fold_<n> directory containing the checkpoint is used "
        "(fold_all excluded).",
    )
    p.add_argument(
        "--checkpoint-name",
        type=str,
        default="checkpoint_final.pth",
        help="Checkpoint filename inside each fold_<n> directory (default: %(default)s)",
    )
    p.add_argument(
        "--tile-step-size",
        type=float,
        default=0.5,
        help="Sliding-window tile step as a fraction of the patch size "
        "(default: %(default)s -- nnU-Net's Gaussian overlap blending "
        "assumes 0 < step <= 1)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    """Predict a vessel mask for the input OME-Zarr volume.

    Parses CLI arguments, runs all cheap validations (input exists,
    ``--model-dir`` is a directory, ``--output`` does not exist, spacing /
    channel / fold / chunk values are sane) via ``parser.error`` so each
    failure exits 2 naming the offending value, configures logging via
    ``basicConfig`` on the root logger, optionally connects to a remote
    Dask scheduler when ``--dask_scheduler`` is given, then imports the
    heavy stack function-scope and delegates to
    :func:`liom_toolkit.segmentation.vseg.prediction.predict_volume`.
    Each heavy module carries an ``ImportError`` guard naming the ``[ai]``
    extra if torch/nnunetv2 is absent.
    """
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    # All cheap validation runs BEFORE the heavy imports: a typo'd path or
    # malformed value surfaces as a clear parser.error (exit 2 naming the
    # value), never a traceback from inside zarr/torch/nnU-Net. The
    # --output existence check surfaces predict_volume's FileExistsError
    # contract early -- before any model loading.
    if not Path(args.input).exists():
        parser.error(f"input does not exist: {args.input}")
    if not Path(args.model_dir).is_dir():
        parser.error(f"--model-dir is not a directory: {args.model_dir}")
    if Path(args.output).exists():
        parser.error(
            f"--output already exists: {args.output} -- refusing to "
            "overwrite (remove it or choose a new location)"
        )
    if args.spacing is not None and any(not math.isfinite(s) or s <= 0 for s in args.spacing):
        parser.error(
            f"--spacing must be three positive finite floats (SZ SY SX), got {args.spacing}"
        )
    if args.channel < 0:
        parser.error(f"--channel must be a non-negative int, got {args.channel}")
    if args.z_chunk_size is not None and args.z_chunk_size < 1:
        parser.error(f"--z-chunk-size must be a positive int, got {args.z_chunk_size}")
    if args.folds is not None and any(f < 0 for f in args.folds):
        parser.error(f"--folds must be non-negative ints, got {args.folds}")
    if not 0 < args.tile_step_size <= 1:
        parser.error(
            f"--tile-step-size must be in the interval (0, 1], got "
            f"{args.tile_step_size} -- a step > 1 leaves un-predicted image regions"
        )

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.dask_scheduler:
        from liom_toolkit.utils import dask_client_manager

        dask_client_manager.set_client(args.dask_scheduler, n_workers=args.n_workers)

    # Heavy imports INSIDE main, after validation: keeps the module (and
    # --help) importable with only core deps. Each module carries an
    # ImportError guard naming the [ai] extra if torch/nnunetv2 is absent.
    from liom_toolkit.segmentation.vseg.dataset import OmeZarrDataset
    from liom_toolkit.segmentation.vseg.model_v2 import NnUnetV2Model
    from liom_toolkit.segmentation.vseg.prediction import predict_volume

    # device=None lets the wrapper resolve cuda-if-available-else-cpu
    # itself; a string like 'cuda:0' is converted to torch.device there.
    model = NnUnetV2Model(
        args.model_dir,
        device=args.device,
        use_folds=tuple(args.folds) if args.folds else None,
        checkpoint_name=args.checkpoint_name,
        tile_step_size=args.tile_step_size,
        allow_tqdm=True,
    )
    # patch_size must be a concrete tuple: OmeZarrDataset.__init__ indexes
    # it unconditionally to compute grid_shape (None raises TypeError).
    # (1, 1, 1) is safe because the nnU-Net path reads dataset.data and
    # dataset.zarr_path only -- it never iterates the patch index, so the
    # dataset's device is likewise inert here ('cpu' avoids implying a GPU
    # requirement when --device is omitted).
    dataset = OmeZarrDataset(
        zarr_path=args.input,
        channel=args.channel,
        patch_size=(1, 1, 1),
        rotate_patches=False,
        pre_process=False,
        normalise=False,
        device=args.device or "cpu",
    )

    spacing = tuple(args.spacing) if args.spacing is not None else None
    predict_volume(
        model,
        dataset,
        args.output,
        spacing=spacing,
        z_chunk_size=args.z_chunk_size,
    )
    logger.info("Wrote predicted mask volume to %s", args.output)


if __name__ == "__main__":
    main()
