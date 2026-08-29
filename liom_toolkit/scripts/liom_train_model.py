#!/usr/bin/env python3
"""CLI: train the vessel segmentation U-Net on an OME-Zarr dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from liom_toolkit.scripts._common import build_common_parser


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the model-training CLI.

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
    p.add_argument("dataset_file", help="Path to the OME-Zarr dataset file")
    p.add_argument("node_name", help="Name of the node in the zarr file")
    p.add_argument(
        "--output-train",
        type=str,
        default="training",
        help="Output directory for training artifacts (default=%(default)s)",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=62,
        help="Number of training epochs (default=%(default)s)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=35,
        help="Training batch size (default=%(default)s)",
    )
    p.add_argument(
        "--learning-rate",
        type=float,
        default=0.003673,
        help="Optimizer learning rate (default=%(default)s)",
    )
    p.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="wandb entity/team (default: wandb user default)",
    )
    p.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="wandb project name (default: wandb default project)",
    )
    p.add_argument(
        "--pretrained-artifact",
        type=str,
        default=None,
        help="wandb artifact path for pretrained weights (default: train from scratch)",
    )
    p.add_argument(
        "--wandb-mode",
        type=str,
        default="offline",
        choices=["online", "offline", "disabled"],
        help="wandb mode: online/offline/disabled (default=%(default)s)",
    )
    p.add_argument(
        "--ddp",
        action="store_true",
        help="Enable DistributedDataParallel training (launch via torchrun; "
        "requires RANK/WORLD_SIZE/LOCAL_RANK env vars)",
    )
    p.add_argument(
        "--amp",
        action="store_true",
        help="Enable Automatic Mixed Precision (torch.amp.GradScaler; no-op on CPU/gloo)",
    )
    return p


def main() -> None:
    """Train the vessel segmentation U-Net.

    Parses CLI arguments, configures logging via ``basicConfig`` on the root
    logger, optionally connects to a remote Dask scheduler when
    ``--dask_scheduler`` is given, and delegates to
    :func:`liom_toolkit.segmentation.vseg.training.train_model`.

    Note
    ----
    The ``--resume`` flag is threaded through to ``train_model``: the
    manifest's ``last_completed_epoch`` is read and training continues from
    ``last_completed_epoch + 1`` using the existing per-epoch
    ``checkpoint.{epoch}.pth`` weights artifact.

    Raises
    ------
    ImportError
        If PyTorch or wandb is not installed (re-raised with an actionable message).
    """
    parser = _build_argument_parser()
    args = parser.parse_args()

    # Validate the dataset_file positional at the argparse boundary so a
    # nonexistent path exits 2 with a clear message instead of a raw
    # zarr/torch traceback from inside train_model. Runs BEFORE the heavy-dep
    # lazy-import guard so a bad path surfaces regardless of whether the ai
    # extra is installed. node_name is a string label, not a path — not checked.
    if not Path(args.dataset_file).exists():
        parser.error(f"input file does not exist: {args.dataset_file}")

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        import torch  # ruff: ignore[unused-import] — guard surfaces a clear ImportError before train_model
    except ImportError as e:
        raise ImportError(
            "Please install PyTorch (ai extra) to use the training CLI of the LIOM toolkit."
        ) from e
    try:
        import wandb  # ruff: ignore[unused-import] — guard surfaces a clear ImportError before train_model
    except ImportError as e:
        raise ImportError(
            "Please install wandb (ai extra) to use the training CLI of the LIOM toolkit."
        ) from e

    if args.dask_scheduler:
        from liom_toolkit.utils import dask_client_manager

        dask_client_manager.set_client(args.dask_scheduler, n_workers=args.n_workers)

    from liom_toolkit.segmentation.vseg.training import train_model

    train_model(
        dataset_file=args.dataset_file,
        node_name=args.node_name,
        output_train=args.output_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        pretrained_artifact=args.pretrained_artifact,
        wandb_mode=args.wandb_mode,
        resume=args.resume,
        ddp=args.ddp,
        use_amp=args.amp,
    )


if __name__ == "__main__":
    main()
