#!/usr/bin/env python3
"""CLI: run masked-inpainting self-supervised pretraining on unlabeled LSFM volumes.

Builds the nnU-Net 2D ResEnc network (via ``build_pretrain_network`` so the
saved checkpoint's state_dict keys match what the warm-start
``-pretrained_weights`` flag expects), constructs the SSL slice corpus from
the supplied OME-Zarr volumes (multi-plane, z-scored per-channel), applies
the vessel-aware block mask (Frangi-biased hole placement), and runs the
masked-inpainting reconstruction loop. The checkpoint is saved as
``{'network_weights': network.state_dict()}`` at ``--pretrained-output`` --
the format the warm-start loader consumes.

All paths are parameterized (CLI args) -- no hardcoded lab paths (AGENTS
section 1). The SSL pipeline callables (``SSLCorpus``,
``vessel_aware_block_mask``, ``masked_inpainting_pretrain``,
``build_pretrain_network``) are imported INSIDE ``main()`` so this module
imports cleanly with only the core deps installed; the ``[ai,benchmark]``
extra is required at call time (torch + MONAI + nnunetv2).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from liom_toolkit.scripts._common import build_common_parser

logger = logging.getLogger(__name__)


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the SSL pretraining CLI.

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
        "--volume-paths",
        nargs="+",
        required=True,
        help="Paths to the unlabeled OME-Zarr stores to pretrain on (required -- "
        "no lab-specific default; all paths are parameters)",
    )
    p.add_argument(
        "--plans",
        required=True,
        help="Path to the nnU-Net plans JSON (the architecture block the network is "
        "built from -- the checkpoint keys match the warm-start loader)",
    )
    p.add_argument(
        "--dataset-json",
        required=True,
        help="Path to the nnU-Net dataset.json (channel_names + numClasses for the "
        "network's input/output channel counts)",
    )
    p.add_argument(
        "--pretrained-output",
        required=True,
        help="Path to write the pretrained checkpoint "
        "({'network_weights': state_dict} -- the warm-start format)",
    )
    p.add_argument(
        "--plane-mix",
        type=float,
        nargs=3,
        default=(0.5, 0.25, 0.25),
        metavar=("CORONAL", "SAGITTAL", "AXIAL"),
        help="Multi-plane sampling proportions (coronal, sagittal, axial); must sum "
        "to 1.0 (default: %(default)s -- coronal is the production inference plane)",
    )
    p.add_argument(
        "--mask-ratio",
        type=float,
        default=0.25,
        help="Fraction of the spatial grid to mask per batch (default: %(default)s)",
    )
    p.add_argument(
        "--frangi-sigmas",
        type=int,
        nargs="+",
        default=(1, 2, 3),
        help="Frangi vesselness sigmas for the vessel-aware hole placement (default: %(default)s)",
    )
    p.add_argument(
        "--block-size",
        type=int,
        nargs=2,
        default=(8, 8),
        metavar=("BH", "BW"),
        help="Block mask size in voxels (default: %(default)s)",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of pretraining epochs (default: %(default)s)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size per pretraining step (default: %(default)s)",
    )
    p.add_argument(
        "--steps-per-epoch",
        type=int,
        default=100,
        help="Gradient steps per epoch (each step samples a fresh batch of "
        "random patches -- avoids materializing the whole corpus; "
        "default: %(default)s)",
    )
    p.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(512, 512),
        metavar=("PH", "PW"),
        help="Random patch size cropped from each slice before batching "
        "(the network is fully-convolutional so the state_dict keys are "
        "patch-size-independent; must be large enough to survive the "
        "encoder's 8 stride-2 downsamples without collapsing below 2x2 "
        "at the bottleneck -- InstanceNorm2d requires >1 spatial element "
        "in training mode; default: %(default)s -- fits a 49GB A6000)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for patch sampling (reproducibility; default: %(default)s)",
    )
    p.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Optimizer learning rate (default: %(default)s)",
    )
    p.add_argument(
        "--ddp",
        action="store_true",
        default=False,
        help="Use DistributedDataParallel (requires torchrun env vars RANK/WORLD_SIZE/LOCAL_RANK)",
    )
    p.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="Use AMP mixed precision (no-op on CPU; the scaler disables itself "
        "when CUDA is unavailable)",
    )
    return p


def main() -> None:
    """Run masked-inpainting SSL pretraining on the supplied unlabeled volumes.

    Parses CLI arguments, validates the volume paths exist (``parser.error``
    exits 2 with the offending path on a bad path -- argparse convention),
    configures logging via ``basicConfig`` on the root logger, then imports
    the SSL pipeline callables function-scope (so this module imports without
    the ``[ai,benchmark]`` extra) and runs the pretraining loop.
    """
    parser = _build_argument_parser()
    args = parser.parse_args()

    # Validate volume paths BEFORE importing the heavy SSL stack -- a typo'd
    # path surfaces as a clear parser.error (exit 2) rather than a cryptic
    # zarr/torch traceback from inside the corpus builder.
    for vp in args.volume_paths:
        if not Path(vp).exists():
            parser.error(f"volume path does not exist: {vp}")
    for p_path in (args.plans, args.dataset_json):
        if not Path(p_path).is_file():
            parser.error(f"required file does not exist: {p_path}")

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s %(name)s: %(message)s",
    )

    # Import the SSL pipeline INSIDE main (function-scope) so this module
    # imports cleanly with only the core deps installed. The [ai,benchmark]
    # extra (torch + MONAI + nnunetv2) is required at call time -- each
    # module's top-level torch guard raises ImportError with an actionable
    # message naming the extra if it is absent.
    import json

    import torch

    from liom_toolkit.segmentation.vseg.ssl.corpus import SSLCorpus
    from liom_toolkit.segmentation.vseg.ssl.masking import vessel_aware_block_mask
    from liom_toolkit.segmentation.vseg.ssl.pretrain import (
        build_pretrain_network,
        masked_inpainting_pretrain,
    )

    with Path(args.plans).open(encoding="utf-8") as f:
        plans = json.load(f)
    with Path(args.dataset_json).open(encoding="utf-8") as f:
        dataset_json = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # DDP setup: when --ddp is passed, torchrun injects
    # RANK/WORLD_SIZE/LOCAL_RANK/MASTER_ADDR/MASTER_PORT. Init the process
    # group (nccl on CUDA, gloo on CPU) and pin the device to LOCAL_RANK so
    # each rank drives its own GPU. The pretraining loop wraps the network in
    # DistributedDataParallel and shards the dataset across ranks.
    if args.ddp:
        import os

        import torch.distributed as dist

        missing = [v for v in ("RANK", "WORLD_SIZE", "LOCAL_RANK") if v not in os.environ]
        if missing:
            parser.error(f"--ddp requires torchrun env vars; missing: {missing}")
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")

    # Build the network with output_channels == input_channels so the
    # masked-inpainting reconstruction objective is well-formed (the network
    # reconstructs the masked image, not a segmentation map). The encoder +
    # decoder weights transfer to the production segmentation network
    # unchanged; only the seg_layers (the final classifier head) differ, and
    # load_pretrained_weights skips shape-mismatched keys.
    n_input_channels = len(dataset_json["channel_names"])
    network = build_pretrain_network(
        plans,
        dataset_json,
        configuration="2d",
        device=device,
        output_channels=n_input_channels,
    )

    corpus = SSLCorpus(
        volume_paths=args.volume_paths,
        plane_mix=tuple(args.plane_mix),
    )
    n_corpus = len(corpus)
    if n_corpus == 0:
        parser.error(
            f"corpus is empty -- no slices sampled from {args.volume_paths} "
            "(check the volume paths and the plane-mix config)"
        )

    # Sample random patches on-the-fly via corpus.get_patch(): each step
    # draws a fresh batch of patches, reading ONLY the patch region from disk
    # (dask slicing before .compute()). This is the real-run path --
    # materializing full 2048x2048 slices per patch (the __getitem__ path)
    # would make pretraining disk-bound (~8MB read per slice just to crop a
    # 512x512 patch). The network is fully-convolutional so the state_dict
    # keys are patch-size-independent -- the checkpoint loads into the
    # production network regardless of the patch size used here.
    patch_size = tuple(args.patch_size)

    def _sample_batch() -> torch.Tensor:
        patches = [corpus.get_patch(patch_size) for _ in range(args.batch_size)]
        return torch.stack([torch.as_tensor(p) for p in patches]).to(device)

    batches = [_sample_batch() for _ in range(args.steps_per_epoch)]

    def _mask_transform(batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return vessel_aware_block_mask(
            batch,
            mask_ratio=args.mask_ratio,
            block_size=tuple(args.block_size),
            frangi_sigmas=tuple(args.frangi_sigmas),
        )

    masked_inpainting_pretrain(
        network,
        batches,
        epochs=args.epochs,
        output_path=args.pretrained_output,
        device=device,
        mask_transform=_mask_transform,
        learning_rate=args.learning_rate,
        use_amp=args.amp,
        ddp=args.ddp,
    )
    # Under DDP only rank 0 writes the checkpoint + logs; suppress the log on
    # other ranks to avoid duplicate output.
    if not args.ddp or torch.distributed.get_rank() == 0:
        logger.info("Wrote pretrained checkpoint to %s", args.pretrained_output)
    if args.ddp:
        import torch.distributed as dist

        dist.barrier()
        if dist.get_rank() == 0:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
