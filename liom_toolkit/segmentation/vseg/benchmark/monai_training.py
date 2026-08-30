"""Dedicated MONAI training loop for the benchmark contenders.

The legacy ``train_model`` in ``training.py`` is tightly coupled to the
zarr-based ``OmeZarrLabelDataSet``, ``VsegModel``, ``DiceBCELoss``, ``Adam``,
and ``ReduceLROnPlateau``. The MONAI contenders need a different stack:
PNG-slice dataset, MONAI ``UNet`` / ``SwinUNETR``, composite
``DiceFocalLoss`` + ``SoftclDiceLoss``, ``AdamW``, and a cosine schedule.

Rather than thread five new factories through ``train_model`` (which would
essentially rewrite it), this module provides a dedicated
:func:`train_monai_model` that replicates the DDP entry contract from
``train_model`` verbatim — env-var auto-detect, explicit backend selection,
``torch.cuda.set_device`` guard, ``DistributedSampler`` with ``set_epoch``,
no-double-wrap ``DistributedDataParallel`` guard, sum+count ``all_reduce``
for validation loss, rank-0-only checkpoint writes, AMP + grad-clip
ordering, and ``destroy_process_group`` in ``finally``.

The trained ``state_dict`` is saved to
``{output_dir}/files/checkpoint.latest.pth`` so the contender's
``predict_on_slices`` (which already loads from that path) picks up the
trained weights — the fix for the silent-random-weights bug where the
MONAI contenders previously predicted with untrained models.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# torch is in the [ai] extra. The upfront ImportError is the honest signal
# on an io-only install — the message names [ai]. The `from e` chain
# preserves the underlying error for debugging (AGENTS §2).
try:
    import torch
    from torch.utils.data import Dataset
except ImportError as e:
    raise ImportError(
        "Please install liom-toolkit[ai] to use the MONAI benchmark training loop."
    ) from e

__all__ = ["train_monai_model"]


def _mask_path_for(slice_path: str) -> str:
    """Resolve the ``<name>_mask.png`` label path for a slice.

    The benchmark dataset convention (shared with
    ``liom_prepare_nnunet_dataset._discover_pairs``) pairs ``<name>.png``
    images with ``<name>_mask.png`` labels. This helper centralises the
    stem-suffix logic so the dataset and the test harness agree.

    Parameters
    ----------
    slice_path : str
        Path to the image slice.

    Returns
    -------
    str
        Path to the matching ``<stem>_mask.png`` label.

    Raises
    ------
    ValueError
        If the label file does not exist (no silent fallback to an
        all-zero mask — AGENTS §2).
    """
    p = Path(slice_path)
    mask = p.with_name(f"{p.stem}_mask{p.suffix}")
    if not mask.is_file():
        raise ValueError(
            f"train_monai_model: no matching label for {slice_path} — "
            f"expected {mask} (the <name>_mask.png convention)"
        )
    return str(mask)


class _MonaiPngDataset(Dataset):
    """PNG (image, mask) dataset for MONAI training.

    Reads each image slice and its ``<stem>_mask.png`` label, normalises
    the image to ``[0, 1]`` float32, binarises the mask at 0.5, and applies
    a random crop to ``patch_size`` for training augmentation. Both tensors
    are channel-first ``(1, H, W)`` — the layout the MONAI ``UNet`` /
    ``SwinUNETR`` forward expects.

    The mask is binarised to ``{0.0, 1.0}`` float32 (NOT 0/255 uint8) so
    the composite loss (``DiceFocalLoss(sigmoid=True)`` +
    ``SoftclDiceLoss(sigmoid=True)``) receives the target shape its
    internals assume — a 0/255 target would scale the Dice numerator by
    255 and silently produce wrong gradients (AGENTS §2).
    """

    def __init__(
        self,
        slice_paths: list[str],
        patch_size: tuple[int, int],
        *,
        crop: bool = True,
    ) -> None:
        self.slice_paths = list(slice_paths)
        self.patch_size = patch_size
        self.crop = crop
        # Pre-resolve + validate mask paths so a missing label surfaces
        # before training starts (not mid-epoch in a worker process).
        self.mask_paths = [_mask_path_for(p) for p in self.slice_paths]

    def __len__(self) -> int:
        return len(self.slice_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        import imageio.v3 as iio

        img = np.asarray(iio.imread(self.slice_paths[idx]), dtype=np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        mask = np.asarray(iio.imread(self.mask_paths[idx]), dtype=np.float32)
        if mask.max() > 1.0:
            mask = mask / 255.0
        mask = (mask > 0.5).astype(np.float32)

        # (H, W) → (1, H, W) — channel-first for MONAI.
        img_t = torch.from_numpy(img).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        if self.crop:
            img_t, mask_t = _random_crop(img_t, mask_t, self.patch_size)
        return img_t, mask_t


def _random_crop(
    img: torch.Tensor, mask: torch.Tensor, patch_size: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random-crop (or pad) an (1, H, W) image + mask to ``patch_size``.

    If the slice is larger than ``patch_size``, a random crop is taken
    (training augmentation). If smaller, the slice is zero-padded so the
    spatial dims are exactly ``patch_size`` — the MONAI ``UNet`` requires
    divisibility by ``2^len(strides)`` and ``SwinUNETR`` by 32; the
    benchmark patch size ``(256, 256)`` satisfies both.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        The cropped (image, mask) pair, each ``(1, ph, pw)``.
    """
    _, h, w = img.shape
    ph, pw = patch_size
    if h >= ph and w >= pw:
        top = random.randint(0, h - ph)  # ruff: ignore[suspicious-non-cryptographic-random-usage] - non-cryptographic crop
        left = random.randint(0, w - pw)  # ruff: ignore[suspicious-non-cryptographic-random-usage] - non-cryptographic crop
        return img[:, top : top + ph, left : left + pw], mask[:, top : top + ph, left : left + pw]
    # Pad bottom/right to reach patch_size.
    pad_w = max(0, pw - w)
    pad_h = max(0, ph - h)
    return (
        torch.nn.functional.pad(img, (0, pad_w, 0, pad_h)),
        torch.nn.functional.pad(mask, (0, pad_w, 0, pad_h)),
    )


def _composite_loss(lambda_cldice: float = 0.5) -> torch.nn.Module:
    """Build the composite DiceFocal + soft-clDice loss.

    Both components expect LOGITS (``sigmoid=True`` applies sigmoid
    internally — the model forward must NOT also apply sigmoid, or the
    loss computes ``sigmoid(sigmoid(logits))`` → wrong gradients, the
    silent-wrong-data failure mode). soft-clDice is NEVER used alone — it
    is combined with DiceFocal per the architecture decision.

    Returns
    -------
    torch.nn.Module
        A callable composite loss: ``loss(logits, target) -> scalar``.
    """
    from monai.losses import DiceFocalLoss, SoftclDiceLoss

    dice_focal = DiceFocalLoss(
        sigmoid=True,
        lambda_dice=1.0,
        lambda_focal=1.0,
        gamma=2.0,
    )
    soft_cldice = SoftclDiceLoss(sigmoid=True, iter_=3)

    class _CompositeLoss(torch.nn.Module):
        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return dice_focal(pred, target) + lambda_cldice * soft_cldice(pred, target)

    return _CompositeLoss()


def _unwrap_state_dict(model: torch.nn.Module, ddp_cls: type | None) -> dict[str, torch.Tensor]:
    """Extract ``state_dict`` from a model, unwrapping DDP if needed.

    Under DDP, ``model.module`` holds the underlying model; without DDP,
    ``model`` itself is the model. The ``ddp_cls`` argument is the
    ``DistributedDataParallel`` class (or ``None`` when not using DDP),
    used for the ``isinstance`` guard.

    Returns
    -------
    dict[str, torch.Tensor]
        The unwrapped model's ``state_dict``.
    """
    if ddp_cls is not None and isinstance(model, ddp_cls):
        return model.module.state_dict()
    return model.state_dict()


def train_monai_model(
    model: torch.nn.Module,
    train_slices: list[str],
    output_dir: str,
    *,
    patch_size: tuple[int, int, int] = (1, 256, 256),
    epochs: int = 50,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    lambda_cldice: float = 0.5,
    ddp: bool = False,
    use_amp: bool = True,
    val_split: float = 0.2,
    device: torch.device | None = None,
) -> Path:
    """Train a MONAI model on PNG slices and save ``checkpoint.latest.pth``.

    Replicates the DDP entry contract from ``train_model`` verbatim:
    env-var auto-detect (``RANK`` / ``WORLD_SIZE`` / ``LOCAL_RANK``),
    explicit backend selection (``nccl`` on CUDA, ``gloo`` on CPU),
    ``torch.cuda.set_device`` guarded by ``torch.cuda.is_available()``,
    ``DistributedSampler`` with ``set_epoch``, no-double-wrap
    ``DistributedDataParallel`` guard, sum+count ``all_reduce`` for
    validation loss, rank-0-only checkpoint writes, AMP + grad-clip
    ordering (``scale → backward → unscale → clip → step → update``),
    and ``destroy_process_group`` in ``finally``.

    The trained ``state_dict`` is saved to
    ``{output_dir}/files/checkpoint.latest.pth`` (best validation loss)
    and ``{output_dir}/files/checkpoint.epoch_{N}.pth`` every 10 epochs.
    The contender's ``predict_on_slices`` loads
    ``checkpoint.latest.pth`` into a fresh model — the fix for the
    silent-random-weights bug.

    Parameters
    ----------
    model : torch.nn.Module
        The MONAI model (``UNet`` or ``SwinUNETR``) to train. Built by
        the contender's ``_build_model()``.
    train_slices : list[str]
        PNG slice paths to train on. Each ``<name>.png`` must have a
        matching ``<name>_mask.png`` label.
    output_dir : str
        Directory to write checkpoints into (``{output_dir}/files/``).
    patch_size : tuple[int, int, int]
        The (Z, Y, X) patch size — only the (Y, X) spatial dims are
        used for the random crop. Default ``(1, 256, 256)``.
    epochs : int
        Number of training epochs. Default 50.
    batch_size : int
        Training batch size. Default 4.
    learning_rate : float
        AdamW learning rate. Default 1e-4.
    lambda_cldice : float
        Soft-clDice weight in the composite loss
        (``loss = DiceFocal + lambda_cldice * soft_clDice``). Default 0.5.
    ddp : bool
        If True, enable DistributedDataParallel training. Requires the
        ``torchrun`` env vars.
    use_amp : bool
        If True, enable Automatic Mixed Precision. The scaler no-ops on
        CPU/gloo. Default True.
    val_split : float
        Fraction of train_slices held out for validation (best-val-loss
        checkpoint selection). Default 0.2.
    device : torch.device | None
        The device to use. ``None`` resolves to ``cuda`` if available,
        else ``cpu``. Under DDP, overridden by ``cuda:{LOCAL_RANK}`` or
        ``cpu``.

    Returns
    -------
    Path
        The path to ``checkpoint.latest.pth``.

    Raises
    ------
    RuntimeError
        If ``ddp=True`` but the ``torchrun`` env vars are not set.
    ValueError
        If ``train_slices`` is empty, or any slice has no matching
        ``<name>_mask.png`` label.
    """
    if not train_slices:
        raise ValueError("train_monai_model: train_slices is empty — cannot train on zero slices")

    # Resolve device (CPU default when no CUDA, so the test suite runs
    # without a GPU).
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- DDP entry (replicated from train_model) -------------------------
    dist = None
    DistributedDataParallel = None
    DistributedSampler = None
    rank = 0

    if ddp:
        missing = [v for v in ("RANK", "WORLD_SIZE", "LOCAL_RANK") if v not in os.environ]
        if missing:
            raise RuntimeError(
                f"--ddp requires torchrun env vars (RANK/WORLD_SIZE/LOCAL_RANK); missing: {missing}"
            )
        import torch.distributed as dist_mod
        from torch.nn.parallel import DistributedDataParallel as DDP_cls
        from torch.utils.data import DistributedSampler as DSamp

        dist = dist_mod
        DistributedDataParallel = DDP_cls
        DistributedSampler = DSamp

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        else:
            device = torch.device("cpu")
        rank = dist.get_rank()

    try:
        from torch.utils.data import DataLoader, random_split

        spatial = (patch_size[1], patch_size[2])
        full_dataset = _MonaiPngDataset(train_slices, spatial, crop=True)
        n_val = max(1, int(len(full_dataset) * val_split))
        n_train = len(full_dataset) - n_val
        train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])

        pin_memory = device.type != "cpu"
        if ddp:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                num_workers=0,
                pin_memory=pin_memory,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                sampler=val_sampler,
                num_workers=0,
                pin_memory=pin_memory,
            )
        else:
            train_sampler = None
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=pin_memory,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=pin_memory,
            )

        model = model.to(device)
        already_wrapped = DistributedDataParallel is not None and isinstance(
            model, DistributedDataParallel
        )
        if ddp and not already_wrapped:
            device_ids = [int(os.environ["LOCAL_RANK"])] if torch.cuda.is_available() else None
            model = DistributedDataParallel(model, device_ids=device_ids)

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        loss_fn = _composite_loss(lambda_cldice)

        scaler = torch.amp.GradScaler(enabled=use_amp and torch.cuda.is_available())
        max_norm = 1.0

        ckpt_dir = Path(output_dir) / "files"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_latest = ckpt_dir / "checkpoint.latest.pth"

        best_val_loss = float("inf")
        for epoch in range(epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            model.train()
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()

            # Validation: sum+count all-reduce under DDP (NOT mean-of-means).
            model.eval()
            val_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)
                    y_pred = model(x)
                    vloss = loss_fn(y_pred, y)
                    val_sum += float(vloss.item())
                    val_count += 1
            if ddp and dist is not None and dist.get_world_size() > 1:
                pair = torch.tensor(
                    [val_sum, float(val_count)],
                    dtype=torch.float64,
                    device=device,
                )
                dist.all_reduce(pair, op=dist.ReduceOp.SUM)
                val_sum = float(pair[0].item())
                val_count = int(pair[1].item())
            val_loss = val_sum / val_count if val_count > 0 else float("inf")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if rank == 0:
                    torch.save(_unwrap_state_dict(model, DistributedDataParallel), str(ckpt_latest))
            if rank == 0 and epoch % 10 == 0:
                torch.save(
                    _unwrap_state_dict(model, DistributedDataParallel),
                    str(ckpt_dir / f"checkpoint.epoch_{epoch}.pth"),
                )

        # Guarantee a checkpoint exists even if val_count was 0 (e.g. a
        # single-slice dataset where val_split leaves 0 val slices). Save
        # the final-epoch weights so predict_on_slices has something to
        # load — never silently proceed with random weights (AGENTS §2).
        if rank == 0 and not ckpt_latest.exists():
            torch.save(_unwrap_state_dict(model, DistributedDataParallel), str(ckpt_latest))

        return ckpt_latest
    finally:
        if ddp and dist is not None:
            dist.destroy_process_group()
