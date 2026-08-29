"""Training loop, evaluation, and visualisation for the vessel segmentation U-Net."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

# pandas + scikit-image are moved into the [seg]/[stats] extras (D-01/D-05).
# The upfront ImportError here is the honest signal on an io-only install.
# pandas is shared between [stats] and [antspy]; the message names both
# relevant extras. The `from e` chain preserves the underlying error for
# debugging (AGENTS §2). torch stays function-scope lazy (lines below) --
# do NOT move it to module top.
try:
    import pandas as pd
    from skimage.color import gray2rgb, label2rgb
except ImportError as e:
    raise ImportError(
        "Please install liom-toolkit[seg] and [stats] (or [pipeline]) to use the "
        "vessel segmentation training module."
    ) from e

from .utils import calculate_metrics, create_dir

if TYPE_CHECKING:
    import torch
    from torch import device
    from torch.utils.data import DataLoader

    from .model import VsegModel

logger = logging.getLogger(__name__)


def train(
    model: VsegModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Train the model for an epoch.

    Parameters
    ----------
    model : VsegModel
        The model to train.
    loader : torch.utils.data.DataLoader
        The data loader.
    optimizer : torch.optim.Optimizer
        The optimizer.
    loss_fn : torch.nn.Module
        The loss function.
    device : torch.device
        The device to use for training.
    use_amp : bool
        If True, enable Automatic Mixed Precision via
        ``torch.amp.GradScaler``. The scaler no-ops on CPU/gloo
        (``enabled=use_amp and torch.cuda.is_available()``) so the same
        code path serves both modes. Default ``False`` preserves
        reproducibility for existing notebook callers.

    Returns
    -------
    epoch_loss : float
        The mean epoch loss.
    y : torch.Tensor
        The true labels from the last batch.
    y_pred : torch.Tensor
        The predicted labels from the last batch.
    x : torch.Tensor
        The inputs from the last batch.
    """
    # Initialize epoch loss to 0. Pre-bind the loop variables to None so an
    # empty loader does not raise UnboundLocalError at the return statement
    # (the for-loop body never assigns them when the loader yields nothing).
    epoch_loss = 0.0
    y = None
    y_pred = None
    x = None

    # Put in training mode
    model.train()

    # Iterate over train-loader
    for x, y in tqdm(loader, desc="Training", leave=False, position=1):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Normalize cumulative loss for number of examples. Guard the
    # divide-by-zero when the loader is empty (len(loader) == 0).
    if len(loader) > 0:
        epoch_loss = epoch_loss / len(loader)
    return epoch_loss, y, y_pred, x


def evaluate(
    model: VsegModel, loader: DataLoader, loss_fn: torch.nn.Module, device: torch.device
) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, float, float, float, float]:
    """Evaluate the model for an epoch.

    Parameters
    ----------
    model : VsegModel
        The model to evaluate.
    loader : torch.utils.data.DataLoader
        The data loader.
    loss_fn : torch.nn.Module
        The loss function.
    device : torch.device
        The device to use for evaluation.

    Returns
    -------
    epoch_loss : float
        The mean epoch loss.
    y : torch.Tensor
        The true labels from the last batch.
    y_pred : torch.Tensor
        The predicted labels from the last batch.
    x : torch.Tensor
        The inputs from the last batch.
    f1 : float
        The mean F1 score.
    accuracy : float
        The mean accuracy.
    jaccard : float
        The mean Jaccard score.
    recall : float
        The mean recall.

    Raises
    ------
    ImportError
        If PyTorch is not installed (re-raised with an actionable message).
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "Please install PyTorch to use the vessel segmentation module of the LIOM toolkit."
        ) from e
    # Initialize epoch loss to 0
    # Added metrics. Pre-bind the loop variables to None so an empty loader
    # does not raise UnboundLocalError at the return statement.
    epoch_loss = 0.0
    f1 = 0.0
    accuracy = 0.0
    jaccard = 0.0
    recall = 0.0
    y = None
    y_pred = None
    x = None

    # Put in eval mode
    model.eval()
    with torch.no_grad():
        # Iterate over val-loader
        for x, y in tqdm(loader, desc="Validation", leave=False, position=1):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()
            # metrics
            y_m = y.to("cpu")
            y_m = y_m.numpy()
            y_pred_m = y_pred.to("cpu")
            y_pred_m = y_pred_m.numpy()
            [score_f1, score_recall, score_acc, score_jaccard, _score_precision] = (
                calculate_metrics(y_m, y_pred_m)
            )
            f1 += score_f1
            accuracy += score_acc
            jaccard += score_jaccard
            recall += score_recall

        # Normalize cumulative loss for number of examples. Guard the
        # divide-by-zero when the loader is empty (len(loader) == 0).
        if len(loader) > 0:
            epoch_loss = epoch_loss / len(loader)
            f1 = f1 / len(loader)
            accuracy = accuracy / len(loader)
            jaccard = jaccard / len(loader)
            recall = recall / len(loader)

    return epoch_loss, y, y_pred, x, f1, accuracy, jaccard, recall


def create_images(
    x: torch.Tensor, y: torch.Tensor, pred: torch.Tensor, num_images: int = 4
) -> list[NDArray[np.generic]]:
    """Create images for visualization.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor.
    y : torch.Tensor
        The true labels.
    pred : torch.Tensor
        The predicted labels.
    num_images : int
        The number of images to create.

    Returns
    -------
    list[NDArray[np.generic]]
        The list of visualisation images (RGB composites from ``label2rgb``).
    """
    y_mask = y.cpu().detach().numpy()
    pred_mask = pred.cpu().detach().numpy()
    x = x.cpu().detach().numpy()
    images: list[NDArray[np.generic]] = []
    num_images = int(min(num_images, x.shape[0]))
    i = 0
    while len(images) < num_images:
        img = mask_image(x, y_mask, pred_mask, i)
        images.append(img)
        i += 1

    return images


def mask_image(
    x: NDArray[np.generic],
    y_mask: NDArray[np.floating],
    pred_mask: NDArray[np.floating],
    i: int,
) -> NDArray[np.generic]:
    """Overlay the ground-truth and predicted masks on a single input image.

    Parameters
    ----------
    x : NDArray[np.generic]
        The input image stack.
    y_mask : NDArray[np.generic]
        The ground-truth mask stack.
    pred_mask : NDArray[np.generic]
        The predicted mask stack.
    i : int
        The image index within the stacks.

    Returns
    -------
    NDArray[np.generic]
        The RGB-overlaid image (output of ``skimage.color.label2rgb``).
    """
    img = x[i, :, :, :].squeeze()
    img = gray2rgb(img)
    y_mask = y_mask[i, :, :, :].squeeze()
    pred_mask = pred_mask[i, :, :, :].squeeze()

    np.place(y_mask, y_mask >= 0.5, 1)
    np.place(y_mask, y_mask < 0.5, 0)

    diff_mask = (pred_mask - y_mask) * pred_mask
    np.place(diff_mask, diff_mask > 0.5, 1)
    np.place(diff_mask, diff_mask <= 0.5, 0)

    np.place(pred_mask, pred_mask > 0.5, 1)
    np.place(pred_mask, pred_mask <= 0.5, 0)

    diff_mask = diff_mask * 3
    pred_mask = (pred_mask - diff_mask) * 2

    labels = np.max([y_mask, diff_mask, pred_mask], axis=0)
    return label2rgb(
        labels,
        image=img,
        colors=[[0, 0, 1], [0, 1, 0], [1, 0, 0]],
        alpha=0.3,
        bg_label=0,
        bg_color=None,
    )


def train_model(
    dataset_file: str,
    node_name: str,
    dev: device | None = None,
    output_train: str = "training",
    learning_rate: float = 0.003673,
    batch_size: int = 35,
    epochs: int = 62,
    wandb_mode: str = "offline",
    filter_empty_patches: bool = True,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    pretrained_artifact: str | None = None,
    pin_memory: bool = True,
    resume: bool = False,
    ddp: bool = False,
    use_amp: bool = False,
) -> None:
    """Train the vessel segmentation model.

    Parameters
    ----------
    dataset_file : str
        The file to the dataset (zarr).
    node_name : str
        The name of the node in the zarr file.
    dev : torch.device | None
        The device to use for training. ``None`` resolves to ``torch.device("cuda")``
        inside the function body (avoids a def-time ``torch.device`` call).
        Under DDP, device mapping is via ``torch.cuda.set_device(LOCAL_RANK)``
        (guarded behind ``torch.cuda.is_available()``).
    output_train : str
        The output directory for the training.
    learning_rate : float
        The learning rate for the optimizer.
    batch_size : int
        The batch size for training.
    epochs : int
        The number of epochs to train.
    wandb_mode : str
        The mode for wandb.
    wandb_project : str | None
        The wandb project name. ``None`` lets wandb use its own default
        (the toolkit is lab-config-free on import).
    wandb_entity : str | None
        The wandb entity (team/user) name. ``None`` lets wandb use the
        user's default entity (no hardcoded lab entity).
    pretrained_artifact : str | None
        The wandb artifact path (``"entity/project/name:version"``) of a
        pretrained model to initialise from. ``None`` trains from scratch.
        When non-None, threads through to ``VsegModel(pretrained=True,
        pretrained_artifact=...)``.
    filter_empty_patches : bool
        Whether to filter empty patches.
    pin_memory : bool
        Whether to pin memory in the data loader. Speeds up for CUDA.
    resume : bool
        If True, resume from a previous checkpoint. The manifest's
        ``last_completed_epoch`` is read and the model loads
        ``checkpoint.{last_completed_epoch}.pth`` (the existing per-epoch
        weights artifact); the epoch loop continues from
        ``last_completed_epoch + 1``. A params-hash mismatch (code/param
        change between runs) invalidates the checkpoint and re-runs from
        epoch 0.

        .. note::
            1.0.0 limitation: full-state ``.pth`` augmentation
            (optimizer / scheduler / RNG / dataloader-epoch state) is
            deferred to 1.1. Resume continues from epoch ``N+1`` with a
            re-initialized optimizer state — this is a known, documented
            limitation, not a silent wrong-data fallback. The manifest
            records the epoch index (complementary to the per-epoch
            ``checkpoint.*.pth`` weights artifact); the ``.pth`` is the
            weights, the manifest is the bookkeeper.
    ddp : bool
        If True, enable DistributedDataParallel training. Requires the
        ``torchrun`` env vars (``RANK``/``WORLD_SIZE``/``LOCAL_RANK``) to
        be set — raises :class:`RuntimeError` naming the missing env state
        if any is absent (no silent single-process fallback). When the env
        vars are present, calls ``dist.init_process_group()`` (no backend
        arg — torchrun injects everything; auto-selects gloo/nccl), guards
        ``torch.cuda.set_device(int(LOCAL_RANK))`` behind
        ``torch.cuda.is_available()`` (CPU-only torch raises
        ``AttributeError`` otherwise), wraps the model with
        ``DistributedDataParallel`` (guarded by ``isinstance(model, DDP)``
        to prevent double-wrapping), and uses
        :class:`~liom_toolkit.utils.checkpoint.DDPResumeManager` for
        rank-0-only manifest/``.pth`` writes + post-restore barrier.
    use_amp : bool
        If True, enable Automatic Mixed Precision via
        ``torch.amp.GradScaler``. The scaler no-ops on CPU/gloo
        (``enabled=use_amp and torch.cuda.is_available()``) so the same
        code path serves both modes. Default ``False`` preserves
        reproducibility for existing notebook callers (a default-on AMP
        change would silently shift every notebook user's val_loss curve).

    Raises
    ------
    ImportError
        If PyTorch (or wandb) is not installed (re-raised with an actionable message).
    RuntimeError
        If ``ddp=True`` but the ``torchrun`` env vars
        (``RANK``/``WORLD_SIZE``/``LOCAL_RANK``) are not set.
    """
    try:
        import torch
        from torch.utils.data import DataLoader, Subset, random_split

        from .dataset import OmeZarrLabelDataSet
        from .loss import DiceBCELoss
        from .model import VsegModel
    except ImportError as e:
        raise ImportError(
            "Please install PyTorch to use the vessel segmentation module of the LIOM toolkit."
        ) from e
    try:
        import wandb
    except ImportError as e:
        raise ImportError(
            "Please install wandb (ai extra) to use the vessel segmentation "
            "training of the LIOM toolkit."
        ) from e
    if dev is None:
        dev = torch.device("cuda")

    # --- DDP entry (D-01/D-01a) -----------------------------------------
    # When ddp=True, require the torchrun env vars (RANK/WORLD_SIZE/
    # LOCAL_RANK). Missing any → raise with the missing-var name (no silent
    # single-process fallback — AGENTS §2). When present, init the process
    # group (no backend arg — torchrun injects everything; auto-selects
    # gloo on CPU / nccl on CUDA) and set the device. set_device is guarded
    # behind torch.cuda.is_available() because CPU-only torch raises
    # AttributeError on torch.cuda.set_device (the 2-rank CPU gloo smoke
    # cannot run without this guard).
    if ddp:
        missing = [v for v in ("RANK", "WORLD_SIZE", "LOCAL_RANK") if v not in os.environ]
        if missing:
            raise RuntimeError(
                f"--ddp requires torchrun env vars (RANK/WORLD_SIZE/LOCAL_RANK); missing: {missing}"
            )
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel
        from torch.utils.data import DistributedSampler

        dist.init_process_group()  # no backend arg — torchrun injects everything
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        rank = dist.get_rank()
        if torch.cuda.is_available():
            dev = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    else:
        rank = 0
        dist = None
        DistributedDataParallel = None
        DistributedSampler = None

    # Setup training parameters and wandb run
    hyperparameter_defaults = {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
    }

    # Init wandb. entity=wandb_entity (None -> wandb uses the user's default
    # entity; project=wandb_project (None -> wandb default project). No
    # hardcoded lab config. Under DDP, only rank 0 inits a wandb run (the
    # rank-0-only W&B invariant — D-01); other ranks use a disabled run so
    # wandb.log/watch/finish calls are safe no-ops on non-rank-0.
    if ddp and rank != 0:
        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            mode="disabled",
            config=hyperparameter_defaults,
        )
    else:
        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            mode=wandb_mode,
            config=hyperparameter_defaults,
        )

    config = wandb.config

    # Load the dataset
    full_dataset = OmeZarrLabelDataSet(
        dataset_file,
        node_name,
        device=dev,
        pre_process=False,
        patch_size=(1, 256, 256),
        filter_empty=filter_empty_patches,
        normalise_label=False,
    )
    train_dataset, test_dataset = random_split(full_dataset, [0.8, 0.2])

    if filter_empty_patches:
        # Map the split indices through valid_indices to get the actual
        # dataset indices. random_split returns Subset objects whose
        # .indices are integers in 0..len(valid_indices)-1 (i.e. indices
        # INTO the filtered dataset, since full_dataset.__len__ returns
        # len(valid_indices) when filter_empty is set). The previous code
        # checked whether a filtered-dataset index was a VALUE in
        # valid_indices (which contains original dataset indices) -- a
        # namespace mismatch that succeeded only by numerical coincidence
        # and produced a tiny, non-random training subset. Map each split
        # index through valid_indices to recover the true dataset index.
        train_valid_indices = [
            int(full_dataset.valid_indices[idx]) for idx in train_dataset.indices
        ]
        test_valid_indices = [int(full_dataset.valid_indices[idx]) for idx in test_dataset.indices]

        # Create new subsets for dataloaders
        train_dataset = Subset(full_dataset, train_valid_indices)
        test_dataset = Subset(full_dataset, test_valid_indices)

    # Create data loaders. Under DDP, wrap both loaders with
    # DistributedSampler (D-02b: set_epoch called per-epoch on the train
    # sampler so each epoch sees a different shuffle across ranks). The val
    # loader also uses DistributedSampler (all-rank eval, D-02) — the
    # sum+count all-reduce in evaluate() handles len(val_set) % world_size
    # != 0.
    if ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(test_dataset, shuffle=False)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=0,
            pin_memory=pin_memory,
        )
        validation_loader = DataLoader(
            test_dataset,
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
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        )
        validation_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        )

    # Setup check point dir
    best_epoch = -1
    create_dir(f"{output_train}")
    create_dir(f"{output_train}/files")
    checkpoint_path = f"{output_train}/files/checkpoint"
    create_dir(f"{output_train}/patch_seg")

    # Resume bookkeeping: the manifest records last_completed_epoch
    # (complementary to the per-epoch checkpoint.*.pth weights artifact).
    # The manifest is the bookkeeper; the .pth is the weights.
    #
    # NOTE: steps_total=epochs is passed for manifest completeness, but
    # train_model does NOT call start_step/finish_step per epoch — it uses
    # set_last_completed_epoch + the complete sentinel instead. The
    # completed_steps set in the manifest is therefore always empty for
    # this pipeline; last_completed_epoch is the authoritative epoch index.
    # Future maintainers should not expect completed_steps to track epoch
    # completion here.
    from liom_toolkit.utils.checkpoint import DDPResumeManager, ResumeManager

    # Under DDP, use DDPResumeManager (rank-0-only writes + post-restore
    # barrier + model.module unwrap). Single-process callers get the base
    # ResumeManager unchanged (D-03 — the base class stays
    # single-process-agnostic).
    resume_mgr_cls = DDPResumeManager if ddp else ResumeManager
    resume_mgr = resume_mgr_cls(
        output_dir=Path(output_train),
        pipeline="train_model",
        params={
            "dataset_file": dataset_file,
            "node_name": node_name,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            # Include training-affecting parameters so a config change
            # between runs invalidates the checkpoint (params-hash
            # mismatch). pretrained_artifact determines the initial
            # weights (from-scratch vs. fine-tuning); filter_empty_patches
            # changes which patches are used; dev (CPU vs. GPU) affects
            # floating-point reproducibility; pin_memory changes data
            # loading behavior. Without these, resume with a different
            # training config would silently continue from a checkpoint
            # that was initialized differently — a stale-checkpoint bug.
            "pretrained_artifact": pretrained_artifact,
            "filter_empty_patches": filter_empty_patches,
            "dev": str(dev) if dev is not None else None,
            "pin_memory": pin_memory,
            "ddp": ddp,
            "use_amp": use_amp,
        },
        steps_total=epochs,
    )
    if resume and resume_mgr.is_complete():
        logger.info("train_model: checkpoint complete, nothing to do.")
        if rank == 0:
            run.finish()
        return

    # Initialise the model. pretrained_artifact threads through to VsegModel;
    # None trains from scratch (no silent fallback to a hardcoded lab artifact).
    model = VsegModel(
        pretrained=pretrained_artifact is not None,
        pretrained_artifact=pretrained_artifact,
    )

    # Resume: load the per-epoch checkpoint.{last_completed_epoch}.pth weights
    # and continue from epoch N+1. 1.0.0 limitation: the optimizer / scheduler
    # / RNG state is re-initialized (full-state .pth augmentation is deferred
    # to 1.1 — resume is not bit-deterministic across optimizer/RNG state
    # until 1.1). This is a known, documented limitation, not a silent
    # wrong-data fallback. Under DDP, DDPResumeManager.restore_weights loads
    # on rank 0 then barriers so all ranks sync before training.
    start_epoch = 0
    if resume:
        last_epoch = resume_mgr.get_last_completed_epoch()
        if last_epoch is not None:
            ckpt_file = f"{checkpoint_path}.epoch_{last_epoch}.pth"
            if Path(ckpt_file).exists():
                # weights_only=True restricts deserialization to tensor
                # storage (no arbitrary pickle code execution). PyTorch 2.6+
                # defaults to this, but the ai extra does not pin a torch
                # version, so a user on torch < 2.6 would otherwise load with
                # weights_only=False (pickle.load under the hood) — a
                # malicious or swapped .pth would execute arbitrary code.
                # Match the model.py load path (weights_only=True).
                if ddp:
                    # DDPResumeManager handles rank-0 load + post-restore
                    # barrier (single barrier after rank-0 save; the next
                    # backward AllReduce is the subsequent sync point).
                    resume_mgr.restore_weights(model, ckpt_file)
                else:
                    model.load_state_dict(torch.load(ckpt_file, weights_only=True))
                start_epoch = last_epoch + 1
                logger.info(
                    "train_model: resuming from epoch %d (loaded %s).",
                    start_epoch,
                    ckpt_file,
                )
            else:
                logger.warning(
                    "train_model: manifest says last_completed_epoch=%d but "
                    "%s is missing — starting from epoch 0.",
                    last_epoch,
                    ckpt_file,
                )

    model = model.to(dev)

    # Wrap with DDP after the model is on the device. The isinstance guard
    # prevents double-wrapping so Phase 15 can reuse the identical entry
    # path without an API undo. device_ids=None is mandatory on CPU/gloo.
    if ddp and not isinstance(model, DistributedDataParallel):
        model = DistributedDataParallel(model, device_ids=None)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5)
    loss_fn = DiceBCELoss()

    # Track model with Wandb — rank-0 only (D-01 W&B invariant).
    if rank == 0:
        wandb.watch(model, criterion=loss_fn, log="all", log_freq=25, log_graph=False)

    """ Training the model """
    train_losses = []
    val_losses = []

    best_valid_loss = float("inf")

    epoch_range = range(start_epoch, config.epochs)
    for epoch in (pbar := tqdm(epoch_range, desc="Epochs", leave=False, position=0)):
        # D-02b: set_epoch per-epoch so each epoch sees a different shuffle
        # across ranks. No-op when train_sampler is None (single-process).
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss, train_y, train_y_pred, x_train = train(
            model, train_loader, optimizer, loss_fn, dev, use_amp=use_amp
        )

        val_loss, val_y, val_y_pred, x_val, f1_score, accuracy, jaccard, recall = evaluate(
            model, validation_loader, loss_fn, dev
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        """ Saving model """
        if val_loss < best_valid_loss:
            best_epoch = epoch
            # DDPResumeManager.save_weights unwraps model.module for DDP and
            # is rank-0-only; base ResumeManager path uses torch.save
            # directly (single-process unchanged).
            if ddp:
                resume_mgr.save_weights(model, f"{checkpoint_path}.latest.pth")
            else:
                torch.save(model.state_dict(), f"{checkpoint_path}.latest.pth")
            best_valid_loss = val_loss

        if epoch % 10 == 0:
            if ddp:
                resume_mgr.save_weights(model, f"{checkpoint_path}.epoch_{epoch}.pth")
            else:
                torch.save(model.state_dict(), f"{checkpoint_path}.epoch_{epoch}.pth")

        # Record the epoch as complete (complementary to the per-epoch .pth).
        # A crash after epoch N leaves last_completed_epoch=N; resume
        # continues from N+1. DDPResumeManager no-ops on non-rank-0.
        resume_mgr.set_last_completed_epoch(epoch)

        pbar.set_postfix(loss=f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        train_images = create_images(x_train, train_y, train_y_pred)
        val_images = create_images(x_val, val_y, val_y_pred)
        # wandb.log rank-0 only (D-01). On non-rank-0 the run is disabled so
        # this is a safe no-op, but the explicit guard makes the invariant
        # auditable.
        if rank == 0:
            wandb.log(
                {
                    "Training Loss": train_loss,
                    "Validation Loss": val_loss,
                    "Accuracy": accuracy,
                    "Jaccard": jaccard,
                    "F1 score": f1_score,
                    "Recall": recall,
                    "Images": {
                        "Training": [wandb.Image(x, mode="RGB") for x in train_images],
                        "Validation": [wandb.Image(x, mode="RGB") for x in val_images],
                    },
                }
            )

    # Atomic complete sentinel — written LAST, after all epochs done. A crash
    # on the final epoch does NOT leave complete=True (write_manifest is
    # atomic: temp + Path.replace). DDPResumeManager no-ops on non-rank-0.
    resume_mgr.mark_complete()

    logger.info("Finished Training: Best Epoch = %s", best_epoch)
    # W&B artifact + finish: rank-0 only (D-01).
    if rank == 0:
        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(f"{checkpoint_path}.latest.pth")
        run.log_artifact(artifact)

        # For sweeps
        if ddp:
            resume_mgr.save_weights(model, str(Path(wandb.run.dir) / "model.pt"))
        else:
            torch.save(model.state_dict(), str(Path(wandb.run.dir) / "model.pt"))
        run.finish()

    # final_metrics.csv relocated to Path(output_train)/final_metrics.csv
    # (D-03b — eliminates the concurrent-run CWD collision). Rank-0 only
    # under DDP (only rank 0 has the complete train_losses/val_losses; the
    # all-reduce in evaluate() gives every rank the same val_loss but the
    # train_loss accumulation is per-rank — write on rank 0 to avoid the
    # collision the relocation was meant to fix).
    if rank == 0:
        final_loss = pd.DataFrame(data=[train_losses, val_losses]).T
        final_loss.to_csv(Path(output_train) / "final_metrics.csv", encoding="utf-8", index=False)
