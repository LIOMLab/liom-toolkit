"""Masked-inpainting pretraining loop for the nnU-Net v2 2D ResEnc U-Net.

This module is the pretraining layer of the SSL stack: it builds the nnU-Net
2D ResEnc network via ``get_network_from_plans`` (so the saved checkpoint's
state_dict keys are EXACTLY the keys ``load_pretrained_weights`` expects --
the key-match guarantee that makes the warm-start load with no
``AssertionError``), runs a masked-inpainting reconstruction objective (MSE
on the masked regions vs the original unmasked image), and saves the
checkpoint as ``{'network_weights': network.state_dict()}`` -- the format
the warm-start loader consumes.

Why ``get_network_from_plans`` and not a generic MONAI UNet: the upstream
``load_pretrained_weights`` matches state_dict keys by name + shape and
skips only ``.seg_layers.`` (the segmentation heads). A generic MONAI UNet's
keys (different module names, different architecture) would NOT match and
would raise ``AssertionError`` at warm-start load. Building the pretraining
network with the SAME ``get_network_from_plans`` call the warm-start uses
guarantees the keys match by construction.

The mask transform is a pluggable parameter: the default uses MONAI
``RandCoarseDropoutd`` for 2D block masking (channel-preserving via
``spatial_size=(-1, H, W)``); a later plan supplies the vessel-aware
(Frangi/Hessian-biased) variant without editing this module.

nnunetv2 is in the ``[ai]`` extra and MONAI is in the ``[benchmark]`` extra;
both are imported function-scope so this module loads with only torch
installed. Validation uses ``if ...: raise ValueError(...)`` with the
offending value in the message (AGENTS section 2 -- never ``assert`` for
validation, never a silent zero-fill / NaN fallback on an empty corpus).
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

# torch is in the [ai] extra. The upfront ImportError here is the honest
# signal on an io-only install -- the message names [ai,benchmark] (the
# torch + MONAI path the SSL pretraining stack needs). The `from e` chain
# preserves the underlying error for debugging (AGENTS section 2). MONAI
# and nnunetv2 are imported function-scope so this module loads with only
# torch installed.
try:
    import torch
    from torch import nn
except ImportError as e:  # pragma: no cover - exercised only on io-only installs
    raise ImportError(
        "Please install liom-toolkit[ai,benchmark] to use the SSL pretraining loop."
    ) from e

__all__ = ["build_pretrain_network", "masked_inpainting_pretrain"]


# A mask transform maps a batched input tensor (B, C, H, W) to a tuple of
# (masked_input, mask) where ``masked_input`` is the network input (holes
# zero-filled) and ``mask`` is a boolean tensor (B, C, H, W) flagging the
# regions the network must reconstruct (True = masked / dropout region).
# The reconstruction target is the ORIGINAL unmasked image; the loss is MSE
# over the masked regions. Plugging in a vessel-aware (Frangi/Hessian-biased)
# mask transform is how a later plan biases hole placement toward thin-
# structure regions without editing this loop.
MaskTransform = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def build_pretrain_network(
    plans: dict[str, Any],
    dataset_json: dict[str, Any],
    configuration: str = "2d",
    device: torch.device | str = "cpu",
    deep_supervision: bool = False,
    allow_init: bool = True,
    output_channels: int | None = None,
) -> nn.Module:
    """Build the nnU-Net 2D ResEnc network via ``get_network_from_plans``.

    The network is constructed from the architecture block in ``plans`` so
    its state_dict keys are exactly the keys ``load_pretrained_weights``
    expects (``encoder.*``, ``decoder.*``, ``decoder.seg_layers.*``). This
    is the key-match guarantee: a checkpoint saved from a network built by
    this function loads into a second network built by this function (with
    the same plan) with NO ``AssertionError``.

    Parameters
    ----------
    plans : dict
        Either a full nnU-Net plans dict (with a ``configurations`` mapping
        whose ``configuration`` entry carries an ``architecture`` block) or
        a bare architecture block (``arch_class_name`` / ``arch_kwargs`` /
        ``arch_kwargs_req_import`` keys at the top level). The bare form is
        what the tracer tests use; the full form is what the real nnU-Net
        planner writes.
    dataset_json : dict
        The nnU-Net dataset.json. ``channel_names`` gives the input channel
        count (``len(channel_names)``) and ``numClasses`` gives the output
        channel count (background + foreground).
    configuration : str, optional
        The nnU-Net configuration name to read from ``plans['configurations']``
        when ``plans`` is a full plans dict. Defaults to ``"2d"`` (the
        Phase-14-settled configuration -- 3D pretraining is deferred per the
        locked decisions).
    device : torch.device | str, optional
        Where to place the network. Defaults to CPU (the tracer runs on
        CPU; the real run passes a CUDA device).
    deep_supervision : bool, optional
        Whether the network produces deep-supervision outputs (a list of
        tensors at multiple resolutions). The pretraining reconstruction
        target is a single full-resolution image, so deep supervision is
        off by default.
    allow_init : bool, optional
        Whether to apply the network's ``initialize`` weights init. Defaults
        to ``True`` (matches nnU-Net's default).

    Returns
    -------
    nn.Module
        The nnU-Net 2D ResEnc network on ``device``.

    Raises
    ------
    ValueError
        If ``plans`` is not a recognizable plans dict (neither a full
        nnU-Net plans dict with the requested configuration nor a bare
        architecture block), or ``dataset_json`` is missing
        ``channel_names`` / ``numClasses``.
    """
    # nnunetv2 is in the [ai] extra -- import function-scope so the module
    # loads with only torch installed.
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

    # Resolve the architecture block: a full nnU-Net plans dict stores it
    # under configurations[configuration]["architecture"]; the tracer tests
    # pass a bare architecture block directly.
    if "arch_class_name" in plans and "arch_kwargs" in plans:
        arch = plans
    elif "configurations" in plans and configuration in plans["configurations"]:
        arch = plans["configurations"][configuration]["architecture"]
    elif "network_class_name" in plans and "arch_kwargs" in plans:
        # A real nnU-Net v2 architecture block passed bare (network_class_name
        # is the nnU-Net v2 plans.json key; the tracer fixture uses
        # arch_class_name -- handle both).
        arch = plans
    else:
        raise ValueError(
            f"plans must be either a full nnU-Net plans dict with a "
            f"{configuration!r} configuration or a bare architecture block "
            f"with arch_class_name/arch_kwargs (tracer) or "
            f"network_class_name/arch_kwargs (real nnU-Net v2 plans) keys; "
            f"got keys {sorted(plans.keys())}"
        )

    # nnU-Net v2's real plans.json stores the network class name under
    # ``network_class_name`` and the import-required kwargs under
    # ``_kw_requires_import``; the tracer fixture uses ``arch_class_name`` /
    # ``arch_kwargs_req_import`` (the literal arg names get_network_from_plans
    # takes). Accept both so the same builder serves the tracer and the real
    # plans.json (the D-01a key-match guarantee holds for either form -- the
    # state_dict keys are architecture-determined, not key-name-determined).
    arch_class_name = arch.get("arch_class_name") or arch.get("network_class_name")
    if arch_class_name is None:
        raise ValueError(
            f"architecture block must carry the network class name under "
            f"'arch_class_name' (tracer) or 'network_class_name' (real nnU-Net "
            f"v2 plans); got keys {sorted(arch.keys())}"
        )
    arch_kwargs_req_import = arch.get("arch_kwargs_req_import")
    if arch_kwargs_req_import is None:
        arch_kwargs_req_import = arch.get("_kw_requires_import", [])

    channel_names = dataset_json.get("channel_names")
    if channel_names is None:
        raise ValueError(
            f"dataset_json must contain 'channel_names' (input channel spec), "
            f"got keys {sorted(dataset_json.keys())}"
        )
    input_channels = len(channel_names)
    # numClasses: the network's output channel count. For the masked-inpainting
    # reconstruction objective the output channels MUST equal the input
    # channels (the network reconstructs the masked image, not a segmentation
    # map) -- the pretraining loop enforces this and raises on a mismatch.
    # ``output_channels`` lets a caller override the derived value; otherwise
    # derive from ``numClasses`` (nnU-Net v1 plans) or from ``labels`` (nnU-Net
    # v2 dataset.json, which has no numClasses key -- the count is
    # len(labels), one output channel per label class). For pretraining the
    # caller passes output_channels == input_channels so the reconstruction
    # objective is well-formed.
    num_classes = dataset_json.get("numClasses")
    if num_classes is None:
        labels = dataset_json.get("labels")
        if labels is None:
            raise ValueError(
                f"dataset_json must contain 'numClasses' or 'labels' (output "
                f"channel count), got keys {sorted(dataset_json.keys())}"
            )
        num_classes = len(labels)
    # An explicit output_channels override wins (the masked-inpainting
    # reconstruction objective needs output_channels == input_channels, not
    # the segmentation class count -- the pretraining loop enforces this).
    if output_channels is not None:
        num_classes = output_channels

    network = get_network_from_plans(
        arch_class_name=arch_class_name,
        arch_kwargs=arch["arch_kwargs"],
        arch_kwargs_req_import=arch_kwargs_req_import,
        input_channels=input_channels,
        output_channels=int(num_classes),
        allow_init=allow_init,
        deep_supervision=deep_supervision,
    )
    if isinstance(device, str):
        device = torch.device(device)
    return network.to(device)


def _default_block_mask(
    batch: torch.Tensor,
    *,
    holes: int = 4,
    max_holes: int = 8,
    spatial_size: tuple[int, int] = (8, 8),
    max_spatial_size: tuple[int, int] | None = (12, 12),
    fill_value: float = -1.0e6,
    prob: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Default 2D block mask using MONAI ``RandCoarseDropoutd`` (channel-preserving).

    ``spatial_size=(-1, H, W)`` preserves the channel dim so the dropout
    regions are 2D blocks on HxW applied identically across channels (the
    CNN-MAE-appropriate block masking variant -- contiguous blocks large
    enough that the interior is genuinely masked from the conv boundary).

    A sentinel ``fill_value`` (outside the z-scored data range) is used to
    derive the boolean mask reliably (a 0.0 fill would be ambiguous on z-
    scored data where 0.0 is the channel mean); the masked input is then
    zero-filled for the network input.

    Parameters
    ----------
    batch : torch.Tensor
        The input batch ``(B, C, H, W)``.
    holes, max_holes : int
        Number of dropout regions (MONAI samples uniformly in
        ``[holes, max_holes]``).
    spatial_size, max_spatial_size : tuple[int, int]
        The (H, W) block size range. The channel dim is preserved (the
        ``-1`` is prepended internally).
    fill_value : float
        The sentinel used to derive the mask; replaced with 0.0 in the
        returned masked input.
    prob : float
        Per-call probability of applying the dropout (the tracer uses 1.0
        via the loop wrapper so every batch is masked).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(masked_input, mask)``: ``masked_input`` is the batch with
        dropout regions zero-filled (the network input); ``mask`` is a
        boolean tensor (True = masked region) the loop uses to compute the
        reconstruction MSE.

    Raises
    ------
    ValueError
        If ``batch`` is not a 4D ``(B, C, H, W)`` tensor.
    """
    # MONAI is in the [benchmark] extra -- import function-scope so the
    # module loads with only torch installed.
    from monai.transforms import RandCoarseDropoutd

    if batch.ndim != 4:
        raise ValueError(
            f"_default_block_mask expects a (B, C, H, W) 4D tensor, got ndim={batch.ndim}"
        )
    # spatial_size=(-1, H, W): -1 = don't dropout the channel dim -> 2D block
    # masking on HxW applied across all channels.
    mask_transform = RandCoarseDropoutd(
        keys=["image"],
        holes=holes,
        spatial_size=(-1, *spatial_size),
        max_holes=max_holes,
        max_spatial_size=(-1, *max_spatial_size) if max_spatial_size else None,
        fill_value=fill_value,
        prob=prob,
    )
    out = mask_transform({"image": batch})
    sentinel_filled = out["image"]
    # Derive the boolean mask from the sentinel, then zero-fill the masked
    # regions for the network input (the network sees zeros in the holes,
    # not the sentinel).
    mask = sentinel_filled == fill_value
    masked_input = torch.where(mask, torch.zeros_like(batch), batch)
    return masked_input, mask


def _save_checkpoint_atomic(state: dict[str, Any], output_path: str) -> None:
    """Save a checkpoint dict atomically via a temp-file + ``Path.replace``.

    Writes to a unique temp file in the destination directory, then
    ``Path.replace`` renames it into place (atomic on POSIX). On any
    failure the temp file is unlinked and the error re-raised (cleanup-
    then-reraise, never swallow -- AGENTS section 2; the sanctioned
    ``except BaseException`` broad catch).
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(out.parent), suffix=".pth.tmp", prefix=".pretrain_")
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(state, f)
        Path(tmp).replace(out)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def masked_inpainting_pretrain(
    network: nn.Module,
    dataset: Sequence[torch.Tensor] | None = None,
    *,
    epochs: int,
    output_path: str,
    device: torch.device | str = "cpu",
    mask_transform: MaskTransform | None = None,
    learning_rate: float = 1e-3,
    use_amp: bool = False,
    max_grad_norm: float = 1.0,
    ddp: bool = False,
    batch_sampler: Callable[[], torch.Tensor] | None = None,
    steps_per_epoch: int | None = None,
) -> list[float]:
    """Run the masked-inpainting pretraining loop and save the checkpoint.

    For each epoch, for each batch: apply the mask transform (default: MONAI
    ``RandCoarseDropoutd`` 2D block masking), forward the masked input,
    compute MSE on the masked regions vs the original unmasked image (the
    D-02 masked-inpainting objective), and step the optimizer with AMP +
    grad-clip scaler ordering. After all epochs the network weights are saved
    as ``{'network_weights': network.state_dict()}`` -- the exact format
    ``load_pretrained_weights`` expects.

    Two batch-supply modes:

    * **Pre-built sequence** (``dataset``): a sequence of ``(B, C, H, W)``
      tensors iterated per epoch. The tracer tests use this. Under DDP the
      sequence is sharded across ranks (rank r takes every ``world_size``-th
      batch starting at r).
    * **On-demand sampler** (``batch_sampler`` + ``steps_per_epoch``): a
      callable that returns a fresh ``(B, C, H, W)`` tensor each call. The
      real run uses this so CPU batch-prep (corpus patch sampling + Frangi
      mask) interleaves with GPU forward/backward instead of pre-building
      all batches upfront (which would be ~40 min of CPU work before the
      first GPU step). Under DDP each rank calls the sampler independently
      -- the gradients are synchronized by the DDP all-reduce, so the
      effective batch throughput scales with world_size.

    Parameters
    ----------
    network : nn.Module
        The nnU-Net 2D ResEnc network (built via :func:`build_pretrain_network`).
    dataset : Sequence[torch.Tensor] | None
        A sequence of batched input tensors ``(B, C, H, W)`` (the pre-built
        mode). Ignored when ``batch_sampler`` is provided. An empty sequence
        raises ``ValueError`` (no zero-fill / no silent pass). Under DDP the
        dataset is sharded across ranks (rank r takes every ``world_size``-th
        batch starting at r) so each rank sees a disjoint subset and the
        effective batch throughput scales with world_size.
    epochs : int
        Number of epochs to run.
    output_path : str
        Where to save the checkpoint (``{'network_weights': state_dict}``).
        Under DDP only rank 0 writes the file (the other ranks' state_dicts
        are identical post-all-reduce).
    device : torch.device | str, optional
        The device to run on. Defaults to CPU.
    mask_transform : MaskTransform | None, optional
        A callable ``(batch) -> (masked_input, mask)``. Defaults to
        :func:`_default_block_mask` (MONAI ``RandCoarseDropoutd``). Plugging
        in a vessel-aware (Frangi/Hessian-biased) transform is how a later
        plan biases hole placement without editing this loop.
    learning_rate : float, optional
        The optimizer learning rate. Defaults to ``1e-3``.
    use_amp : bool, optional
        Whether to use AMP mixed precision. AMP is no-op on CPU (the scaler
        disables itself when CUDA is unavailable) so the same code path
        serves the CPU tracer and the CUDA real run.
    max_grad_norm : float, optional
        Gradient clipping max norm. Defaults to ``1.0``.
    ddp : bool, optional
        Whether to wrap the network in ``DistributedDataParallel`` and shard
        the dataset across ranks. Requires ``torch.distributed`` to be
        initialized (the CLI calls ``init_process_group`` under torchrun).
        Defaults to ``False`` (single-process).

    Returns
    -------
    list[float]
        The per-epoch mean reconstruction loss (finite -- no NaN/inf).

    Raises
    ------
    ValueError
        If ``dataset`` is empty (the loop cannot train on an empty corpus),
        ``epochs`` is not positive, or a batch is not a 4D tensor.
    RuntimeError
        If ``ddp=True`` but ``torch.distributed`` is not initialized (run
        under torchrun so the CLI can call ``init_process_group``).
    """
    if batch_sampler is None and not dataset:
        raise ValueError(
            "masked_inpainting_pretrain: dataset is empty and no batch_sampler "
            "was provided -- cannot pretrain on an empty corpus. Supply either "
            "a non-empty dataset or a batch_sampler + steps_per_epoch."
        )
    if batch_sampler is not None and steps_per_epoch is None:
        raise ValueError(
            "masked_inpainting_pretrain: batch_sampler requires steps_per_epoch "
            "(the number of sampler calls per epoch)"
        )
    if steps_per_epoch is not None and steps_per_epoch < 1:
        raise ValueError(
            f"masked_inpainting_pretrain: steps_per_epoch must be >= 1, "
            f"got steps_per_epoch={steps_per_epoch}"
        )
    if epochs < 1:
        raise ValueError(f"masked_inpainting_pretrain: epochs must be >= 1, got epochs={epochs}")
    if isinstance(device, str):
        device = torch.device(device)
    if mask_transform is None:
        mask_transform = _default_block_mask

    # DDP setup: when ddp=True, torch.distributed must already be
    # initialized (the CLI calls init_process_group under torchrun). Wrap the
    # network in DistributedDataParallel so gradients are all-reduced across
    # ranks, and shard the dataset so each rank processes a disjoint subset
    # (effective throughput scales with world_size). Rank 0 writes the
    # checkpoint; the other ranks' state_dicts are identical post-all-reduce.
    rank = 0
    world_size = 1
    if ddp:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel

        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "masked_inpainting_pretrain: ddp=True requires torch.distributed "
                "to be initialized -- run under torchrun (the CLI calls "
                "init_process_group when --ddp is passed)"
            )
        world_size = dist.get_world_size()
        rank = dist.get_rank()

    network = network.to(device)
    network.train()
    if ddp:
        network = DistributedDataParallel(
            network,
            device_ids=[int(os.environ["LOCAL_RANK"])] if torch.cuda.is_available() else None,
        )
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    # AMP GradScaler -- no-ops on CPU (enabled=use_amp and
    # torch.cuda.is_available()) so the same code path serves the CPU tracer
    # and the CUDA real run. When use_amp=False, scaler.scale(loss).backward()
    # == loss.backward(), scaler.unscale_ is a no-op, scaler.step ==
    # optimizer.step, and scaler.update is a no-op.
    scaler = torch.amp.GradScaler(enabled=use_amp and torch.cuda.is_available())

    epoch_losses: list[float] = []
    for _epoch in range(epochs):
        loss_sum = 0.0
        loss_count = 0
        if batch_sampler is not None:
            # On-demand sampler mode: each rank calls the sampler
            # steps_per_epoch times per epoch. CPU batch-prep (corpus patch
            # sampling + Frangi mask) interleaves with GPU forward/backward
            # instead of pre-building all batches upfront. The DDP
            # all-reduce synchronizes gradients across ranks each step.
            batch_iter = (batch_sampler() for _ in range(steps_per_epoch))
        else:
            # Pre-built sequence mode: shard the dataset across ranks (rank r
            # takes every world_size-th batch starting at r). Each rank sees
            # a disjoint subset; effective throughput scales with world_size.
            batch_iter = (dataset[batch_idx] for batch_idx in range(rank, len(dataset), world_size))
        for batch in batch_iter:
            if not isinstance(batch, torch.Tensor) or batch.ndim != 4:
                raise ValueError(
                    f"masked_inpainting_pretrain: each batch must be a 4D "
                    f"(B, C, H, W) tensor, got {type(batch).__name__} with "
                    f"ndim={getattr(batch, 'ndim', None)}"
                )
            batch = batch.to(device)
            original = batch  # the reconstruction target is the unmasked image

            masked_input, mask = mask_transform(batch)
            masked_input = masked_input.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            output = network(masked_input)
            # The network may return a list (deep supervision) or a single
            # tensor; the reconstruction target is the full-resolution
            # image, so take the first / only output.
            if isinstance(output, (list, tuple)):
                output = output[0]
            # MSE on the masked regions only (the D-02 masked-inpainting
            # objective). The mask is (B, C, H, W) bool; broadcast across
            # channels if the network output channel count differs from the
            # input (e.g. a segmentation head with num_classes channels).
            # For the reconstruction objective the network output channels
            # must match the input channels; if they do not, we cannot
            # compute a per-pixel reconstruction MSE -- raise explicitly
            # rather than silently producing a wrong-shaped loss.
            if output.shape[1] != original.shape[1]:
                raise ValueError(
                    f"masked_inpainting_pretrain: network output channels "
                    f"({output.shape[1]}) must match input channels "
                    f"({original.shape[1]}) for the reconstruction objective; "
                    f"build the pretraining network with output_channels == "
                    f"input_channels (numClasses == len(channel_names))"
                )
            # Flatten the masked elements and compute MSE over them. Using
            # torch.masked_select preserves the per-element squared error
            # mean over the masked set (not a per-patch mean).
            masked_pred = torch.masked_select(output, mask)
            masked_target = torch.masked_select(original, mask)
            if masked_pred.numel() == 0:
                # A batch where the mask transform dropped nothing this
                # call (prob < 1) -- skip rather than producing a NaN from
                # a zero-divisor MSE. This is NOT the empty-corpus case
                # (that raises above); it is a single empty-mask batch.
                continue
            loss = nn.functional.mse_loss(masked_pred, masked_target)

            # AMP+grad-clip scaler ordering: scale -> backward -> unscale ->
            # clip_grad_norm_ -> step -> update. unscale_ may only be called
            # once per optimizer per step (RuntimeError otherwise). When
            # use_amp=False the scaler no-ops so this is equivalent to
            # loss.backward() + optimizer.step() with grad clipping.
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            loss_sum += float(loss.item())
            loss_count += 1

        if loss_count == 0:
            raise ValueError(
                "masked_inpainting_pretrain: no batches produced a loss this "
                "epoch -- every batch's mask transform dropped zero elements "
                "(check mask_transform prob / block size vs the input size)."
            )
        epoch_losses.append(loss_sum / loss_count)

    # Save the checkpoint as {'network_weights': state_dict} -- the exact
    # format load_pretrained_weights expects. Atomic temp-file write so a
    # crash mid-write does not leave a partial checkpoint. Under DDP only
    # rank 0 writes (the other ranks' state_dicts are identical post-all-
    # reduce); a barrier ensures all ranks finished the last step before the
    # save, so rank 0 does not exit and tear down the process group first.
    if ddp:
        import torch.distributed as dist

        dist.barrier()
    if rank == 0:
        # Under DDP, unwrap the module to get the underlying network's
        # state_dict (DDP prefixes keys with "module.").
        sd_module = network.module if hasattr(network, "module") else network
        _save_checkpoint_atomic({"network_weights": sd_module.state_dict()}, output_path)
    return epoch_losses
