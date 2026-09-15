"""Contender Protocol + architecture implementations for the vseg benchmark.

All four contenders (the legacy 2D U-Net + three wired architectures) share
the same ``Contender`` Protocol so the benchmark harness can train, predict,
and score them identically through the ship-gate eval-metric matrix.

The legacy ``Improved2DContender`` wraps :class:`liom_toolkit.segmentation.vseg.model.VsegModel`
and reuses the existing ``train_model`` orchestration + ``predict_one``
inference path — it is the measured baseline the other architectures are
compared against.

The three remaining contenders:

* ``MonaiUnetContender`` — MONAI :class:`~monai.networks.nets.UNet` with a
  residual encoder (``num_res_units=2``, ``strides=(2, 2, 2)`` — 3 strided
  stages, not the legacy 4) + composite loss (DiceFocal + soft-clDice) +
  :class:`~monai.inferers.SlidingWindowInferer` (Gaussian overlap blending).
  The model outputs LOGITS (no sigmoid in forward — MONAI losses apply
  sigmoid internally via ``sigmoid=True``).
* ``SwinUnetContender`` — MONAI :class:`~monai.networks.nets.SwinUNETR`
  (``spatial_dims=2``, ``use_checkpoint=True``; no ``img_size=`` — removed in
  MONAI 1.5) + the same composite loss + SlidingWindowInferer path.
* ``NnUnetContender`` — nnU-Net v2 run fully in-process. ``nnunetv2`` is a
  member of the ``[ai]`` extra, so the contender drives nnU-Net's own
  Python API in the same environment: dataset conversion via
  :func:`~liom_toolkit.scripts.liom_prepare_nnunet_dataset.prepare_nnunet_2d`,
  fingerprint/plan/preprocess via ``nnunetv2.experiment_planning``, training
  via ``nnunetv2.run.run_training.run_training`` (``num_gpus > 1`` engages
  nnU-Net's built-in DDP), and prediction via
  :class:`~liom_toolkit.segmentation.vseg.model_v2.NnUnetV2Model`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

# torch is in the [ai] extra. The upfront ImportError is the honest signal
# on an io-only install — the message names [ai,benchmark] (the torch +
# MONAI path the benchmark contenders need). The `from e` chain preserves
# the underlying error for debugging (AGENTS §2). MONAI is imported inside
# the contender methods (function-scope) so this module imports cleanly
# with only torch installed; MONAI is a separate [benchmark] dep.
try:
    import torch
except ImportError as e:
    raise ImportError(
        "Please install liom-toolkit[ai,benchmark] to use the benchmark contenders."
    ) from e

if TYPE_CHECKING:
    from liom_toolkit.segmentation.vseg.model_v2 import NnUnetV2Model

__all__ = [
    "Contender",
    "Improved2DContender",
    "MonaiUnetContender",
    "NnUnetContender",
    "SwinUnetContender",
]


@runtime_checkable
class Contender(Protocol):
    """A benchmark architecture contender.

    All contenders share the same per-volume split and are scored by the
    same ship-gate eval-metric matrix. The harness calls
    :meth:`train_and_predict` to obtain predicted masks, then hands them to
    the eval-metric functions.
    """

    name: str

    def train_and_predict(
        self,
        train_slices: list[str],
        test_slices: list[str],
        output_dir: str,
        patch_size: tuple[int, int, int] = (1, 256, 256),
        ddp: bool = False,
    ) -> list[NDArray[np.bool_]]:
        """Train on ``train_slices``, predict on ``test_slices``, return binary masks.

        The returned masks are boolean (``NDArray[np.bool_]``), one per test
        slice in ``test_slices`` order, so the eval-metric matrix receives
        the dtype its signatures require.

        Returns
        -------
        list[NDArray[np.bool_]]
            One boolean mask per test slice, in ``test_slices`` order.
        """
        ...

    def predict_on_slices(
        self,
        slices: list[str],
        checkpoint_path: str,
    ) -> list[NDArray[np.bool_]]:
        """Load ``checkpoint_path`` and predict binary masks for ``slices``.

        Returns
        -------
        list[NDArray[np.bool_]]
            One boolean mask per slice, in ``slices`` order.
        """
        ...


class Improved2DContender:
    """Legacy 2D U-Net contender — wraps :class:`VsegModel` (the measured baseline).

    Reuses the existing ``train_model`` orchestration (DDP entry, AMP,
    checkpointing) and the ``predict_one`` inference path. The prediction
    path uses ``filter_empty=False`` semantics (predict on ALL test slices,
    including vessel-free ones) — required for the FPR-on-empty metric.
    """

    name: str = "improved_2d"

    def __init__(self, device: str = "cpu") -> None:
        """Initialise the contender.

        Parameters
        ----------
        device : str
            The torch device string for training + inference (``"cpu"`` or
            ``"cuda"``).
        """
        self.device = device

    def train_and_predict(
        self,
        train_slices: list[str],
        test_slices: list[str],
        output_dir: str,
        patch_size: tuple[int, int, int] = (1, 256, 256),
        ddp: bool = False,
    ) -> list[NDArray[np.bool_]]:
        """Train the legacy VsegModel and predict boolean masks on test slices.

        The first ``train_slices`` entry is the zarr dataset path the
        benchmark caller packs the train slices into; ``train_model`` trains
        on it and writes a checkpoint to ``output_dir``. The checkpoint is
        loaded into a fresh ``VsegModel`` and ``predict_one`` is called per
        test slice. ``predict_one`` returns uint8 0/255 output; it is
        binarized to bool so the eval-metric matrix receives
        ``NDArray[np.bool_]`` (a 0/255 uint8 array passed to ``cl_score``
        would scale tprec/tsens by 255 — silent wrong data).

        Returns
        -------
        list[NDArray[np.bool_]]
            One boolean mask per test slice, in ``test_slices`` order.

        Raises
        ------
        RuntimeError
            If the checkpoint is not found at
            ``output_dir/files/checkpoint.latest.pth`` after ``train_model``
            (training likely failed — no silent proceed with an untrained
            model).
        """
        from liom_toolkit.segmentation.vseg.model import VsegModel
        from liom_toolkit.segmentation.vseg.prediction import predict_one
        from liom_toolkit.segmentation.vseg.training import train_model

        dev = torch.device(self.device)
        dataset_file = train_slices[0] if train_slices else ""
        train_model(
            dataset_file=dataset_file,
            node_name="channel_0",
            dev=dev,
            output_train=output_dir,
            wandb_mode="disabled",
            epochs=50,
            batch_size=4,
            patch_size=patch_size,
            ddp=ddp,
            normalisation_value=255,
            num_workers=0,
            iterations_per_epoch=250,
        )

        model = VsegModel(pretrained=False, device=dev)
        checkpoint_path = Path(output_dir) / "files" / "checkpoint.latest.pth"
        if not checkpoint_path.exists():
            # Training failed silently, disk full, or wrong path — raise
            # rather than proceed with a randomly-initialized VsegModel and
            # present random predictions as trained-model output (the
            # silent-wrong-data path AGENTS §2 forbids).
            raise RuntimeError(
                f"Improved2DContender: checkpoint not found after train_model: "
                f"{checkpoint_path} — training likely failed"
            )
        state = torch.load(str(checkpoint_path), map_location=dev, weights_only=True)
        model.load_state_dict(state)
        model.to(dev)
        model.eval()

        masks: list[NDArray[np.bool_]] = []
        for slice_path in test_slices:
            pred = predict_one(
                model, slice_path, save_path=output_dir, dev=self.device, patching=False
            )
            masks.append(np.asarray(pred).astype(bool))
        return masks

    def predict_on_slices(
        self,
        slices: list[str],
        checkpoint_path: str,
    ) -> list[NDArray[np.bool_]]:
        """Load ``checkpoint_path`` into a VsegModel and predict boolean masks.

        Returns
        -------
        list[NDArray[np.bool_]]
            One boolean mask per slice, in ``slices`` order.
        """
        from liom_toolkit.segmentation.vseg.model import VsegModel
        from liom_toolkit.segmentation.vseg.prediction import predict_one

        dev = torch.device(self.device)
        model = VsegModel(pretrained=False, device=dev)
        state = torch.load(checkpoint_path, map_location=dev, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        save_path = str(Path(checkpoint_path).parent)
        masks: list[NDArray[np.bool_]] = []
        for slice_path in slices:
            pred = predict_one(
                model, slice_path, save_path=save_path, dev=self.device, patching=False
            )
            masks.append(np.asarray(pred).astype(bool))
        return masks


def _read_slice_as_tensor(slice_path: str, device: torch.device) -> torch.Tensor:
    """Read a 2D image slice as a (1, 1, H, W) float32 tensor on ``device``.

    The MONAI contenders feed (B, C, H, W) tensors to the model + inferer.
    The slice is read via imageio.v3, converted to float32 in [0, 1], and
    unsqueezed to add the batch + channel dims.

    Parameters
    ----------
    slice_path : str
        Path to the 2D image slice (any imageio-readable format).
    device : torch.device
        The device to place the tensor on.

    Returns
    -------
    torch.Tensor
        A (1, 1, H, W) float32 tensor on ``device``.
    """
    import imageio.v3 as iio

    img = iio.imread(slice_path)
    arr = np.asarray(img, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    # (H, W) → (1, 1, H, W) — batch=1, channel=1.
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)


def _monai_predict_slices(
    model: torch.nn.Module,
    test_slices: list[str],
    device: torch.device,
    roi_size: tuple[int, int] = (256, 256),
) -> list[NDArray[np.bool_]]:
    """Run SlidingWindowInferer + post-processing on each test slice.

    Uses :class:`~monai.inferers.SlidingWindowInferer` with Gaussian overlap
    blending (replaces the legacy manual patch pasting — no overlap severed
    capillaries). Post-processing applies sigmoid (logits → probabilities)
    then a 0.5 threshold (probabilities → binary). The model outputs LOGITS
    (no sigmoid in forward — MONAI losses apply sigmoid internally; here
    sigmoid is applied by ``Activations(sigmoid=True)`` post-processing).

    Parameters
    ----------
    model : torch.nn.Module
        The MONAI model (UNet or SwinUNETR) in eval mode.
    test_slices : list[str]
        Paths to the 2D test slices to predict on.
    device : torch.device
        The device to run inference on.
    roi_size : tuple[int, int]
        The sliding-window ROI size (matches the training patch size).

    Returns
    -------
    list[NDArray[np.bool_]]
        One boolean mask per test slice, in ``test_slices`` order.
    """
    from monai.inferers import SlidingWindowInferer
    from monai.transforms import Activations, AsDiscrete, Compose

    inferer = SlidingWindowInferer(
        roi_size=roi_size,
        sw_batch_size=4,
        overlap=0.25,
        mode="gaussian",
    )
    post = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    model.eval()
    masks: list[NDArray[np.bool_]] = []
    with torch.no_grad():
        for slice_path in test_slices:
            tensor = _read_slice_as_tensor(slice_path, device)
            logits = inferer(tensor, model)
            pred = post(logits)
            # (1, 1, H, W) → (H, W) bool.
            masks.append(np.asarray(pred.squeeze().cpu().numpy(), dtype=bool))
    return masks


def _monai_composite_loss(lambda_cldice: float = 0.5) -> torch.nn.Module:
    """Build the composite DiceFocal + soft-clDice loss.

    Both components expect LOGITS (``sigmoid=True`` applies sigmoid
    internally — the model forward must NOT also apply sigmoid, or the loss
    computes sigmoid(sigmoid(logits)) → wrong gradients, the silent-wrong-data
    failure mode). soft-clDice is NEVER used alone — it is combined with
    DiceFocal per the architecture decision.

    Parameters
    ----------
    lambda_cldice : float
        The soft-clDice weight (``loss = DiceFocal + lambda_cldice * soft_clDice``).

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


class MonaiUnetContender:
    """MONAI UNet contender — residual encoder, ≤3 strided stages.

    Builds :class:`~monai.networks.nets.UNet` with ``num_res_units=2`` and
    ``strides=(2, 2, 2)`` (3 strided stages — the legacy 4-stage MaxPool2d
    destroys sub-2-voxel capillaries). The model outputs LOGITS (no sigmoid
    in forward). Training delegates to the DDP entry; inference uses
    :class:`~monai.inferers.SlidingWindowInferer` with Gaussian overlap
    blending. Composite loss: DiceFocal + soft-clDice (soft-clDice never
    alone).
    """

    name: str = "monai_unet"

    def __init__(
        self,
        device: str = "cpu",
        *,
        epochs: int = 50,
        batch_size: int = 4,
    ) -> None:
        """Initialise the contender.

        Parameters
        ----------
        device : str
            The torch device string for training + inference.
        epochs : int
            Number of training epochs (forwarded to ``train_monai_model``).
            Default 50 for production; tests use 1.
        batch_size : int
            Training batch size (forwarded to ``train_monai_model``).
        """
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size

    def _build_model(self) -> torch.nn.Module:
        """Build the MONAI UNet (residual encoder, ≤3 strided stages).

        Returns
        -------
        torch.nn.Module
            The MONAI UNet (residual encoder, 3 strided stages, logits output).
        """
        from monai.networks.nets import UNet

        # UNet has NO attention= kwarg (use AttentionUnet for attention gates).
        # num_res_units=2 adds residual units (the "improved" part);
        # strides=(2,2,2) = 3 strided stages (≤3 per the architecture decision).
        return UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(32, 64, 128, 256),
            strides=(2, 2, 2),
            num_res_units=2,
            act="prelu",
            norm="instance",
        )

    def train_and_predict(
        self,
        train_slices: list[str],
        test_slices: list[str],
        output_dir: str,
        patch_size: tuple[int, int, int] = (1, 256, 256),
        ddp: bool = False,
    ) -> list[NDArray[np.bool_]]:
        """Train the MONAI UNet on PNG slices, then predict boolean masks.

        Delegates training to
        :func:`~liom_toolkit.segmentation.vseg.benchmark.monai_training.train_monai_model`,
        which builds a PNG-slice dataset, trains with the composite
        DiceFocal + soft-clDice loss, AdamW, cosine schedule, AMP + grad-clip,
        and saves ``checkpoint.latest.pth``. The trained ``state_dict`` is
        loaded into a fresh MONAI UNet before
        :func:`_monai_predict_slices` runs SlidingWindowInferer + sigmoid +
        0.5 threshold → bool masks.

        Returns
        -------
        list[NDArray[np.bool_]]
            One boolean mask per test slice, in ``test_slices`` order.

        Raises
        ------
        RuntimeError
            If the checkpoint is not found after training (training likely
            failed — no silent proceed with a randomly-initialized model).
        """
        from liom_toolkit.segmentation.vseg.benchmark.monai_training import (
            train_monai_model,
        )

        dev = torch.device(self.device)
        model = self._build_model()
        ckpt_path = train_monai_model(
            model=model,
            train_slices=train_slices,
            output_dir=output_dir,
            patch_size=patch_size,
            epochs=self.epochs,
            batch_size=self.batch_size,
            ddp=ddp,
            device=dev,
        )
        if not ckpt_path.exists():
            raise RuntimeError(
                f"MonaiUnetContender: checkpoint not found after training: "
                f"{ckpt_path} — training likely failed"
            )
        # Load the trained weights into a fresh model for prediction
        # (the training model may be DDP-wrapped or on a different device).
        pred_model = self._build_model().to(dev)
        state = torch.load(str(ckpt_path), map_location=dev, weights_only=True)
        pred_model.load_state_dict(state)
        return _monai_predict_slices(pred_model, test_slices, dev)

    def predict_on_slices(
        self,
        slices: list[str],
        checkpoint_path: str,
    ) -> list[NDArray[np.bool_]]:
        """Load ``checkpoint_path`` into a MONAI UNet and predict boolean masks.

        Returns
        -------
        list[NDArray[np.bool_]]
            One boolean mask per slice, in ``slices`` order.
        """
        dev = torch.device(self.device)
        model = self._build_model().to(dev)
        state = torch.load(checkpoint_path, map_location=dev, weights_only=True)
        model.load_state_dict(state)
        return _monai_predict_slices(model, slices, dev)


class SwinUnetContender:
    """MONAI SwinUNETR contender — Swin Transformer U-Net (2D).

    Builds :class:`~monai.networks.nets.SwinUNETR` with ``spatial_dims=2``
    and ``use_checkpoint=True`` (gradient checkpointing for VRAM savings).
    No ``img_size=`` kwarg — it was deprecated in MONAI 1.3 and removed in
    1.5 (passing it raises ``TypeError``). Input spatial dims must be
    divisible by 32. Same DDP entry + composite loss + SlidingWindowInferer
    path as the UNet contender.
    """

    name: str = "monai_swinunetr"

    def __init__(
        self,
        device: str = "cpu",
        *,
        epochs: int = 50,
        batch_size: int = 4,
    ) -> None:
        """Initialise the contender.

        Parameters
        ----------
        device : str
            The torch device string for training + inference.
        epochs : int
            Number of training epochs (forwarded to ``train_monai_model``).
            Default 50 for production; tests use 1.
        batch_size : int
            Training batch size (forwarded to ``train_monai_model``).
        """
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size

    def _build_model(self) -> torch.nn.Module:
        """Build the MONAI SwinUNETR (2D, gradient checkpointing, no img_size=).

        Returns
        -------
        torch.nn.Module
            The MONAI SwinUNETR (2D, gradient checkpointing, logits output).
        """
        from monai.networks.nets import SwinUNETR

        # NO img_size= kwarg — removed in MONAI 1.5 (passing it raises
        # TypeError). Input size validation happens during forward(); spatial
        # dims must be divisible by 32.
        return SwinUNETR(
            in_channels=1,
            out_channels=1,
            feature_size=48,
            spatial_dims=2,
            use_checkpoint=True,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            norm_name="instance",
        )

    def train_and_predict(
        self,
        train_slices: list[str],
        test_slices: list[str],
        output_dir: str,
        patch_size: tuple[int, int, int] = (1, 256, 256),
        ddp: bool = False,
    ) -> list[NDArray[np.bool_]]:
        """Train the MONAI SwinUNETR on PNG slices, then predict boolean masks.

        Delegates training to
        :func:`~liom_toolkit.segmentation.vseg.benchmark.monai_training.train_monai_model`,
        which builds a PNG-slice dataset, trains with the composite
        DiceFocal + soft-clDice loss, AdamW, cosine schedule, AMP + grad-clip,
        and saves ``checkpoint.latest.pth``. The trained ``state_dict`` is
        loaded into a fresh SwinUNETR before
        :func:`_monai_predict_slices` runs SlidingWindowInferer + sigmoid +
        0.5 threshold → bool masks.

        Returns
        -------
        list[NDArray[np.bool_]]
            One boolean mask per test slice, in ``test_slices`` order.

        Raises
        ------
        RuntimeError
            If the checkpoint is not found after training (training likely
            failed — no silent proceed with a randomly-initialized model).
        """
        from liom_toolkit.segmentation.vseg.benchmark.monai_training import (
            train_monai_model,
        )

        dev = torch.device(self.device)
        model = self._build_model()
        ckpt_path = train_monai_model(
            model=model,
            train_slices=train_slices,
            output_dir=output_dir,
            patch_size=patch_size,
            epochs=self.epochs,
            batch_size=self.batch_size,
            ddp=ddp,
            device=dev,
        )
        if not ckpt_path.exists():
            raise RuntimeError(
                f"SwinUnetContender: checkpoint not found after training: "
                f"{ckpt_path} — training likely failed"
            )
        pred_model = self._build_model().to(dev)
        state = torch.load(str(ckpt_path), map_location=dev, weights_only=True)
        pred_model.load_state_dict(state)
        return _monai_predict_slices(pred_model, test_slices, dev)

    def predict_on_slices(
        self,
        slices: list[str],
        checkpoint_path: str,
    ) -> list[NDArray[np.bool_]]:
        """Load ``checkpoint_path`` into a SwinUNETR and predict boolean masks.

        Returns
        -------
        list[NDArray[np.bool_]]
            One boolean mask per slice, in ``slices`` order.
        """
        dev = torch.device(self.device)
        model = self._build_model().to(dev)
        state = torch.load(checkpoint_path, map_location=dev, weights_only=True)
        model.load_state_dict(state)
        return _monai_predict_slices(model, slices, dev)


class NnUnetContender:
    """nnU-Net v2 contender — the full lifecycle runs in-process.

    ``nnunetv2`` is a member of the ``[ai]`` extra, so the contender drives
    nnU-Net's own Python API in the same environment — no subprocess, no
    separate venv. ``train_and_predict`` chains:

    1. :func:`~liom_toolkit.segmentation.vseg.ssl.warmstart.validate_nnunet_env`
       — the ``nnUNet_raw`` / ``nnUNet_preprocessed`` / ``nnUNet_results``
       env contract, then the ``<stem>_mask.png`` label guard.
    2. :func:`~liom_toolkit.scripts.liom_prepare_nnunet_dataset.prepare_nnunet_2d`
       — converts the train slices + labels to nnU-Net raw format.
    3. ``extract_fingerprint_dataset`` → ``plan_experiment_dataset`` →
       ``preprocess_dataset`` — nnU-Net self-configures patch size, batch
       size, and architecture from the dataset statistics.
    4. ``run_training`` — nnU-Net's own training loop; ``num_gpus > 1``
       engages its built-in DDP.
    5. :class:`~liom_toolkit.segmentation.vseg.model_v2.NnUnetV2Model` —
       the shared ``nnUNetPredictor`` wrapper turns each test slice into a
       boolean mask (vessel-channel softmax ``> 0.5``).

    The trained-model directory derived from ``nnUNet_results`` (and the
    ``checkpoint_path`` accepted by :meth:`predict_on_slices`) must be a
    TRUSTED nnU-Net training output — upstream loads checkpoints with
    ``torch.load(weights_only=False)``.
    """

    name: str = "nnunet_v2"

    def __init__(
        self,
        device: str = "cpu",
        dataset_id: int = 999,
        num_gpus: int = 1,
        *,
        trainer_name: str = "nnUNetTrainer_50epochs",
        spacing: tuple[float, float] = (1.0, 1.0),
        num_processes: int = 8,
    ) -> None:
        """Initialise the contender.

        Parameters
        ----------
        device : str
            The torch device string for training + inference (``"cpu"`` or
            ``"cuda"``); forwarded to ``run_training`` and the predictor
            wrapper.
        dataset_id : int
            The nnU-Net dataset id. The dataset is written to
            ``nnUNet_raw/Dataset{id:03d}_LIOM6p5`` and the trained model is
            read back from the matching ``nnUNet_results`` subtree.
        num_gpus : int
            Number of GPUs for nnU-Net's built-in DDP training. Default 1.
        trainer_name : str
            The nnU-Net trainer class name passed to ``run_training``.
            Default ``"nnUNetTrainer_50epochs"`` — the installed 50-epoch
            variant, matching the 50-epoch budget of the MONAI contenders
            for a fair comparison.
        spacing : tuple[float, float]
            In-plane ``(row, col)`` spacing forwarded to the predictor's
            ``predict_proba`` (promoted internally to the 3-element spacing
            nnunetv2 requires; the through-plane entry is a pass-through
            for ``'2d'`` configurations). Benchmark PNGs are pixel-space —
            pass the real in-plane spacing if the dataset carries physical
            units.
        num_processes : int
            Worker processes for fingerprint extraction and preprocessing.
            Default 8 (nnU-Net's own default).
        """
        self.device = device
        self.dataset_id = dataset_id
        self.num_gpus = num_gpus
        self.trainer_name = trainer_name
        self.spacing = spacing
        self.num_processes = num_processes

    def train_and_predict(
        self,
        train_slices: list[str],
        test_slices: list[str],
        output_dir: str,
        patch_size: tuple[int, int, int] = (1, 256, 256),
        ddp: bool = False,
    ) -> list[NDArray[np.bool_]]:
        """Full nnU-Net pipeline in-process: prepare → plan → train → predict.

        Validates the ``nnUNet_*`` env vars first, then the per-slice
        ``<stem>_mask.png`` labels, then runs the library chain:
        ``prepare_nnunet_2d`` → ``extract_fingerprint_dataset`` →
        ``plan_experiment_dataset`` → ``preprocess_dataset`` →
        ``run_training`` → ``NnUnetV2Model`` predictions.

        ``patch_size`` and ``ddp`` are unused — nnU-Net self-configures its
        patch size from the dataset statistics and manages multi-GPU
        training internally via ``num_gpus``. Kept for Protocol structural
        conformance.

        Returns
        -------
        list[NDArray[np.bool_]]
            One boolean mask per test slice, in ``test_slices`` order.

        Raises
        ------
        ValueError
            If a train slice has no matching ``<stem>_mask.png`` label, or
            the expected trained-model directory does not exist after
            ``run_training`` (a silently crashed trainer must not be
            glob-guessed into a wrong results dir — AGENTS §2).

        Notes
        -----
        ``RuntimeError`` propagates from
        :func:`~liom_toolkit.segmentation.vseg.ssl.warmstart.validate_nnunet_env`
        when any ``nnUNet_*`` env var is unset — the message names the
        missing vars.
        """
        # nnunetv2 is in the [ai] extra — function-scope imports so the
        # module loads with only torch installed.
        from nnunetv2.experiment_planning.plan_and_preprocess_api import (
            extract_fingerprint_dataset,
            plan_experiment_dataset,
            preprocess_dataset,
        )
        from nnunetv2.run.run_training import run_training

        from liom_toolkit.scripts.liom_prepare_nnunet_dataset import (
            prepare_nnunet_2d,
        )
        from liom_toolkit.segmentation.vseg.ssl.warmstart import validate_nnunet_env

        env = validate_nnunet_env()

        train_label_paths = [
            str(Path(p).with_name(f"{Path(p).stem}_mask{Path(p).suffix}")) for p in train_slices
        ]
        # Validate that all mask files exist before starting the long
        # pipeline (no silent failure mid-run).
        for lbl in train_label_paths:
            if not Path(lbl).is_file():
                raise ValueError(
                    f"NnUnetContender: no matching label for a train slice — "
                    f"expected {lbl} (the <name>_mask.png convention)"
                )

        # nnU-Net locates datasets as $nnUNet_raw/Dataset{id:03d}_{name}/ —
        # the raw dir must be written there, not to a local output dir.
        # prepare_nnunet_2d raises FileExistsError if the dir already exists
        # non-empty: a shared nnUNet_raw + fixed dataset_id would otherwise
        # leave stale case files from a previous run that contaminate
        # fingerprint extraction and training.
        dataset_dirname = f"Dataset{self.dataset_id:03d}_LIOM6p5"
        raw_dir = str(Path(env["nnUNet_raw"]) / dataset_dirname)
        prepare_nnunet_2d(
            image_paths=train_slices,
            label_paths=train_label_paths,
            output_dir=raw_dir,
            dataset_id=self.dataset_id,
        )

        extract_fingerprint_dataset(
            self.dataset_id,
            num_processes=self.num_processes,
            check_dataset_integrity=True,
        )
        _plans_dict, plans_identifier = plan_experiment_dataset(self.dataset_id)
        # preprocess_dataset's num_processes is a per-configuration tuple.
        preprocess_dataset(
            self.dataset_id,
            plans_identifier=plans_identifier,
            configurations=("2d",),
            num_processes=(self.num_processes,),
        )
        run_training(
            dataset_name_or_id=str(self.dataset_id),
            configuration="2d",
            fold=0,
            trainer_class_name=self.trainer_name,
            plans_identifier=plans_identifier,
            num_gpus=self.num_gpus,
            device=torch.device(self.device),
        )

        # nnU-Net writes trained models to
        # $nnUNet_results/Dataset{id}_{name}/{trainer}__{plans}__{config}/.
        # Check the expected path explicitly — never glob-guess a results
        # dir after a silently-crashed trainer.
        model_dir = (
            Path(env["nnUNet_results"])
            / dataset_dirname
            / f"{self.trainer_name}__{plans_identifier}__2d"
        )
        if not model_dir.is_dir():
            raise ValueError(
                f"NnUnetContender: expected trained-model dir missing after "
                f"run_training: {model_dir} — the trainer crashed silently "
                f"or wrote elsewhere"
            )

        from liom_toolkit.segmentation.vseg.model_v2 import NnUnetV2Model

        model = NnUnetV2Model(model_dir, device=self.device)
        return self._predict_masks(model, test_slices)

    def predict_on_slices(
        self,
        slices: list[str],
        checkpoint_path: str,
    ) -> list[NDArray[np.bool_]]:
        """Predict from a trained nnU-Net model directory in-process.

        ``checkpoint_path`` is the nnU-Net MODEL DIRECTORY (the
        ``<Dataset>/<trainer>__<plans>__<config>/`` results-layout folder),
        not a single ``.pth`` file — it is handed to the predictor wrapper,
        which validates the directory contract up front.

        Returns
        -------
        list[NDArray[np.bool_]]
            One boolean mask per slice, in ``slices`` order.

        Raises
        ------
        ValueError
            If ``slices`` is empty, or ``checkpoint_path`` fails the
            trained-model directory contract (propagated from the wrapper).
        """
        if not slices:
            raise ValueError("NnUnetContender.predict_on_slices: slices list is empty")

        from liom_toolkit.segmentation.vseg.model_v2 import NnUnetV2Model

        model = NnUnetV2Model(checkpoint_path, device=self.device)
        return self._predict_masks(model, slices)

    def _predict_masks(
        self,
        model: NnUnetV2Model,
        slices: list[str],
    ) -> list[NDArray[np.bool_]]:
        """Predict one boolean mask per slice through the shared wrapper.

        Each slice is read via imageio, cast to float32, promoted to
        ``(1, 1, H, W)`` (channel + dummy z-axis -- the only input rank a
        real nnunetv2 preprocessor accepts, since ``transpose_forward`` is
        always length 3), and passed to the wrapper's ``predict_proba``;
        the vessel channel (index 1) is thresholded at a strict ``> 0.5``
        after dropping the dummy z-axis.

        Parameters
        ----------
        model : NnUnetV2Model
            The initialized predictor wrapper.
        slices : list[str]
            Paths to the 2D image slices to predict on.

        Returns
        -------
        list[NDArray[np.bool_]]
            One boolean mask per slice, in ``slices`` order.

        Raises
        ------
        ValueError
            If the model returns fewer than 2 probability channels — a
            single-class softmax has no vessel channel, so reading
            ``probs[1]`` would be a silent wrong-channel read.
        RuntimeError
            If a returned mask's shape differs from its input slice — a
            shape drift would misalign the eval metrics.
        """
        import imageio.v3 as iio

        masks: list[NDArray[np.bool_]] = []
        for slice_path in slices:
            img = np.asarray(iio.imread(slice_path))
            # Promote to (1, 1, H, W) + 3-element spacing -- a real nnunetv2
            # preprocessor's transpose_forward is always length 3, so a
            # (1,H,W) input crashes on the 4-element transpose permutation.
            # The through-plane spacing is a pass-through for a '2d'
            # configuration, so the in-plane value doubles as placeholder.
            probs = model.predict_proba(
                img.astype(np.float32)[None, None],
                spacing=(self.spacing[0], *self.spacing),
            )
            if probs.shape[0] < 2:
                raise ValueError(
                    f"NnUnetContender: predict_proba returned "
                    f"{probs.shape[0]} probability channel(s) for "
                    f"{slice_path}; the binary vessel contract requires "
                    f"at least 2 (channel 1 is the vessel probability)"
                )
            # probs is (2, 1, H, W) for the (1,1,H,W) input -- drop the
            # dummy z-axis so the mask matches the (H, W) input slice.
            mask = probs[1, 0] > 0.5
            if mask.shape != img.shape:
                raise RuntimeError(
                    f"NnUnetContender: prediction shape {mask.shape} does "
                    f"not match input shape {img.shape} for {slice_path}"
                )
            masks.append(mask)
        return masks
