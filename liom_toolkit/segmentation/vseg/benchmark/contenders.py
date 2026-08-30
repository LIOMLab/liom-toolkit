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
* ``NnUnetContender`` — nnU-Net v2 via the subprocess bridge
  (:func:`~liom_toolkit.segmentation.vseg.benchmark.nnunet_bridge.nnunet_predict`).
  nnU-Net runs in a separate venv (torch-clobbering isolation); the
  contender converts the train slices to nnU-Net raw format via
  :func:`~liom_toolkit.scripts.liom_prepare_nnunet_dataset.prepare_nnunet_2d`,
  shells out to nnU-Net, and reads the predicted PNGs back as boolean masks.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

# torch is in the [ai] extra. The upfront ImportError is the honest signal
# on an io-only install — the message names [ai]. The `from e` chain
# preserves the underlying error for debugging (AGENTS §2). MONAI is imported
# inside the contender methods (function-scope) so this module imports
# cleanly with only torch installed; MONAI is a separate [ai] dep.
try:
    import torch
except ImportError as e:
    raise ImportError("Please install liom-toolkit[ai] to use the benchmark contenders.") from e

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
            batch_size=1,
            patch_size=patch_size,
            ddp=ddp,
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
    """nnU-Net v2 contender — subprocess bridge to a separate venv.

    nnU-Net v2 pins its own torch/CUDA build that conflicts with the
    liom-toolkit ``[ai]`` extra's torch (the torch-clobbering hazard). This
    contender runs nnU-Net as a subprocess in a separate venv via
    :func:`~liom_toolkit.segmentation.vseg.benchmark.nnunet_bridge.nnunet_predict`
    — the liom-toolkit process never imports ``nnunetv2``.

    The contender converts the train slices to nnU-Net v2 raw format via
    :func:`~liom_toolkit.scripts.liom_prepare_nnunet_dataset.prepare_nnunet_2d`,
    invokes ``nnUNetv2_predict`` (subprocess bridge), and reads the predicted
    PNGs back as boolean masks (nnU-Net writes 0/255 uint8 PNGs; binarizing
    to bool is required so the eval-metric matrix receives
    ``NDArray[np.bool_]`` — a 0/255 uint8 array passed to ``cl_score`` would
    scale tprec/tsens by 255 → wrong values).
    """

    name: str = "nnunet_v2"

    def __init__(
        self,
        device: str = "cpu",
        dataset_id: int = 999,
        nnunet_venv_python: str | None = None,
        num_gpus: int = 1,
    ) -> None:
        """Initialise the contender.

        Parameters
        ----------
        device : str
            Unused for nnU-Net (it runs in its own venv with its own device
            config) — kept for Protocol structural conformance.
        dataset_id : int
            The nnU-Net dataset id to use for the converter + predictor.
        nnunet_venv_python : str | None
            Path to the Python interpreter in the separate nnU-Net venv
            (torch-clobbering isolation). Required — there is no
            lab-independent default. A lab that installs nnU-Net elsewhere
            must pass its own venv-python path (AGENTS §1: no hardcoded lab
            config). ``None`` raises :class:`ValueError`.
        num_gpus : int
            Number of GPUs for nnU-Net's built-in DDP training
            (``-num_gpus`` CLI flag). Default 1.

        Raises
        ------
        ValueError
            If ``nnunet_venv_python`` is ``None`` (no lab-independent
            default exists).
        """
        if nnunet_venv_python is None:
            raise ValueError(
                "nnunet_venv_python is required — path to the Python "
                "interpreter in the separate nnU-Net venv "
                "(torch-clobbering isolation). There is no lab-independent "
                "default; pass the venv-python path for your environment."
            )
        self.device = device
        self.dataset_id = dataset_id
        self.nnunet_venv_python = nnunet_venv_python
        self.num_gpus = num_gpus

    def train_and_predict(
        self,
        train_slices: list[str],
        test_slices: list[str],
        output_dir: str,
        patch_size: tuple[int, int, int] = (1, 256, 256),
        ddp: bool = False,
    ) -> list[NDArray[np.bool_]]:
        """Full nnU-Net pipeline: convert → preprocess → train → predict → read back.

        Runs the complete nnU-Net v2 subprocess sequence in the separate
        venv (torch-clobbering isolation):

        1. Convert train slices + their ``<stem>_mask.png`` labels to
           nnU-Net raw format via
           :func:`~liom_toolkit.scripts.liom_prepare_nnunet_dataset.prepare_nnunet_2d`.
        2. ``nnUNetv2_plan_and_preprocess`` — nnU-Net self-configures its
           patch/batch/architecture from the dataset statistics.
        3. ``nnUNetv2_train`` (2d config, fold 0) — trains the model.
        4. Copy test slices to a temp input folder in nnU-Net's
           ``{case}_0000.png`` naming convention.
        5. ``nnUNetv2_predict`` — predicts on the test folder.
        6. Read predictions back as boolean masks (nnU-Net writes 0/255
           uint8 PNGs; binarizing to bool is required so the eval-metric
           matrix receives ``NDArray[np.bool_]``).

        The ``ddp`` flag is unused — nnU-Net manages its own
        multi-GPU training internally. Kept for Protocol structural
        conformance.

        Returns
        -------
        list[NDArray[np.bool_]]
            One boolean mask per test slice, in ``test_slices`` order.

        Raises
        ------
        ValueError
            If a train slice has no matching ``<name>_mask.png`` label.
        RuntimeError
            If any nnU-Net subprocess exits non-zero (the returncode and
            stderr tail are in the message), or if a prediction file is
            missing after ``nnUNetv2_predict`` (no silent zero-mask
            fallback — AGENTS §2).
        """
        import imageio.v3 as iio

        from liom_toolkit.scripts.liom_prepare_nnunet_dataset import (
            prepare_nnunet_2d,
        )
        from liom_toolkit.segmentation.vseg.benchmark.nnunet_bridge import (
            nnunet_plan_and_preprocess,
            nnunet_predict,
            nnunet_train,
        )

        # 1. Convert train slices + masks to nnU-Net raw format.
        # nnU-Net looks for datasets in $nnUNet_raw/Dataset{id:03d}_{name}/,
        # so prepare_nnunet_2d must write there (not to a local output dir).
        # The env vars are validated by nnunet_plan_and_preprocess below,
        # but we need them here too — read them early with an explicit check.
        # The nnUNet_raw env var name is mandated by the nnU-Net v2 CLI
        # (upstream convention — renaming would break nnU-Net's dataset lookup).
        nnunet_raw_env = os.environ.get("nnUNet_raw")  # ruff: ignore[uncapitalized-environment-variables]
        if nnunet_raw_env is None:
            raise RuntimeError(
                "NnUnetContender: nnUNet_raw env var is not set — "
                "nnU-Net needs it to locate the raw dataset directory"
            )
        train_label_paths = [
            str(Path(p).with_name(f"{Path(p).stem}_mask{Path(p).suffix}")) for p in train_slices
        ]
        # Validate that all mask files exist before starting the long
        # subprocess chain (no silent failure mid-pipeline).
        for lbl in train_label_paths:
            if not Path(lbl).is_file():
                raise ValueError(
                    f"NnUnetContender: no matching label for a train slice — "
                    f"expected {lbl} (the <name>_mask.png convention)"
                )
        raw_dir = str(Path(nnunet_raw_env) / f"Dataset{self.dataset_id:03d}_LIOM6p5")
        prepare_nnunet_2d(
            image_paths=train_slices,
            label_paths=train_label_paths,
            output_dir=raw_dir,
            dataset_id=self.dataset_id,
        )

        # 2. Plan + preprocess (nnU-Net self-configures).
        nnunet_plan_and_preprocess(
            dataset_id=self.dataset_id,
            nnunet_venv_python=self.nnunet_venv_python,
        )

        # 3. Train (2d config, fold 0, 50-epoch trainer for fair comparison).
        nnunet_train(
            dataset_id=self.dataset_id,
            configuration="2d",
            fold=0,
            num_gpus=self.num_gpus,
            nnunet_venv_python=self.nnunet_venv_python,
        )

        # 4. Copy test slices to a temp input folder in nnU-Net's
        #    {case}_0000.png naming convention (single channel = _0000).
        predict_input = Path(output_dir) / "nnunet_predict_input"
        predict_input.mkdir(parents=True, exist_ok=True)
        case_names: list[str] = []
        for slice_path in test_slices:
            stem = Path(slice_path).stem
            case_name = f"{stem}_0000"
            iio.imwrite(
                predict_input / f"{case_name}.png",
                iio.imread(slice_path),
            )
            case_names.append(stem)

        # 5. Predict.
        predict_output = Path(output_dir) / "nnunet_predict_output"
        predict_output.mkdir(parents=True, exist_ok=True)
        nnunet_predict(
            input_folder=str(predict_input),
            output_folder=str(predict_output),
            dataset_id=self.dataset_id,
            configuration="2d",
            fold="0",
            nnunet_venv_python=self.nnunet_venv_python,
        )

        # 6. Read predictions back as bool masks. nnU-Net writes
        #    {case}.png (without the _0000 channel suffix) to the output
        #    folder. Raise on a missing prediction file — no silent
        #    zero-mask fallback (AGENTS §2).
        masks: list[NDArray[np.bool_]] = []
        for stem in case_names:
            pred_path = predict_output / f"{stem}.png"
            if not pred_path.is_file():
                raise RuntimeError(
                    f"NnUnetContender: nnU-Net prediction file missing: "
                    f"{pred_path} — nnUNetv2_predict likely failed for this "
                    f"case (no silent zero-mask fallback)"
                )
            masks.append(np.asarray(iio.imread(pred_path), dtype=bool))
        return masks

    def predict_on_slices(
        self,
        slices: list[str],
        checkpoint_path: str,
    ) -> list[NDArray[np.bool_]]:
        """Predict from a trained nnU-Net model via the subprocess bridge.

        ``checkpoint_path`` here is the nnU-Net output folder (the directory
        nnU-Net wrote predictions to in a prior run); the bridge re-invokes
        ``nnUNetv2_predict`` on the slices.

        Returns
        -------
        list[NDArray[np.bool_]]
            One boolean mask per slice, in ``slices`` order.

        Raises
        ------
        ValueError
            If ``slices`` is empty, or the slices span more than one parent
            folder (the bridge predicts on a single input folder).
        RuntimeError
            If a prediction file is missing after ``nnUNetv2_predict`` (no
            silent zero-mask fallback — AGENTS §2).
        """
        import imageio.v3 as iio

        from liom_toolkit.segmentation.vseg.benchmark.nnunet_bridge import nnunet_predict

        if not slices:
            raise ValueError("NnUnetContender.predict_on_slices: slices list is empty")
        parents = {Path(s).parent for s in slices}
        if len(parents) != 1:
            raise ValueError(
                f"NnUnetContender.predict_on_slices: all slices must be in one "
                f"folder (the bridge predicts on a single input folder); got "
                f"{len(parents)} distinct folders"
            )
        nnunet_predict(
            input_folder=str(Path(slices[0]).parent),
            output_folder=checkpoint_path,
            dataset_id=self.dataset_id,
            configuration="2d",
            fold="all",
            nnunet_venv_python=self.nnunet_venv_python,
        )
        masks: list[NDArray[np.bool_]] = []
        for slice_path in slices:
            pred_path = Path(checkpoint_path) / f"{Path(slice_path).stem}.png"
            if not pred_path.is_file():
                raise RuntimeError(
                    f"NnUnetContender.predict_on_slices: prediction file "
                    f"missing: {pred_path} — nnUNetv2_predict likely failed "
                    f"for this case (no silent zero-mask fallback, AGENTS §2)"
                )
            masks.append(np.asarray(iio.imread(pred_path), dtype=bool))
        return masks
