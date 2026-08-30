"""Contender Protocol + architecture implementations for the vseg benchmark.

All four contenders (the legacy 2D U-Net + three architectures wired in a
later plan) share the same ``Contender`` Protocol so the benchmark harness
can train, predict, and score them identically through the ship-gate
eval-metric matrix.

The legacy ``Improved2DContender`` wraps :class:`liom_toolkit.segmentation.vseg.model.VsegModel`
and reuses the existing ``train_model`` orchestration + ``predict_one``
inference path — it is the measured baseline the other architectures are
compared against. The three remaining contenders (MONAI UNet, SwinUNETR,
nnU-Net v2) are skeletal in this plan: the classes exist and satisfy the
``Contender`` Protocol structurally so the harness can enumerate them, but
``train_and_predict`` raises ``NotImplementedError`` until their wiring
lands after the MONAI dependency is added to the ``[ai]`` extra.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

# torch is in the [ai] extra. The upfront ImportError is the honest signal
# on an io-only install — the message names [ai]. The `from e` chain
# preserves the underlying error for debugging (AGENTS §2). MONAI is NOT
# imported here: the three skeletal contenders do not use it yet (their
# wiring lands in a later plan), so importing it at module top would make
# this module un-importable before the dep is added.
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
            epochs=1,
            batch_size=1,
            patch_size=patch_size,
            ddp=ddp,
        )

        model = VsegModel(pretrained=False, device=dev)
        checkpoint_path = Path(output_dir) / "files" / "checkpoint.latest.pth"
        if checkpoint_path.exists():
            state = torch.load(str(checkpoint_path), map_location=dev, weights_only=True)
            model.load_state_dict(state)
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


class MonaiUnetContender:
    """Skeletal MONAI UNet contender (wiring lands after the MONAI dep is added)."""

    name: str = "monai_unet"

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def train_and_predict(
        self,
        train_slices: list[str],
        test_slices: list[str],
        output_dir: str,
        patch_size: tuple[int, int, int] = (1, 256, 256),
        ddp: bool = False,
    ) -> list[NDArray[np.bool_]]:
        """Train + predict — not yet wired (raises NotImplementedError)."""
        raise NotImplementedError(
            "MONAI/nnU-Net contender wiring lands in a later plan after the dep is added to [ai]"
        )

    def predict_on_slices(
        self,
        slices: list[str],
        checkpoint_path: str,
    ) -> list[NDArray[np.bool_]]:
        """Predict from checkpoint — not yet wired (raises NotImplementedError)."""
        raise NotImplementedError(
            "MONAI/nnU-Net contender wiring lands in a later plan after the dep is added to [ai]"
        )


class SwinUnetContender:
    """Skeletal MONAI SwinUNETR contender (wiring lands after the MONAI dep is added)."""

    name: str = "monai_swinunetr"

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def train_and_predict(
        self,
        train_slices: list[str],
        test_slices: list[str],
        output_dir: str,
        patch_size: tuple[int, int, int] = (1, 256, 256),
        ddp: bool = False,
    ) -> list[NDArray[np.bool_]]:
        """Train + predict — not yet wired (raises NotImplementedError)."""
        raise NotImplementedError(
            "MONAI/nnU-Net contender wiring lands in a later plan after the dep is added to [ai]"
        )

    def predict_on_slices(
        self,
        slices: list[str],
        checkpoint_path: str,
    ) -> list[NDArray[np.bool_]]:
        """Predict from checkpoint — not yet wired (raises NotImplementedError)."""
        raise NotImplementedError(
            "MONAI/nnU-Net contender wiring lands in a later plan after the dep is added to [ai]"
        )


class NnUnetContender:
    """Skeletal nnU-Net v2 contender (subprocess bridge wiring lands in a later plan)."""

    name: str = "nnunet_v2"

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def train_and_predict(
        self,
        train_slices: list[str],
        test_slices: list[str],
        output_dir: str,
        patch_size: tuple[int, int, int] = (1, 256, 256),
        ddp: bool = False,
    ) -> list[NDArray[np.bool_]]:
        """Train + predict — not yet wired (raises NotImplementedError)."""
        raise NotImplementedError(
            "MONAI/nnU-Net contender wiring lands in a later plan after the dep is added to [ai]"
        )

    def predict_on_slices(
        self,
        slices: list[str],
        checkpoint_path: str,
    ) -> list[NDArray[np.bool_]]:
        """Predict from checkpoint — not yet wired (raises NotImplementedError)."""
        raise NotImplementedError(
            "MONAI/nnU-Net contender wiring lands in a later plan after the dep is added to [ai]"
        )
