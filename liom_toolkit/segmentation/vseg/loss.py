"""Dice and Dice+BCE loss functions for vessel segmentation training."""

from __future__ import annotations

# torch is in the [ai] extra. The upfront ImportError here is the honest
# signal on an io-only install -- the message names [ai] (the torch path),
# matching the guard pattern used in the other vseg modules. The `from e`
# chain preserves the underlying error for debugging (AGENTS §2).
try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError as e:
    raise ImportError(
        "Please install liom-toolkit[ai] to use the vessel segmentation loss functions."
    ) from e


class DiceLoss(nn.Module):
    """Dice loss function.

    Expects ``inputs`` to be PROBABILITIES in ``[0, 1]`` (sigmoid already
    applied), matching :class:`DiceBCELoss` and the package's own
    :class:`~liom_toolkit.segmentation.vseg.model.VsegModel`, whose
    ``forward`` applies ``nn.Sigmoid()``. Applying ``torch.sigmoid`` here
    as well would compute the loss on ``sigmoid(sigmoid(logits))`` --
    wrong gradients and wrong loss value. If you have raw logits, apply
    sigmoid before passing them in.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth: int = 1) -> torch.Tensor:
        """Calculate the Dice loss.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor (probabilities in ``[0, 1]`` -- sigmoid already
            applied). Raw logits must be sigmoided by the caller before
            passing them in.
        targets : torch.Tensor
            Target tensor.
        smooth : int
            Smoothing factor to avoid division by zero.

        Returns
        -------
        torch.Tensor
            The Dice loss.
        """
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class DiceBCELoss(nn.Module):
    """Dice + BCE loss function."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth: int = 1) -> torch.Tensor:
        """Calculate the Dice + BCE loss.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor (probabilities — sigmoid is NOT applied here).
        targets : torch.Tensor
            Target tensor.
        smooth : int
            Smoothing factor to avoid division by zero.

        Returns
        -------
        torch.Tensor
            The combined Dice + BCE loss.
        """
        # inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction="mean")
        return BCE + dice_loss
