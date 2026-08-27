"""Dice and Dice+BCE loss functions for vessel segmentation training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss function."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth: int = 1) -> torch.Tensor:
        """Calculate the Dice loss.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor (logits).
        targets : torch.Tensor
            Target tensor.
        smooth : int
            Smoothing factor to avoid division by zero.

        Returns
        -------
        torch.Tensor
            The Dice loss.
        """
        inputs = torch.sigmoid(inputs)

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
        Dice_BCE = BCE + dice_loss
        return Dice_BCE
