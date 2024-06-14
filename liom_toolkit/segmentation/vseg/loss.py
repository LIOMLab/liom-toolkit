import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice loss function
    """

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth: int = 1) -> torch.Tensor:
        """
        Calculate the Dice loss

        :param inputs: input tensor
        :type inputs: torch.Tensor
        :param targets: target tensor
        :type targets: torch.Tensor
        :param smooth: smoothing factor
        :type smooth: int
        :return: Dice loss
        :rtype: torch.Tensor
        """
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class DiceBCELoss(nn.Module):
    """
    Dice + BCE loss function
    """

    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth: int = 1) -> torch.Tensor:
        """
        Calculate the Dice + BCE loss

        :param inputs: input tensor
        :type inputs: torch.Tensor
        :param targets: target tensor
        :type targets: torch.Tensor
        :param smooth: smoothing factor
        :type smooth: int
        :return: Dice + BCE loss
        :rtype: torch.Tensor
        """
        # inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        return Dice_BCE
