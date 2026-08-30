"""Loss functions for vessel segmentation training.

All losses in this module expect RAW LOGITS (not sigmoid-activated
probabilities). The model (:class:`~liom_toolkit.segmentation.vseg.model.VsegModel`)
outputs logits from its final conv layer — sigmoid is applied internally
by the loss functions for numerical stability (the log-sum-exp trick in
``BCEWithLogitsLoss``).

The primary loss for benchmark training is :class:`DiceFocalClDiceLoss`,
a composite of Dice-Focal (handles class imbalance — vessels are sparse)
and soft-clDice (preserves vessel centerline topology / connectivity).
This matches the loss used by the MONAI contenders so the architecture
comparison isolates the model architecture, not the loss function.
"""

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

    Expects ``inputs`` to be RAW LOGITS. Sigmoid is applied internally
    before computing the Dice coefficient.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth: int = 1) -> torch.Tensor:
        """Calculate the Dice loss.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor (raw logits — sigmoid is applied internally).
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
    """Dice + BCE loss function.

    Expects ``inputs`` to be RAW LOGITS. Uses ``BCEWithLogitsLoss``
    (the numerically stable fused sigmoid+BCE) instead of the old
    ``F.binary_cross_entropy`` on probabilities.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth: int = 1) -> torch.Tensor:
        """Calculate the Dice + BCE loss.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor (raw logits — sigmoid is applied internally).
        targets : torch.Tensor
            Target tensor.
        smooth : int
            Smoothing factor to avoid division by zero.

        Returns
        -------
        torch.Tensor
            The combined Dice + BCE loss.
        """
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        # BCEWithLogitsLoss fuses sigmoid + BCE for numerical stability.
        BCE = F.binary_cross_entropy_with_logits(inputs_flat, targets_flat, reduction="mean")
        # Dice on sigmoid(logits) for gradient flow through both paths.
        probs = torch.sigmoid(inputs_flat)
        intersection = (probs * targets_flat).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (probs.sum() + targets_flat.sum() + smooth)
        return BCE + dice_loss


class FocalLoss(nn.Module):
    """Focal loss for binary segmentation.

    Down-weights easy (well-classified) pixels and focuses training on
    hard (misclassified) pixels — critical for class-imbalanced vessel
    segmentation where background dominates.

    Expects RAW LOGITS. Applies sigmoid internally.

    Parameters
    ----------
    gamma : float
        Focusing parameter. ``gamma=2`` is the standard value from the
        original paper (Lin et al., 2017). Higher gamma increases the
        down-weighting of easy examples.
    alpha : float | None
        Weighting factor for the positive class (vessels). ``None`` uses
        no class weighting. ``0.25`` is the paper's default for imbalanced
        datasets.
    """

    def __init__(self, gamma: float = 2.0, alpha: float | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate the focal loss.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor (raw logits — sigmoid is applied internally).
        targets : torch.Tensor
            Target tensor (float32, 0.0 or 1.0).

        Returns
        -------
        torch.Tensor
            The focal loss (scalar).
        """
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        # BCEWithLogitsLoss with reduction="none" gives per-pixel loss
        # (sigmoid + BCE fused for numerical stability).
        bce = F.binary_cross_entropy_with_logits(inputs_flat, targets_flat, reduction="none")
        probs = torch.sigmoid(inputs_flat)
        # p_t = probability of the correct class.
        p_t = probs * targets_flat + (1 - probs) * (1 - targets_flat)
        # Focal modulating factor: (1 - p_t)^gamma.
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * bce
        if self.alpha is not None:
            # alpha weighting for the positive class.
            alpha_t = self.alpha * targets_flat + (1 - self.alpha) * (1 - targets_flat)
            loss = alpha_t * loss
        return loss.mean()


class SoftClDiceLoss(nn.Module):
    """Soft centerline Dice loss for topology preservation.

    Computes a soft skeletonization of the predicted and target masks
    using iterative min-pooling, then computes a Dice-like overlap on
    the skeletons. This encourages the model to preserve vessel
    connectivity and thin branches that pure Dice loss misses.

    Expects RAW LOGITS. Applies sigmoid internally.

    Parameters
    ----------
    iter_ : int
        Number of min-pooling iterations for soft skeletonization.
        Higher values capture longer-range connectivity. ``iter_=3``
        matches the MONAI ``SoftclDiceLoss`` default.
    smooth : int
        Smoothing factor for the Dice-like computation.
    """

    def __init__(self, iter_: int = 3, smooth: int = 1) -> None:
        super().__init__()
        self.iter_ = iter_
        self.smooth = smooth

    def _soft_skeletonize(self, prob: torch.Tensor) -> torch.Tensor:
        """Soft skeletonization via iterative min-pooling + max-pooling.

        Produces a soft (differentiable) approximation of the vessel
        centerline / skeleton.

        Returns
        -------
        torch.Tensor
            The soft skeleton (same shape as input).
        """
        for _ in range(self.iter_):
            min_pool = -F.max_pool2d(-prob, kernel_size=3, stride=1, padding=1)
            max_pool = F.max_pool2d(prob, kernel_size=3, stride=1, padding=1)
            # The skeleton is the difference between the min-pooled
            # (eroded) and max-pooled (dilated) versions — the thin
            # centerline that survives erosion but not dilation.
            skeleton = F.relu(min_pool - max_pool)
            prob = skeleton
        return prob

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate the soft-clDice loss.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor (raw logits — sigmoid is applied internally).
        targets : torch.Tensor
            Target tensor (float32, 0.0 or 1.0).

        Returns
        -------
        torch.Tensor
            The soft-clDice loss (1 - cl_dice_score).
        """
        probs = torch.sigmoid(inputs)
        # Skeletonize both prediction and target.
        pred_skel = self._soft_skeletonize(probs)
        target_skel = self._soft_skeletonize(targets)
        # clDice: overlap of skeletons, normalized by skeleton lengths.
        # tprec: fraction of predicted skeleton that overlaps target mask.
        tprec = (pred_skel * targets).sum() / (pred_skel.sum() + self.smooth)
        # tsens: fraction of target skeleton that overlaps predicted mask.
        tsens = (target_skel * probs).sum() / (target_skel.sum() + self.smooth)
        cl_dice = 2.0 * tprec * tsens / (tprec + tsens + self.smooth)
        return 1.0 - cl_dice


class DiceFocalClDiceLoss(nn.Module):
    """Composite Dice-Focal + soft-clDice loss.

    Combines:
    - **Dice-Focal**: Dice loss + Focal loss. Handles class imbalance
      (vessels are sparse) by down-weighting easy background pixels.
    - **Soft-clDice**: Centerline topology loss. Preserves vessel
      connectivity and thin branches.

    This matches the loss used by the MONAI contenders
    (``DiceFocalLoss(sigmoid=True) + SoftclDiceLoss(sigmoid=True)``)
    so the architecture benchmark isolates the model architecture, not
    the loss function.

    Expects RAW LOGITS. All components apply sigmoid internally.

    Parameters
    ----------
    lambda_cldice : float
        Weight of the soft-clDice component in the composite loss.
        ``loss = DiceFocal + lambda_cldice * soft_clDice``.
        Default ``0.5``.
    gamma : float
        Focal loss focusing parameter. Default ``2.0``.
    """

    def __init__(self, lambda_cldice: float = 0.5, gamma: float = 2.0) -> None:
        super().__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(gamma=gamma)
        self.cldice_loss = SoftClDiceLoss(iter_=3)
        self.lambda_cldice = lambda_cldice

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate the composite Dice-Focal + soft-clDice loss.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor (raw logits — sigmoid is applied internally).
        targets : torch.Tensor
            Target tensor (float32, 0.0 or 1.0).

        Returns
        -------
        torch.Tensor
            The composite loss.
        """
        dice = self.dice_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        cldice = self.cldice_loss(inputs, targets)
        return dice + focal + self.lambda_cldice * cldice
