"""Custom-loss nnU-Net v2 trainers for the 6.5 um vessel model.

Two trainer classes plug ``DiceFocalClDiceLoss`` (the composite
Dice-Focal + soft-clDice loss the prior 2D contenders used) into
nnU-Net v2's ``nnUNetTrainer``:

* ``LiomDiceFocalClDiceTrainer`` -- the custom-loss arm: an
  ``nnUNetTrainer`` subclass whose ``_build_loss`` wraps an adapted
  ``DiceFocalClDiceLoss`` in the same ``DeepSupervisionWrapper``
  upstream uses.
* ``LiomDiceFocalClDiceWarmStartTrainer`` -- the combined warm-start +
  custom-loss arm. Identical behavior; it exists solely as a DISTINCT
  CLASS NAME because the trainer name is a component of the nnU-Net
  results directory (``<trainer>__<plans>__<configuration>``). Sharing
  the base arm's name would collide output dirs -- an existing fold_0
  checkpoint would silently resume the wrong arm. Both names must stay
  unique forever.

Adaptation: nnU-Net's binary configuration emits a 2-channel softmax
output ``(B, 2, X, Y)`` and a ``(B, 1, X, Y)`` segmentation target,
while ``DiceFocalClDiceLoss`` expects a single-channel raw logit it
sigmoids internally. ``_BinarySoftmaxToLogitAdapter`` feeds the
log-odds ``log_softmax(net_output, 1)[:, 1:2] - log_softmax(net_output,
1)[:, 0:1]``: since ``sigmoid(log p1 - log p0) == p1`` exactly (the
logsumexp cancels, leaving ``x1 - x0``), the loss's internal sigmoid
reconstructs the true vessel-class probability -- no double-sigmoid and
no raw channel slice that would discard the background channel's
normalization. (The bare ``log_softmax`` vessel channel alone is NOT a
valid logit: ``sigmoid(log p1) == p1/(1+p1)``, not ``p1``.)

Discovery under DDP: nnU-Net resolves custom trainer classes by
scanning the ``nnunetv2.training.nnUNetTrainer`` package tree, then
every directory in the ``nnUNet_extTrainer`` environment variable
(``os.pathsep``-separated, each scanned for ``.py`` files). The env var
is REQUIRED -- ``run_training(num_gpus>1)`` goes through ``mp.spawn``,
whose children are fresh interpreters where module-attribute
registration does not exist. Never register these trainers by assigning
attributes into ``nnunetv2.training.nnUNetTrainer.nnUNetTrainer``; set
``nnUNet_extTrainer`` to the directory containing this file (the
installed package's ``vseg`` directory) before calling
``run_training``.

Arm-3 usage::

    run_training(
        dataset_name_or_id, "2d", fold,
        trainer_class_name="LiomDiceFocalClDiceWarmStartTrainer",
        plans_identifier="nnUNetResEncUNetPlans",
        pretrained_weights=<path to {'network_weights': ...} checkpoint>,
    )

Import contract: this module legitimately requires the full ``[ai]``
extra -- it subclasses an nnunetv2 class at module top, so both torch
and nnunetv2 carry upfront ImportError guards naming ``[ai]`` (the same
guard pattern as ``ssl/warmstart.py`` and ``model_v2.py``).
"""

from __future__ import annotations

import numpy as np

# torch is in the [ai] extra. The upfront ImportError here is the honest
# signal on an io-only install -- the message names [ai], matching the
# guard pattern used in the other vseg modules. The `from e` chain
# preserves the underlying error for debugging (AGENTS section 2).
try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError as e:  # pragma: no cover - exercised only on installs without [ai]
    raise ImportError(
        "Please install liom-toolkit[ai] to use the nnU-Net vessel segmentation trainers."
    ) from e

# nnunetv2 is also in the [ai] extra. Unlike model_v2.py / warmstart.py
# (which import nnunetv2 function-scope so they load with torch alone),
# this module SUBCLASSES an nnunetv2 class at module top -- the import
# cannot be deferred, so a missing nnunetv2 raises the same [ai] guard.
try:
    from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
except ImportError as e:  # pragma: no cover - exercised only on installs without [ai]
    raise ImportError(
        "Please install liom-toolkit[ai] to use the nnU-Net vessel segmentation trainers."
    ) from e

from liom_toolkit.segmentation.vseg.loss import DiceFocalClDiceLoss

__all__ = [
    "LiomDiceFocalClDiceTrainer",
    "LiomDiceFocalClDiceWarmStartTrainer",
]


class _BinarySoftmaxToLogitAdapter(nn.Module):
    """Adapt nnU-Net's 2-class softmax output to a single-logit loss contract.

    Wraps a binary-logit loss (``DiceFocalClDiceLoss``) so it can be
    called with nnU-Net's ``(net_output, target)`` pair: the network's
    ``(B, 2, X, Y)`` raw output is converted to the vessel-class
    log-odds ``log_softmax(net_output, 1)[:, 1:2] -
    log_softmax(net_output, 1)[:, 0:1]`` and the integer segmentation
    target is cast to float.

    The channel difference is the CORRECT single-logit substitute:
    ``sigmoid(log p1 - log p0) == p1`` exactly (the logsumexp cancels,
    so the difference equals ``x1 - x0``), so the inner loss's internal
    sigmoid reconstructs the true vessel-class probability. The bare
    ``log_softmax`` vessel channel is NOT valid -- ``sigmoid(log p1)``
    is ``p1/(1+p1)``, not ``p1``. Feeding ``net_output[:, 1]`` raw would
    discard the background channel's normalization; feeding
    ``sigmoid(net_output)`` or softmax probabilities would
    double-sigmoid inside the loss.

    ``DeepSupervisionWrapper`` calls the wrapped loss per output scale
    as ``loss(pred_i, target_i)`` -- the two-argument ``forward``
    signature matches that contract.
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the inner loss on the vessel-channel logit.

        Parameters
        ----------
        net_output : torch.Tensor
            Network output, ``(B, 2, X, Y)`` raw logits (background,
            vessel).
        target : torch.Tensor
            Segmentation target, ``(B, 1, X, Y)``; cast to ``float32``
            for the inner loss's continuous-overlap terms.

        Returns
        -------
        torch.Tensor
            The inner loss value (scalar).
        """
        # Vessel-class log-odds: log(p1) - log(p0). The logsumexp cancels
        # so this equals net_output[:,1] - net_output[:,0] written through
        # log_softmax -- sigmoid of the result is exactly softmax(x)[1].
        log_probs = F.log_softmax(net_output, dim=1)
        return self.inner(log_probs[:, 1:2] - log_probs[:, 0:1], target.float())


class LiomDiceFocalClDiceTrainer(nnUNetTrainer):
    """nnU-Net v2 trainer using the composite Dice-Focal + soft-clDice loss.

    Identical to stock ``nnUNetTrainer`` except:

    * ``num_epochs`` defaults to 50 (the equalized 250-iterations x
      50-epochs budget the prior contenders trained at, not upstream's
      1000-epoch default).
    * ``_build_loss`` returns the adapted ``DiceFocalClDiceLoss``,
      wrapped in upstream's ``DeepSupervisionWrapper`` with the same
      exponentially-decaying per-scale weights.

    Discovery requires the ``nnUNet_extTrainer`` env var to point at a
    directory containing this file -- see the module docstring.
    """

    def __init__(self, *args: object, num_epochs: int = 50, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.num_epochs = num_epochs

    def _build_loss(self) -> nn.Module:
        """Build the adapted composite loss, wrapped for deep supervision.

        Reproduces upstream nnunetv2 2.8.1 ``_build_loss`` verbatim
        except the inner loss: ``1/2**i`` weight decay over the
        deep-supervision scales, last weight ``0`` -- or ``1e-6`` under
        DDP without ``torch.compile`` (the upstream workaround for DDP's
        unused-parameter crash) -- normalized to sum 1, then wrapped in
        ``DeepSupervisionWrapper``. With ``enable_deep_supervision``
        off, the bare adapter is returned.

        Returns
        -------
        nn.Module
            The adapted loss, or a ``DeepSupervisionWrapper`` around it.
        """
        loss = _BinarySoftmaxToLogitAdapter(DiceFocalClDiceLoss())

        # Upstream weight semantics, verbatim: each deep-supervision
        # output is weighted 1/2**i so higher resolutions dominate; the
        # coarsest output is unused (weight 0). Under DDP without
        # torch.compile a 0 weight crashes on unused parameters, so the
        # coarsest scale keeps 1e-6 instead.
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss


class LiomDiceFocalClDiceWarmStartTrainer(LiomDiceFocalClDiceTrainer):
    """Warm-start twin of ``LiomDiceFocalClDiceTrainer`` -- same loss, distinct name.

    The combined arm loads pretrained encoder weights via
    ``run_training(..., pretrained_weights=<path>)`` into this trainer.
    It shares the parent's ``_build_loss`` and exists only so the arm's
    results dir (``LiomDiceFocalClDiceWarmStartTrainer__<plans>__2d``)
    cannot collide with the base arm's
    (``LiomDiceFocalClDiceTrainer__<plans>__2d``) -- an existing fold
    checkpoint in a shared dir would silently resume the wrong arm.
    """
