"""Tests for ``liom_toolkit/segmentation/vseg/nnunet_trainer.py``.

Covers the custom-loss nnU-Net trainer pair used by the composite-loss
training arms:

* ``_BinarySoftmaxToLogitAdapter`` adapts nnU-Net's 2-class softmax output
  ``(B, 2, X, Y)`` to the single-channel raw-logit contract of
  ``DiceFocalClDiceLoss`` via ``F.log_softmax(net_output, 1)[:, 1:2]``.
  The loss applies sigmoid internally, and
  ``sigmoid(log_softmax(x)[1]) == softmax(x)[1]`` exactly, so the loss sees
  the true vessel-class probability -- no double-sigmoid, no discarded
  background-channel normalization.
* ``LiomDiceFocalClDiceTrainer._build_loss`` reproduces upstream nnunetv2
  2.8.1 deep-supervision weight semantics verbatim (``1/2**i`` weights,
  last weight ``0`` -- or ``1e-6`` under DDP without ``torch.compile`` --
  normalized to sum 1, wrapped in ``DeepSupervisionWrapper``) with the
  adapted composite loss as the inner loss.
* ``LiomDiceFocalClDiceWarmStartTrainer`` is a distinct subclass name: the
  trainer class name is the nnU-Net results-dir component, so the warm-
  start arm cannot share the base arm's output directory (an existing
  fold_0 checkpoint would silently resume the wrong arm).

Every test needs torch + nnunetv2 (the ``[ai]`` extra); each gates with
``pytest.importorskip`` at the first line of the body per AGENTS section 5.
"""

from __future__ import annotations

import pytest


@pytest.mark.ai
def test_adapter_feeds_vessel_probability() -> None:
    """The adapter's logit channel sigmoids to the exact softmax vessel probability.

    ``DiceFocalClDiceLoss`` applies ``torch.sigmoid`` internally on raw
    logits. Feeding ``F.log_softmax(out, 1)[:, 1:2]`` means the internal
    sigmoid reconstructs ``softmax(out)[1]`` -- the true vessel-class
    probability -- rather than a double-sigmoid or a raw channel slice
    that discards the background normalization.
    """
    pytest.importorskip("torch")
    import torch

    from liom_toolkit.segmentation.vseg.nnunet_trainer import _BinarySoftmaxToLogitAdapter

    class _RecordingLoss(torch.nn.Module):
        """Captures the (inputs, targets) pair the adapter forwards."""

        def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            self.seen_inputs = inputs
            self.seen_targets = targets
            return torch.zeros(())

    inner = _RecordingLoss()
    adapter = _BinarySoftmaxToLogitAdapter(inner)
    net_output = torch.randn(2, 2, 8, 8)
    target = torch.randint(0, 2, (2, 1, 8, 8))

    adapter(net_output, target)

    assert inner.seen_inputs.shape == (2, 1, 8, 8)
    assert torch.allclose(
        torch.sigmoid(inner.seen_inputs),
        torch.softmax(net_output, dim=1)[:, 1:2],
    )


@pytest.mark.ai
def test_adapter_passes_float_target() -> None:
    """The adapter casts the target to float before calling the inner loss.

    nnU-Net hands the loss an integer segmentation target; the composite
    loss's Dice/Focal/clDice components compute continuous overlaps that
    require a float target. The cast happens inside the adapter so the
    inner loss always sees ``float32``.
    """
    pytest.importorskip("torch")
    import torch

    from liom_toolkit.segmentation.vseg.nnunet_trainer import _BinarySoftmaxToLogitAdapter

    class _RecordingLoss(torch.nn.Module):
        def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            self.seen_targets = targets
            return torch.zeros(())

    inner = _RecordingLoss()
    adapter = _BinarySoftmaxToLogitAdapter(inner)
    target = torch.ones(1, 1, 8, 8, dtype=torch.int64)

    adapter(torch.randn(1, 2, 8, 8), target)

    assert inner.seen_targets.dtype == torch.float32
    assert torch.equal(inner.seen_targets, target.float())


@pytest.mark.ai
def test_build_loss_wraps_adapter_with_upstream_weights_non_ddp() -> None:
    """Non-DDP ``_build_loss`` returns a DeepSupervisionWrapper with last weight 0.

    Mirrors upstream nnunetv2 2.8.1 ``_build_loss`` verbatim except the
    inner loss: weights decay as ``1/2**i`` over the deep-supervision
    scales, the last (coarsest) scale gets weight 0 outside the DDP
    workaround, and the vector is normalized to sum 1.
    """
    pytest.importorskip("torch")
    pytest.importorskip("nnunetv2")
    import numpy as np
    from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

    from liom_toolkit.segmentation.vseg.nnunet_trainer import (
        LiomDiceFocalClDiceTrainer,
        _BinarySoftmaxToLogitAdapter,
    )

    trainer = object.__new__(LiomDiceFocalClDiceTrainer)
    trainer.enable_deep_supervision = True
    trainer.is_ddp = False
    trainer._get_deep_supervision_scales = lambda: [(8, 8), (4, 4), (2, 2)]
    trainer._do_i_compile = lambda: False

    loss = trainer._build_loss()

    assert isinstance(loss, DeepSupervisionWrapper)
    assert isinstance(loss.loss, _BinarySoftmaxToLogitAdapter)
    weights = np.asarray(loss.weight_factors, dtype=np.float64)
    # [1, 0.5, 0] normalized -> [2/3, 1/3, 0].
    assert weights.sum() == pytest.approx(1.0)
    assert weights[0] == pytest.approx(2.0 / 3.0)
    assert weights[1] == pytest.approx(1.0 / 3.0)
    assert weights[-1] == 0.0


@pytest.mark.ai
def test_build_loss_ddp_workaround_sets_tiny_last_weight() -> None:
    """Under DDP without compile, the last weight is 1e-6-derived (not 0).

    Upstream's DDP workaround: ``weights[-1] = 0`` makes DDP crash on
    unused parameters, so the coarsest scale keeps a 1e-6 weight (the
    crash does not occur under ``torch.compile``). The override must
    reproduce the workaround exactly -- a 0 here crashes arm training
    under multi-GPU DDP.
    """
    pytest.importorskip("torch")
    pytest.importorskip("nnunetv2")
    import numpy as np
    from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

    from liom_toolkit.segmentation.vseg.nnunet_trainer import LiomDiceFocalClDiceTrainer

    trainer = object.__new__(LiomDiceFocalClDiceTrainer)
    trainer.enable_deep_supervision = True
    trainer.is_ddp = True
    trainer._get_deep_supervision_scales = lambda: [(8, 8), (4, 4), (2, 2)]
    trainer._do_i_compile = lambda: False

    loss = trainer._build_loss()

    weights = np.asarray(loss.weight_factors, dtype=np.float64)
    # [1, 0.5, 1e-6] normalized -> last = 1e-6 / 1.500001.
    assert weights.sum() == pytest.approx(1.0)
    assert weights[-1] == pytest.approx(1e-6 / 1.500001)
    assert weights[-1] > 0.0


@pytest.mark.ai
def test_build_loss_without_deep_supervision_returns_bare_adapter() -> None:
    """``enable_deep_supervision=False`` returns the raw adapter, unwrapped.

    Same contract as upstream: the wrapper is only applied when deep
    supervision is enabled, so disabling it must hand back the inner loss
    itself -- a silently-always-wrapped loss would re-introduce downsampled
    supervision targets the caller turned off.
    """
    pytest.importorskip("torch")
    pytest.importorskip("nnunetv2")

    from liom_toolkit.segmentation.vseg.nnunet_trainer import (
        LiomDiceFocalClDiceTrainer,
        _BinarySoftmaxToLogitAdapter,
    )

    trainer = object.__new__(LiomDiceFocalClDiceTrainer)
    trainer.enable_deep_supervision = False
    trainer.is_ddp = False
    trainer._do_i_compile = lambda: False

    loss = trainer._build_loss()

    assert isinstance(loss, _BinarySoftmaxToLogitAdapter)


@pytest.mark.ai
def test_wrapped_loss_forward_returns_finite_scalar() -> None:
    """A forward through the DeepSupervisionWrapper yields a finite scalar.

    Exercises the adapter inside the wrapper's per-scale ``zip`` call:
    with the DDP workaround active (last weight nonzero) every scale pair
    is evaluated, so the adapter + composite loss must accept the
    (net_output, target) pairs the wrapper hands it. Targets arrive
    integer-valued, mirroring nnU-Net's downsampled segmentation targets.
    """
    pytest.importorskip("torch")
    pytest.importorskip("nnunetv2")
    import torch
    from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

    from liom_toolkit.segmentation.vseg.nnunet_trainer import LiomDiceFocalClDiceTrainer

    trainer = object.__new__(LiomDiceFocalClDiceTrainer)
    trainer.enable_deep_supervision = True
    trainer.is_ddp = True  # nonzero last weight -> every scale is evaluated
    trainer._get_deep_supervision_scales = lambda: [(8, 8), (8, 8)]
    trainer._do_i_compile = lambda: False

    loss = trainer._build_loss()
    assert isinstance(loss, DeepSupervisionWrapper)

    net_outputs = [torch.randn(1, 2, 8, 8), torch.randn(1, 2, 8, 8)]
    targets = [torch.randint(0, 2, (1, 1, 8, 8)), torch.randint(0, 2, (1, 1, 8, 8))]

    value = loss(net_outputs, targets)

    assert value.ndim == 0
    assert torch.isfinite(value)


@pytest.mark.ai
def test_warmstart_trainer_is_distinct_named_subclass() -> None:
    """``LiomDiceFocalClDiceWarmStartTrainer`` is a distinct subclass name.

    The trainer class name is the nnU-Net results-dir component
    (``<trainer>__<plans>__<config>``). The warm-start arm MUST NOT share
    the base arm's class name or the two arms write into the same output
    directory -- an existing fold_0 checkpoint would silently resume the
    wrong arm.
    """
    pytest.importorskip("torch")
    pytest.importorskip("nnunetv2")
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

    from liom_toolkit.segmentation.vseg.nnunet_trainer import (
        LiomDiceFocalClDiceTrainer,
        LiomDiceFocalClDiceWarmStartTrainer,
    )

    assert issubclass(LiomDiceFocalClDiceWarmStartTrainer, LiomDiceFocalClDiceTrainer)
    assert issubclass(LiomDiceFocalClDiceTrainer, nnUNetTrainer)
    assert LiomDiceFocalClDiceWarmStartTrainer.__name__ != LiomDiceFocalClDiceTrainer.__name__


@pytest.mark.ai
def test_trainer_init_takes_num_epochs_default_50() -> None:
    """``__init__`` accepts a ``num_epochs`` keyword defaulting to 50.

    The arm trains at the equalized 250-iterations x 50-epochs budget used
    by the prior contenders, not nnU-Net's default 1000 epochs. The
    keyword must be popped before ``super().__init__`` so upstream's
    signature (which knows no ``num_epochs``) never sees it.
    """
    pytest.importorskip("torch")
    pytest.importorskip("nnunetv2")
    import inspect

    from liom_toolkit.segmentation.vseg.nnunet_trainer import LiomDiceFocalClDiceTrainer

    sig = inspect.signature(LiomDiceFocalClDiceTrainer.__init__)
    assert "num_epochs" in sig.parameters
    assert sig.parameters["num_epochs"].default == 50
    assert sig.parameters["num_epochs"].kind == inspect.Parameter.KEYWORD_ONLY


@pytest.mark.ai
def test_module_docstring_documents_ext_trainer_discovery() -> None:
    """The module docstring documents the ``nnUNet_extTrainer`` requirement.

    Discovery under DDP spawn works only via the ``nnUNet_extTrainer``
    env var (a directory scanned for ``.py`` files in every spawned
    process); module-attribute registration is lost when ``mp.spawn``
    children re-import fresh. The requirement must live in the module
    docstring so anyone wiring a remote run sees it.
    """
    pytest.importorskip("torch")
    pytest.importorskip("nnunetv2")

    from liom_toolkit.segmentation.vseg import nnunet_trainer

    assert "nnUNet_extTrainer" in (nnunet_trainer.__doc__ or "")
