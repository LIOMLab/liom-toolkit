"""Warm-start helper: load a pretrained checkpoint into the nnU-Net 2D ResEnc trainer.

This module is the warm-start transfer layer of the SSL stack: it loads a
pretrained checkpoint (produced by :mod:`liom_toolkit.segmentation.vseg.ssl.pretrain`)
into a fresh nnU-Net 2D ResEnc network via upstream ``load_pretrained_weights``
-- the in-process path that proves the D-01a key-match guarantee (a
checkpoint built by ``build_pretrain_network`` loads into a second same-
architecture network with NO ``AssertionError``).

The helper wraps the upstream ``load_pretrained_weights`` call so a key /
shape mismatch (upstream raises ``AssertionError``) is surfaced as an
actionable ``RuntimeError`` naming the mismatch and the fix (build the
pretraining network with the same ``get_network_from_plans`` call). It
also validates the ``nnUNet_raw`` / ``nnUNet_preprocessed`` / ``nnUNet_results``
env vars are set before instantiating the in-process ``NNUNetTrainer``
(the trainer reads them from the environment; unset vars cause opaque
``KeyError`` / ``FileNotFoundError`` downstream).

This is the IN-PROCESS path: it imports from ``nnunetv2.run.run_training``
directly, NOT the subprocess ``nnunet_bridge.py`` (which is superseded and
slated for deletion). nnunetv2 is in the ``[ai]`` extra and is imported
function-scope so this module loads with only torch installed. Validation
uses ``if ...: raise ValueError(...)`` / ``RuntimeError(...)`` with the
offending value in the message (AGENTS section 2 -- never ``assert`` for
validation, never a silent partial-load fallback).
"""

from __future__ import annotations

import os
from pathlib import Path

# torch is in the [ai] extra. The upfront ImportError here is the honest
# signal on an io-only install -- the message names [ai,benchmark] (the
# torch + MONAI path the SSL stack needs). The `from e` chain preserves
# the underlying error for debugging (AGENTS section 2). nnunetv2 is
# imported function-scope so this module loads with only torch installed.
try:
    import torch
    from torch import nn
except ImportError as e:  # pragma: no cover - exercised only on io-only installs
    raise ImportError(
        "Please install liom-toolkit[ai,benchmark] to use the SSL warm-start helper."
    ) from e

__all__ = ["load_pretrained_checkpoint", "validate_nnunet_env", "warm_start"]

# The three nnU-Net v2 environment variables the in-process NNUNetTrainer
# reads to locate raw data, preprocessed data, and results. The helper
# validates they are set before instantiating the trainer so an unset env
# surfaces as a clear RuntimeError naming the missing vars, not an opaque
# KeyError / FileNotFoundError several frames into the trainer init.
_NNUNET_ENV_VARS: tuple[str, ...] = ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")


def validate_nnunet_env() -> dict[str, str]:
    """Validate the nnU-Net v2 environment variables are set; return them as a dict.

    The in-process ``NNUNetTrainer`` reads ``nnUNet_raw`` /
    ``nnUNet_preprocessed`` / ``nnUNet_results`` from the environment to
    locate raw data, preprocessed data, and results. If any is unset the
    trainer raises an opaque ``KeyError`` / ``FileNotFoundError`` several
    frames into its init. This helper checks them up front and raises a
    clear ``RuntimeError`` naming the missing vars so the caller can act
    (explicit failure, no silent pass on an empty env).

    Returns
    -------
    dict[str, str]
        The three env vars as a ``{var_name: value}`` dict (the values are
        the resolved env strings).

    Raises
    ------
    RuntimeError
        If any of the three ``nnUNet_*`` env vars is unset. The message
        names the missing vars.
    """
    missing = [v for v in _NNUNET_ENV_VARS if v not in os.environ]
    if missing:
        raise RuntimeError(
            f"nnU-Net v2 environment variables must be set before instantiating "
            f"the trainer (missing: {missing}). Set them to the nnU-Net raw / "
            f"preprocessed / results directories on the lab box."
        )
    return {v: os.environ[v] for v in _NNUNET_ENV_VARS}


def load_pretrained_checkpoint(
    checkpoint_path: str,
    network: nn.Module,
    *,
    verbose: bool = False,
) -> None:
    """Load a pretrained checkpoint into ``network`` via upstream ``load_pretrained_weights``.

    The checkpoint must be in the format ``{'network_weights': state_dict}``
    (the format :func:`masked_inpainting_pretrain` saves and
    ``load_pretrained_weights`` expects). The transfer is by state_dict
    key-name + shape match; keys containing ``.seg_layers.`` (the
    segmentation heads) are skipped.

    When the upstream loader raises ``AssertionError`` (a key is missing or
    a shape mismatches between the pretrained checkpoint and the network),
    this wrapper catches it and re-raises as ``RuntimeError`` with an
    actionable message naming the mismatch and the fix: build the
    pretraining network with the same ``get_network_from_plans`` call the
    warm-start network uses (the D-01a key-match guarantee). This is the
    AGENTS section 2 no-silent-fallback discipline -- surface the key-
    mismatch explicitly rather than silently partial-loading.

    Parameters
    ----------
    checkpoint_path : str
        Path to the pretrained checkpoint file.
    network : nn.Module
        The nnU-Net 2D ResEnc network to load the weights into (built via
        :func:`build_pretrain_network` with the same plan).
    verbose : bool, optional
        Forwarded to ``load_pretrained_weights`` (prints the overlapping
        blocks). Defaults to ``False``.

    Raises
    ------
    ValueError
        If ``checkpoint_path`` does not point to an existing file. The
        message includes the offending path.
    RuntimeError
        If the upstream ``load_pretrained_weights`` raises
        ``AssertionError`` (key / shape mismatch). The message names the
        mismatch and the fix.
    """
    # nnunetv2 is in the [ai] extra -- import function-scope so the module
    # loads with only torch installed.
    from nnunetv2.run.load_pretrained_weights import load_pretrained_weights

    if not Path(checkpoint_path).is_file():
        raise ValueError(f"pretrained checkpoint not found: {checkpoint_path}")
    try:
        load_pretrained_weights(network, checkpoint_path, verbose=verbose)
    except AssertionError as e:
        raise RuntimeError(
            f"pretrained checkpoint key/shape mismatch -- the pretraining "
            f"network must be the SAME nnU-Net 2D ResEnc architecture built "
            f"via get_network_from_plans as the warm-start network (the "
            f"key-match guarantee). Upstream load_pretrained_weights error: {e}"
        ) from e


def warm_start(
    dataset_name_or_id: int | str,
    fold: int,
    pretrained_weights_file: str,
    *,
    configuration: str = "2d",
    device: torch.device | str = "cuda",
    trainer_name: str = "nnUNetTrainer",
    plans_identifier: str = "nnUNetResEncUNetPlans",
    validation_only: bool = False,
    export_validation_probabilities: bool = False,
) -> None:
    """Warm-start the nnU-Net v2 2D ResEnc trainer with a pretrained checkpoint.

    Instantiates the in-process ``NNUNetTrainer`` (no subprocess), loads the
    pretrained weights via ``maybe_load_checkpoint(pretrained_weights_file=...)``
    (which calls ``load_pretrained_weights`` BEFORE ``run_training``), then
    runs training + validation. This is the in-process warm-start path
    (RESEARCH Pattern 2) -- it imports from
    ``nnunetv2.run.run_training`` directly, NOT the subprocess
    ``nnunet_bridge``.

    Pretrained weights can only be used at the BEGINNING of training;
    ``maybe_load_checkpoint`` raises ``RuntimeError`` if both
    ``continue_training`` and ``pretrained_weights_file`` are set. This
    helper never passes ``continue_training=True`` (it always starts a
    fresh warm-started training).

    Parameters
    ----------
    dataset_name_or_id : int | str
        The nnU-Net dataset name or numeric id (e.g. ``"Dataset001_Vessel"``).
    fold : int
        The cross-validation fold to train.
    pretrained_weights_file : str
        Path to the pretrained checkpoint (``{'network_weights': state_dict}``).
    configuration : str, optional
        The nnU-Net configuration. Defaults to ``"2d"`` (the Phase-14-
        settled configuration -- 3D pretraining is deferred).
    device : torch.device | str, optional
        The device to train on. Defaults to ``"cuda"`` (the real run; the
        tracer passes ``"cpu"``).
    trainer_name : str, optional
        The nnU-Net trainer class name. Defaults to ``"nnUNetTrainer"``.
    plans_identifier : str, optional
        The plans identifier. Defaults to ``"nnUNetResEncUNetPlans"`` (the
        ResEnc plans -- the Phase-14-settled architecture).
    validation_only : bool, optional
        If ``True``, skip training and only run validation on the existing
        checkpoint. Defaults to ``False``.
    export_validation_probabilities : bool, optional
        If ``True``, export the validation softmax probabilities (large
        files). Defaults to ``False``.

    Raises
    ------
    ValueError
        If ``pretrained_weights_file`` does not point to an existing file.

    Notes
    -----
    ``RuntimeError`` propagates from :func:`validate_nnunet_env` (unset
    nnU-Net env vars) and :func:`load_pretrained_checkpoint` (key/shape
    mismatch) -- see those helpers' docstrings.
    """
    # nnunetv2 is in the [ai] extra -- import function-scope so the module
    # loads with only torch installed.
    from nnunetv2.run.run_training import get_trainer_from_args, maybe_load_checkpoint

    validate_nnunet_env()
    if not Path(pretrained_weights_file).is_file():
        raise ValueError(f"pretrained checkpoint not found: {pretrained_weights_file}")
    if isinstance(device, str):
        device = torch.device(device)

    trainer = get_trainer_from_args(
        dataset_name_or_id,
        configuration=configuration,
        fold=fold,
        trainer_name=trainer_name,
        plans_identifier=plans_identifier,
        device=device,
    )
    # maybe_load_checkpoint calls load_pretrained_weights(trainer.network,
    # pretrained_weights_file) BEFORE run_training. Pretrained weights can
    # only be used at the BEGINNING of training; never pass both
    # continue_training=True and pretrained_weights_file (maybe_load_checkpoint
    # raises RuntimeError if both are set).
    maybe_load_checkpoint(
        trainer,
        continue_training=False,
        validation_only=validation_only,
        pretrained_weights_file=pretrained_weights_file,
    )
    if not validation_only:
        trainer.run_training()
    trainer.perform_actual_validation(
        export_validation_probabilities=export_validation_probabilities
    )
