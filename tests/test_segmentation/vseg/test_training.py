"""Tests for ``liom_toolkit/segmentation/vseg/training.py`` public API surface.

Covers the lab-config parameterization (CLI-01): ``train_model`` must accept
``wandb_entity`` / ``wandb_project`` / ``pretrained_artifact`` parameters
defaulting to ``None`` (so the toolkit is lab-config-free on import per
PROJECT.md core value), and no ``"liom-lab"`` string may remain in the
module source.

The ``train_model`` body lazy-imports ``torch`` / ``wandb``; these tests only
inspect the public signature and source, so they do NOT require the ``ai``
extra and run on the core-deps CI leg too.
"""

from __future__ import annotations

import inspect

from liom_toolkit.segmentation.vseg import training as training_mod
from liom_toolkit.segmentation.vseg.training import train_model


def test_train_model_has_wandb_entity_param_none_default() -> None:
    """train_model signature has ``wandb_entity: str | None = None``."""
    sig = inspect.signature(train_model)
    assert "wandb_entity" in sig.parameters, "train_model must accept wandb_entity"
    param = sig.parameters["wandb_entity"]
    assert param.default is None, f"wandb_entity must default to None, got {param.default!r}"


def test_train_model_has_wandb_project_param_none_default() -> None:
    """train_model signature has ``wandb_project: str | None = None`` (was hardcoded "vseg")."""
    sig = inspect.signature(train_model)
    assert "wandb_project" in sig.parameters, "train_model must accept wandb_project"
    param = sig.parameters["wandb_project"]
    assert param.default is None, f"wandb_project must default to None, got {param.default!r}"


def test_train_model_has_pretrained_artifact_param_none_default() -> None:
    """train_model signature has ``pretrained_artifact: str | None = None``."""
    sig = inspect.signature(train_model)
    assert "pretrained_artifact" in sig.parameters, (
        "train_model must accept pretrained_artifact"
    )
    param = sig.parameters["pretrained_artifact"]
    assert param.default is None, f"pretrained_artifact must default to None, got {param.default!r}"


def test_train_model_no_liom_lab_hardcoded() -> None:
    """No ``"liom-lab"`` string remains in training.py source (lab-config-free)."""
    source = inspect.getsource(training_mod)
    assert "liom-lab" not in source, (
        "training.py must not hardcode the 'liom-lab' wandb entity — "
        "the toolkit is lab-config-free per PROJECT.md core value"
    )


def test_training_main_block_deleted() -> None:
    """The broken ``if __name__ == '__main__':`` block is deleted from
    training.py — the liom-train-model console script is the real CLI
    (the old block had empty-string placeholders that crashed on invocation)."""
    source = inspect.getsource(training_mod)
    assert "__main__" not in source, (
        "training.py must NOT have an `if __name__ == '__main__':` block — "
        "the liom-train-model console script replaces it"
    )
