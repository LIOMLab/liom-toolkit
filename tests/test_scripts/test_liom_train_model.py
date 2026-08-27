"""Smoke + ImportError-surfacing tests for the ``liom-train-model`` CLI.

Exercises:
* ``--help`` exits 0 and contains the 4 shared flags + the curated training
  flags (dataset_file, node_name, --output_train, --epochs, --batch_size,
  --learning_rate, --wandb_entity, --wandb_project, --pretrained_artifact,
  --wandb_mode).
* The CLI surfaces a clear ``ImportError`` mentioning ``PyTorch`` or ``wandb``
  when the ``ai`` extra is not importable (lazy-import guard pattern).
"""

from __future__ import annotations

import subprocess
import sys


def test_liom_train_model_help_exits_0() -> None:
    """liom-train-model --help exits 0 with shared + curated flags."""
    result = subprocess.run(
        ["uv", "run", "liom-train-model", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"liom-train-model --help failed: rc={result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    out = result.stdout + result.stderr
    for flag in ("--log-level", "--resume", "--dask_scheduler", "--n_workers"):
        assert flag in out, f"liom-train-model --help missing {flag}"
    for flag in ("dataset_file", "node_name"):
        assert flag in out, f"liom-train-model --help missing {flag}"
    for flag in (
        "--output_train",
        "--epochs",
        "--batch_size",
        "--learning_rate",
        "--wandb_entity",
        "--wandb_project",
        "--pretrained_artifact",
        "--wandb_mode",
    ):
        assert flag in out, f"liom-train-model --help missing {flag}"


def test_importerror_surfacing_train_model(monkeypatch) -> None:
    """liom-train-model surfaces a clear ImportError mentioning 'PyTorch' or
    'wandb' when torch/wandb is not importable."""
    # Make `import torch` raise ImportError by poisoning sys.modules.
    monkeypatch.setitem(sys.modules, "torch", None)
    # Provide required positional args so parse_args succeeds; the ImportError
    # fires in main()'s lazy-import guard before train_model is called.
    monkeypatch.setattr(
        sys,
        "argv",
        ["liom-train-model", "dataset.zarr", "node_name"],
    )

    import pytest

    from liom_toolkit.scripts.liom_train_model import main

    with pytest.raises(ImportError, match="PyTorch|wandb"):
        main()
