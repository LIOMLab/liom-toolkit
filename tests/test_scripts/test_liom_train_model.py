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
from unittest.mock import patch


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


def test_liom_train_model_main_smoke(tmp_path, fake_torch, fake_wandb, monkeypatch) -> None:
    """``main()`` reaches the real ``train_model`` with the expected kwargs.

    D-01 expansion slice for a torch+wandb-gated CLI. BOTH lazy-import guards
    (torch at lines 106-112 and wandb at lines 113-119) must pass, so both the
    ``fake_torch`` and ``fake_wandb`` fixtures are required. The domain callee
    is spied via ``patch`` on the imported name so the test does not attempt
    real zarr reads + torch model construction. The spy's ``call_args`` kwargs
    are asserted against the verified kwarg map. A kwarg-name typo in
    ``main()``'s call to ``train_model`` raises ``TypeError`` at the
    ``main()`` call site before the spy is invoked.
    """
    dataset_file = str(tmp_path / "dataset.zarr")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "liom-train-model",
            dataset_file,
            "node_name",
            "--output_train",
            str(tmp_path / "training"),
            "--epochs",
            "1",
            "--batch_size",
            "1",
            "--learning_rate",
            "0.001",
            "--wandb_mode",
            "offline",
        ],
    )

    from liom_toolkit.scripts.liom_train_model import main

    with patch("liom_toolkit.segmentation.vseg.training.train_model") as spy:
        main()

    assert spy.called, "main() did not call train_model -- the domain callee was not reached"
    kwargs = spy.call_args.kwargs
    assert kwargs["dataset_file"] == dataset_file
    assert kwargs["node_name"] == "node_name"
    assert kwargs["output_train"] == str(tmp_path / "training")
    assert kwargs["epochs"] == 1
    assert kwargs["batch_size"] == 1
    assert kwargs["learning_rate"] == 0.001
    assert kwargs["wandb_entity"] is None
    assert kwargs["wandb_project"] is None
    assert kwargs["pretrained_artifact"] is None
    assert kwargs["wandb_mode"] == "offline"
    assert kwargs["resume"] is False
