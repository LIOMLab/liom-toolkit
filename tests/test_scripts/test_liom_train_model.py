"""Smoke + ImportError-surfacing tests for the ``liom-train-model`` CLI.

Exercises:
* ``--help`` exits 0 and contains the 4 shared flags + the curated training
  flags (dataset_file, node_name, --output-train, --epochs, --batch-size,
  --learning-rate, --wandb-entity, --wandb-project, --pretrained-artifact,
  --wandb-mode).
* The CLI surfaces a clear ``ImportError`` mentioning ``PyTorch`` or ``wandb``
  when the ``ai`` extra is not importable (lazy-import guard pattern).

The ``--help`` smoke check invokes the script's ``_build_argument_parser()``
in-process and formats its help text, rather than spawning a ``uv run``
subprocess (avoids the per-invocation venv-reconcile + interpreter-startup
cost).
"""

from __future__ import annotations

import sys
from unittest.mock import patch


def test_liom_train_model_help_exits_0() -> None:
    """liom-train-model --help contains shared + curated flags."""
    from liom_toolkit.scripts.liom_train_model import _build_argument_parser

    out = _build_argument_parser().format_help()
    for flag in ("--log-level", "--resume", "--dask-scheduler", "--n-workers"):
        assert flag in out, f"liom-train-model --help missing {flag}"
    for flag in ("dataset_file", "node_name"):
        assert flag in out, f"liom-train-model --help missing {flag}"
    for flag in (
        "--output-train",
        "--epochs",
        "--batch-size",
        "--learning-rate",
        "--wandb-entity",
        "--wandb-project",
        "--pretrained-artifact",
        "--wandb-mode",
    ):
        assert flag in out, f"liom-train-model --help missing {flag}"


def test_importerror_surfacing_train_model(tmp_path, monkeypatch) -> None:
    """liom-train-model surfaces a clear ImportError mentioning 'PyTorch' or
    'wandb' when torch/wandb is not importable.

    The dataset_file positional must exist on disk so the post-parse
    file-existence check (which runs BEFORE the import torch guard) does not
    exit 2 first; the ImportError then fires in main()'s lazy-import guard.
    """
    # Make `import torch` raise ImportError by poisoning sys.modules.
    monkeypatch.setitem(sys.modules, "torch", None)
    # Materialize the dataset path so the D-01 file-existence check passes
    # before the import torch guard fires.
    dataset_file = tmp_path / "dataset.zarr"
    dataset_file.touch()
    monkeypatch.setattr(
        sys,
        "argv",
        ["liom-train-model", str(dataset_file), "node_name"],
    )

    import pytest

    from liom_toolkit.scripts.liom_train_model import main

    with pytest.raises(ImportError, match=r"PyTorch|wandb"):
        main()


def test_liom_train_model_missing_dataset_exits_2(
    tmp_path, fake_torch, fake_wandb, monkeypatch, capsys
) -> None:
    """liom-train-model exits 2 with a clear message when dataset_file does not exist.

    A nonexistent dataset_file path must surface as ``parser.error`` (exit 2)
    with the offending path in the message, instead of a raw zarr/torch
    traceback from inside ``train_model``. The file-existence check runs
    BEFORE the import torch lazy-import guard, so the heavy-dep import is
    never attempted for a bad path.
    """
    missing = str(tmp_path / "nope.zarr")
    monkeypatch.setattr(
        sys,
        "argv",
        ["liom-train-model", missing, "node_name"],
    )

    import pytest

    from liom_toolkit.scripts.liom_train_model import main

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "does not exist" in captured.err
    assert missing in captured.err


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
    dataset_file = tmp_path / "dataset.zarr"
    dataset_file.touch()  # materialize so the D-01 file-existence check passes
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "liom-train-model",
            str(dataset_file),
            "node_name",
            "--output-train",
            str(tmp_path / "training"),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--learning-rate",
            "0.001",
            "--wandb-mode",
            "offline",
        ],
    )

    from liom_toolkit.scripts.liom_train_model import main

    with patch("liom_toolkit.segmentation.vseg.training.train_model") as spy:
        main()

    assert spy.called, "main() did not call train_model -- the domain callee was not reached"
    kwargs = spy.call_args.kwargs
    assert kwargs["dataset_file"] == str(dataset_file)
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
