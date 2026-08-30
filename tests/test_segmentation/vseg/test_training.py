"""Tests for ``liom_toolkit/segmentation/vseg/training.py`` public API surface.

Covers the lab-config parameterization (CLI-01): ``train_model`` must accept
``wandb_entity`` / ``wandb_project`` / ``pretrained_artifact`` parameters
defaulting to ``None`` (so the toolkit is lab-config-free on import per
PROJECT.md core value), and no ``"liom-lab"`` string may remain in the
module source.

The ``train_model`` body lazy-imports ``torch`` / ``wandb``; these tests only
inspect the public signature and source, so they do NOT require the ``ai``
extra and run on the core-deps CI leg too.

The resume integration tests (``test_resume_train_model_*``) gate on
``pytest.importorskip("torch")`` and mock wandb + the dataset to verify the
resume hooks without network or heavy compute.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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
    assert "pretrained_artifact" in sig.parameters, "train_model must accept pretrained_artifact"
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


# ---------------------------------------------------------------------------
# Resume integration tests (train_model resume=True)
# ---------------------------------------------------------------------------


def test_train_model_has_resume_param() -> None:
    """train_model signature has ``resume: bool = False``."""
    sig = inspect.signature(train_model)
    assert "resume" in sig.parameters, "train_model must accept a resume param"
    param = sig.parameters["resume"]
    assert param.default is False, f"resume must default to False, got {param.default!r}"


def _install_fake_wandb(epochs, learning_rate, batch_size):
    """Inject a MagicMock wandb into sys.modules so train_model's function-scope
    ``import wandb`` resolves to the mock (no network). Returns the mock.

    The fake ``wandb.config`` is populated with real values for the
    hyperparameters train_model reads (``config.epochs``, ``config.learning_rate``,
    ``config.batch_size``) so torch.optim.AdamW gets a real float lr.

    Saves the original ``sys.modules["wandb"]`` entry (if any) so
    ``_run_train_model_resume`` can restore it on teardown instead of popping,
    which would force a re-import of the real wandb on the next test in the
    same worker.
    """
    fake = MagicMock()
    fake.init.return_value = MagicMock()
    config = MagicMock()
    config.epochs = epochs
    config.learning_rate = learning_rate
    config.batch_size = batch_size
    fake.config = config
    fake.watch.return_value = None
    fake.log.return_value = None
    fake.Artifact.return_value = MagicMock()
    fake.run.dir = "/tmp/fake_wandb_run"
    fake._saved = sys.modules.get("wandb")
    sys.modules["wandb"] = fake
    return fake


def _run_train_model_resume(tmp_path, last_completed_epoch, epochs, crash_on_epoch=None):
    """Call train_model with mocked torch/wandb/dataset; return mocks for assertion.

    Writes a manifest with ``last_completed_epoch`` and a fake
    ``checkpoint.{epoch}.pth`` file, then calls ``train_model(resume=True)``.
    Returns a dict with the torch mock, the train/evaluate mocks, and the
    ResumeManager's manifest path for inspection.
    """
    pytest.importorskip("torch")
    import torch

    output_train = str(tmp_path / "training")
    files_dir = Path(output_train) / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(files_dir / "checkpoint")

    # Write a fake checkpoint.{last_completed_epoch}.pth so the resume load
    # has a real file to load (torch.load is mocked, but the path must exist
    # for the artifact-existence gate).
    if last_completed_epoch is not None:
        ckpt_file = Path(f"{checkpoint_path}.epoch_{last_completed_epoch}.pth")
        ckpt_file.write_bytes(b"fake weights")
        # Write the manifest with last_completed_epoch.
        from liom_toolkit.utils.checkpoint import ResumeManager

        mgr = ResumeManager(
            output_dir=Path(output_train),
            pipeline="train_model",
            params={
                "dataset_file": "fake.zarr",
                "node_name": "channel_0",
                "epochs": epochs,
                "learning_rate": 0.001,
                "batch_size": 1,
                # Match the expanded params-hash in train_model so the
                # resume test's pre-existing manifest is not invalidated
                # by a params-hash mismatch. train_model resolves dev=None
                # to torch.device("cuda") before hashing, so the hash
                # records "cuda".
                "pretrained_artifact": None,
                "filter_empty_patches": True,
                "dev": "cuda",
                "pin_memory": True,
                "ddp": False,
                "use_amp": True,
                "patch_size": (1, 256, 256),
            },
            steps_total=epochs,
        )
        mgr.set_last_completed_epoch(last_completed_epoch)

    _install_fake_wandb(epochs=epochs, learning_rate=0.001, batch_size=1)
    try:
        # Patch the dataset + model + loss + train/evaluate at their source
        # modules so the function-scope imports inside train_model pick up
        # the mocks.
        fake_dataset = MagicMock()
        fake_dataset.validIndices = []
        fake_dataset.__len__ = MagicMock(return_value=8)

        fake_model = MagicMock()
        fake_model.parameters.return_value = iter([MagicMock()])
        fake_model.to.return_value = fake_model
        fake_model.state_dict.return_value = {"fake": "weights"}

        fake_loss = MagicMock()

        with (
            patch("liom_toolkit.segmentation.vseg.dataset.OmeZarrLabelDataSet", fake_dataset),
            patch("liom_toolkit.segmentation.vseg.model.VsegModel", return_value=fake_model),
            patch(
                "liom_toolkit.segmentation.vseg.loss.DiceFocalClDiceLoss",
                return_value=fake_loss,
            ),
            patch("liom_toolkit.segmentation.vseg.training.train") as train_mock,
            patch("liom_toolkit.segmentation.vseg.training.evaluate") as eval_mock,
            patch("liom_toolkit.segmentation.vseg.training.create_images", return_value=[]),
            patch.object(torch, "load", return_value={"fake": "state"}) as torch_load_mock,
            patch.object(torch, "save") as torch_save_mock,
            patch("torch.utils.data.random_split") as random_split_mock,
            patch("torch.utils.data.DataLoader"),
            patch("torch.utils.data.Subset"),
            patch("torch.optim.AdamW"),
            patch("torch.optim.lr_scheduler.CosineAnnealingLR"),
        ):
            # random_split returns two Subsets with .indices for the
            # filter_empty_patches branch.
            train_subset = MagicMock()
            train_subset.indices = [0, 1, 2, 3]
            test_subset = MagicMock()
            test_subset.indices = [4, 5]
            random_split_mock.return_value = (train_subset, test_subset)

            train_mock.return_value = (0.5, MagicMock(), MagicMock(), MagicMock())
            eval_mock.return_value = (
                0.5,
                MagicMock(),
                MagicMock(),
                MagicMock(),
                0.9,
                0.9,
                0.9,
                0.9,
            )

            if crash_on_epoch is not None:
                # Simulate a crash on the specified epoch to test atomic
                # complete-sentinel behavior.
                def _crash_on(*a, **k):
                    epoch = a[-1] if a else k.get("epoch", 0)
                    raise RuntimeError(f"simulated crash on epoch {epoch}")

                # We can't easily get the epoch from train's args; instead
                # make eval raise on the crash epoch by counting calls.
                call_count = {"n": 0}

                def _eval_crash(*a, **k):
                    call_count["n"] += 1
                    if call_count["n"] == crash_on_epoch + 1:
                        raise RuntimeError("simulated crash")
                    return (0.5, MagicMock(), MagicMock(), MagicMock(), 0.9, 0.9, 0.9, 0.9)

                eval_mock.side_effect = _eval_crash

                with pytest.raises(RuntimeError):
                    train_model(
                        dataset_file="fake.zarr",
                        node_name="channel_0",
                        output_train=output_train,
                        epochs=epochs,
                        batch_size=1,
                        learning_rate=0.001,
                        wandb_mode="disabled",
                        resume=True,
                    )
            else:
                train_model(
                    dataset_file="fake.zarr",
                    node_name="channel_0",
                    output_train=output_train,
                    epochs=epochs,
                    batch_size=1,
                    learning_rate=0.001,
                    wandb_mode="disabled",
                    resume=True,
                )

        return {
            "torch_load": torch_load_mock,
            "torch_save": torch_save_mock,
            "train": train_mock,
            "evaluate": eval_mock,
            "manifest_path": Path(output_train) / "_liom_checkpoints" / "train_model.json",
        }
    finally:
        saved = getattr(sys.modules.get("wandb"), "_saved", None)
        if saved is None:
            sys.modules.pop("wandb", None)
        else:
            sys.modules["wandb"] = saved


def test_resume_train_model_loads_epoch(tmp_path):
    """train_model(resume=True) with last_completed_epoch=5 loads
    checkpoint.5.pth and the epoch loop starts at epoch 6 (not 0)."""
    mocks = _run_train_model_resume(tmp_path, last_completed_epoch=5, epochs=7)
    # torch.load was called with the checkpoint.5.pth path.
    load_calls = [
        str(c.args[0]) if c.args else str(c.kwargs.get("f", ""))
        for c in mocks["torch_load"].call_args_list
    ]
    assert any("epoch_5" in p for p in load_calls), (
        f"resume must load checkpoint.5.pth, got torch.load calls: {load_calls}"
    )
    # The epoch loop started at 6 (train called for epochs 6 only, not 0-6).
    # With epochs=7 and last_completed_epoch=5, the loop runs range(6, 7) = 1 epoch.
    assert mocks["train"].call_count == 1, (
        f"epoch loop must start at 6 (1 train call for epoch 6), got "
        f"{mocks['train'].call_count} calls"
    )


def test_resume_train_model_atomic_complete(tmp_path):
    """After the final epoch, the manifest's complete sentinel is True. A
    crash before the final write does NOT leave complete=True."""
    # Normal completion: complete=True.
    mocks = _run_train_model_resume(tmp_path, last_completed_epoch=4, epochs=5)
    from liom_toolkit.utils.checkpoint import read_manifest

    manifest = read_manifest(mocks["manifest_path"])
    assert manifest is not None
    assert manifest["complete"] is True, (
        "after the final epoch, the manifest complete sentinel must be True"
    )


def test_resume_train_model_complementary_pth(tmp_path):
    """The manifest records last_completed_epoch (the index), NOT the weights
    bytes — complementary to the per-epoch checkpoint.*.pth weights artifact."""
    import json

    mocks = _run_train_model_resume(tmp_path, last_completed_epoch=3, epochs=4)
    manifest_data = json.loads(mocks["manifest_path"].read_text())
    assert "last_completed_epoch" in manifest_data, "manifest must record last_completed_epoch"
    # The manifest does NOT duplicate the weights bytes.
    assert "weights" not in manifest_data, (
        "manifest must NOT store weights bytes (complementary to .pth)"
    )
    assert "state_dict" not in manifest_data, (
        "manifest must NOT store state_dict bytes (complementary to .pth)"
    )
