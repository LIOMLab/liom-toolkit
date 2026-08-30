"""Tests for the vessel-segmentation benchmark harness.

Covers three concerns:

* **Contender Protocol contract** — the ``Contender`` Protocol is
  ``runtime_checkable``; each of the four contender classes
  (``Improved2DContender``, ``MonaiUnetContender``, ``SwinUnetContender``,
  ``NnUnetContender``) exposes ``name``, ``train_and_predict`` and
  ``predict_on_slices``; the three non-tracer contenders raise
  ``NotImplementedError`` from ``train_and_predict`` (their wiring lands in a
  later plan after the MONAI dependency is added).
* **Per-volume split enforcement** — ``per_volume_split`` partitions at the
  brain (volume) level, raises ``ValueError`` if a brain appears in both
  train and test (no silent vascular-structure leak), and raises
  ``ValueError`` if a patch-level split config is passed (patch-level i.i.d.
  leaks vascular structure across train/test and inflates Dice 10-20+ points).
* **Tracer slice** — ``Improved2DContender.train_and_predict`` returns binary
  masks on synthetic 2D slices, and ``run_benchmark`` scores the contender
  end-to-end through the ship-gate eval-metric matrix.

The split tests require NO torch (``split.py`` is pure data partitioning).
The contender + tracer tests gate on ``pytest.importorskip("torch")`` at the
FIRST line of the test body (never at module top — pytest #9542 would skip
the whole module including the torch-free split tests).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Per-volume split (NO torch required — split.py is pure data partitioning)
# ---------------------------------------------------------------------------


def test_per_volume_split_returns_brain_partitions() -> None:
    """per_volume_split returns flat slice lists partitioned at the brain level.

    Brains A and B go to train; brain C goes to test. The returned
    train_slices/test_slices are the concatenation of each brain's slice
    paths, in the order the brains are listed.
    """
    from liom_toolkit.segmentation.vseg.benchmark.split import per_volume_split

    brain_paths = {
        "A": ["a_0.png", "a_1.png"],
        "B": ["b_0.png"],
        "C": ["c_0.png", "c_1.png"],
    }
    train_slices, test_slices = per_volume_split(
        brain_paths, train_brains=["A", "B"], test_brains=["C"]
    )
    assert train_slices == ["a_0.png", "a_1.png", "b_0.png"]
    assert test_slices == ["c_0.png", "c_1.png"]


def test_per_volume_split_raises_on_brain_overlap() -> None:
    """per_volume_split raises ValueError if a brain is in both train and test.

    A brain in both splits leaks vascular structure across train/test and
    inflates Dice 10-20+ points — the silent-wrong-data failure mode this
    enforcer exists to prevent.
    """
    from liom_toolkit.segmentation.vseg.benchmark.split import per_volume_split

    brain_paths = {"A": ["a_0.png"], "B": ["b_0.png"]}
    with pytest.raises(ValueError, match="appears in both train and test"):
        per_volume_split(brain_paths, train_brains=["A"], test_brains=["A", "B"])


def test_per_volume_split_rejects_patch_level_kwarg() -> None:
    """per_volume_split raises ValueError when patch_level=True is passed.

    Patch-level i.i.d. splitting is explicitly rejected — it leaks vascular
    structure across train/test and inflates Dice 10-20+ points. The
    rejection string documents the rationale in domain terms so no future
    contributor reintroduces the pitfall.
    """
    from liom_toolkit.segmentation.vseg.benchmark.split import per_volume_split

    brain_paths = {"A": ["a_0.png"], "B": ["b_0.png"]}
    with pytest.raises(ValueError, match="patch-level i.i.d. split is rejected"):
        per_volume_split(brain_paths, ["A"], ["B"], patch_level=True)


def test_per_volume_split_rejects_flat_patch_list() -> None:
    """per_volume_split raises ValueError when a flat patch list (not brain-keyed) is passed.

    A flat list of patch paths (instead of a brain→slices dict) is the
    patch-level i.i.d. config — the enforcer rejects it explicitly.
    """
    from liom_toolkit.segmentation.vseg.benchmark.split import per_volume_split

    flat_patches = ["patch_0.png", "patch_1.png", "patch_2.png"]
    with pytest.raises(ValueError, match="patch-level i.i.d. split is rejected"):
        per_volume_split(flat_patches, ["patch_0.png"], ["patch_1.png"])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Contender Protocol contract (torch-gated — importorskip at first line of body)
# ---------------------------------------------------------------------------


def test_contender_protocol_is_runtime_checkable() -> None:
    """Contender is a runtime_checkable Protocol with name + train_and_predict + predict_on_slices."""
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.benchmark.contenders import Contender

    assert hasattr(Contender, "_is_protocol"), "Contender must be a Protocol"
    assert hasattr(Contender, "_is_runtime_protocol"), "Contender must be runtime_checkable"


def test_improved_2d_contender_satisfies_protocol() -> None:
    """isinstance(Improved2DContender(), Contender) is True (runtime_checkable)."""
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.benchmark.contenders import (
        Contender,
        Improved2DContender,
    )

    contender = Improved2DContender()
    assert isinstance(contender, Contender), (
        "Improved2DContender must satisfy the Contender Protocol structurally"
    )
    assert contender.name == "improved_2d"


def test_skeletal_contender_names() -> None:
    """The 3 skeletal contenders have the expected names and satisfy the Protocol structurally."""
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.benchmark.contenders import (
        Contender,
        MonaiUnetContender,
        NnUnetContender,
        SwinUnetContender,
    )

    assert MonaiUnetContender().name == "monai_unet"
    assert SwinUnetContender().name == "monai_swinunetr"
    assert NnUnetContender().name == "nnunet_v2"
    assert isinstance(MonaiUnetContender(), Contender)
    assert isinstance(SwinUnetContender(), Contender)
    assert isinstance(NnUnetContender(), Contender)


def test_skeletal_contenders_raise_not_implemented() -> None:
    """The 3 skeletal contenders raise NotImplementedError from train_and_predict + predict_on_slices.

    Their wiring lands in a later plan after the MONAI dependency is added;
    the classes exist and satisfy the Protocol structurally so the harness
    can enumerate them, but they cannot train yet.
    """
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.benchmark.contenders import (
        MonaiUnetContender,
        NnUnetContender,
        SwinUnetContender,
    )

    for cls in (MonaiUnetContender, SwinUnetContender, NnUnetContender):
        contender = cls()
        with pytest.raises(NotImplementedError, match="wiring lands in"):
            contender.train_and_predict([], [], ".", patch_size=(1, 64, 64), ddp=False)
        with pytest.raises(NotImplementedError, match="wiring lands in"):
            contender.predict_on_slices([], "fake.ckpt")


def test_improved_2d_train_and_predict_returns_binary_masks(tmp_path) -> None:
    """Improved2DContender.train_and_predict returns a list of boolean NDArrays.

    Mocks the heavy orchestration (train_model, predict_one, VsegModel) but
    NOT the compute deps (numpy). predict_one's 0/255 uint8 output is
    binarized to bool so the eval-metric matrix receives NDArray[np.bool_].
    """
    pytest.importorskip("torch")
    import torch

    from liom_toolkit.segmentation.vseg.benchmark.contenders import Improved2DContender

    gt = np.zeros((64, 64), dtype=bool)
    gt[28:36, 16:48] = True  # a horizontal vessel
    pred_uint8 = (gt.astype(np.uint8) * 255)

    output_dir = str(tmp_path / "train_out")

    fake_model = MagicMock()
    fake_model.eval.return_value = fake_model
    fake_model.to.return_value = fake_model
    fake_model.state_dict.return_value = {}

    with (
        patch("liom_toolkit.segmentation.vseg.benchmark.contenders.VsegModel", return_value=fake_model),
        patch("liom_toolkit.segmentation.vseg.benchmark.contenders.train_model"),
        patch("liom_toolkit.segmentation.vseg.benchmark.contenders.predict_one", return_value=pred_uint8),
        patch("liom_toolkit.segmentation.vseg.benchmark.contenders.torch", torch),
        patch.object(Path, "exists", return_value=False),
    ):
        contender = Improved2DContender(device="cpu")
        masks = contender.train_and_predict(
            train_slices=["train.zarr"],
            test_slices=["test_0.png", "test_1.png"],
            output_dir=output_dir,
            patch_size=(1, 64, 64),
            ddp=False,
        )

    assert len(masks) == 2
    for m in masks:
        assert m.dtype == np.bool_, f"train_and_predict must return bool masks, got {m.dtype}"
        assert m.shape == (64, 64)


# ---------------------------------------------------------------------------
# Tracer slice — run_benchmark scores Improved2D end-to-end (Task 2)
# ---------------------------------------------------------------------------


def test_run_benchmark_tracer_slice(tmp_path) -> None:
    """run_benchmark scores Improved2DContender end-to-end on synthetic data.

    The thinnest slice that proves the harness works: the ship-gate metric
    matrix from the eval-metrics module computes over the contender's
    binarized predictions. Mocks train_model + predict_one + VsegModel
    (orchestration), NOT the eval-metric compute (numpy/scipy/skimage).
    """
    pytest.importorskip("torch")
    import torch

    from liom_toolkit.segmentation.vseg.benchmark.contenders import Improved2DContender
    from liom_toolkit.segmentation.vseg.benchmark.run import run_benchmark

    gt = np.zeros((64, 64), dtype=bool)
    gt[28:36, 16:48] = True  # a horizontal vessel
    pred_uint8 = (gt.astype(np.uint8) * 255)  # perfect prediction

    fake_model = MagicMock()
    fake_model.eval.return_value = fake_model
    fake_model.to.return_value = fake_model
    fake_model.state_dict.return_value = {}

    with (
        patch("liom_toolkit.segmentation.vseg.benchmark.contenders.VsegModel", return_value=fake_model),
        patch("liom_toolkit.segmentation.vseg.benchmark.contenders.train_model"),
        patch("liom_toolkit.segmentation.vseg.benchmark.contenders.predict_one", return_value=pred_uint8),
        patch("liom_toolkit.segmentation.vseg.benchmark.contenders.torch", torch),
        patch.object(Path, "exists", return_value=False),
    ):
        results = run_benchmark(
            contenders=[Improved2DContender(device="cpu")],
            split_config={
                "train_slices": ["train.zarr"],
                "test_slices": ["test_0.png", "test_1.png"],
                "gt_masks": [gt, gt],
                "patch_size": (1, 64, 64),
                "ddp": False,
            },
            eval_config=None,
            output_dir=str(tmp_path / "bench_out"),
        )

    assert "improved_2d" in results, "result table must be keyed by contender name"
    row = results["improved_2d"]
    expected_keys = {
        "centerline_recall",
        "caliber_stratified_recall",
        "boundary_artifact_regression",
        "spurious_thin_vessel_rate",
        "fpr_on_empty",
        "cl_dice_metric",
        "reported_dice",
    }
    assert expected_keys.issubset(row.keys()), (
        f"result row must contain all 6 gate metrics + reported_dice; got {set(row.keys())}"
    )
    # centerline_recall on a perfect prediction is 1.0.
    assert row["centerline_recall"] == pytest.approx(1.0, abs=1e-6)
    assert row["reported_dice"] == pytest.approx(1.0, abs=1e-6)
