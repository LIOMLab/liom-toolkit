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
    with pytest.raises(ValueError, match=r"patch-level i\.i\.d\. split is rejected"):
        per_volume_split(brain_paths, ["A"], ["B"], patch_level=True)


def test_per_volume_split_rejects_flat_patch_list() -> None:
    """per_volume_split raises ValueError when a flat patch list (not brain-keyed) is passed.

    A flat list of patch paths (instead of a brain→slices dict) is the
    patch-level i.i.d. config — the enforcer rejects it explicitly.
    """
    from liom_toolkit.segmentation.vseg.benchmark.split import per_volume_split

    flat_patches = ["patch_0.png", "patch_1.png", "patch_2.png"]
    with pytest.raises(ValueError, match=r"patch-level i\.i\.d\. split is rejected"):
        per_volume_split(flat_patches, ["patch_0.png"], ["patch_1.png"])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Contender Protocol contract (torch-gated — importorskip at first line of body)
# ---------------------------------------------------------------------------


def test_contender_protocol_is_runtime_checkable() -> None:
    """Contender is a runtime_checkable Protocol with the required attributes."""
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
    """The 3 wired contenders satisfy the Protocol and have the expected names.

    The contenders are now wired (MONAI UNet, SwinUNETR, nnU-Net subprocess
    bridge); this test confirms they still satisfy the Contender Protocol
    structurally and expose the expected ``name`` attributes. The
    ``train_and_predict`` behavior is exercised by the full 4-contender
    benchmark test below.
    """
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.benchmark.contenders import (
        Contender,
        MonaiUnetContender,
        NnUnetContender,
        SwinUnetContender,
    )

    assert isinstance(MonaiUnetContender(), Contender)
    assert isinstance(SwinUnetContender(), Contender)
    assert isinstance(NnUnetContender(), Contender)


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
    pred_uint8 = gt.astype(np.uint8) * 255

    output_dir = str(tmp_path / "train_out")

    fake_model = MagicMock()
    fake_model.eval.return_value = fake_model
    fake_model.to.return_value = fake_model
    fake_model.state_dict.return_value = {}

    with (
        patch("liom_toolkit.segmentation.vseg.model.VsegModel", return_value=fake_model),
        patch("liom_toolkit.segmentation.vseg.training.train_model"),
        patch("liom_toolkit.segmentation.vseg.prediction.predict_one", return_value=pred_uint8),
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
    pred_uint8 = gt.astype(np.uint8) * 255  # perfect prediction

    fake_model = MagicMock()
    fake_model.eval.return_value = fake_model
    fake_model.to.return_value = fake_model
    fake_model.state_dict.return_value = {}

    with (
        patch("liom_toolkit.segmentation.vseg.model.VsegModel", return_value=fake_model),
        patch("liom_toolkit.segmentation.vseg.training.train_model"),
        patch("liom_toolkit.segmentation.vseg.prediction.predict_one", return_value=pred_uint8),
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


# ---------------------------------------------------------------------------
# nnU-Net subprocess bridge (NO torch required — pure subprocess + path check)
# ---------------------------------------------------------------------------


def test_nnunet_bridge_validates_input_path(tmp_path) -> None:
    """nnunet_predict raises ValueError when input_folder does not exist.

    The bridge validates the input folder before invoking the subprocess so a
    typo'd path surfaces as a clear ValueError (with the offending path) rather
    than a cryptic nnU-Net CLI traceback.
    """
    from liom_toolkit.segmentation.vseg.benchmark.nnunet_bridge import nnunet_predict

    nonexistent = str(tmp_path / "does_not_exist")
    with pytest.raises(ValueError, match="input_folder does not exist"):
        nnunet_predict(
            input_folder=nonexistent,
            output_folder=str(tmp_path / "out"),
            dataset_id=999,
        )


def test_nnunet_bridge_raises_on_nonzero_exit(tmp_path, monkeypatch) -> None:
    """nnunet_predict raises RuntimeError when the subprocess exits non-zero.

    No silent pass on a failed nnU-Net run (AGENTS §2): the returncode and the
    tail of stderr are surfaced in the RuntimeError message so the failure is
    actionable. The subprocess is mocked so no real nnU-Net invocation happens.
    """
    from subprocess import CompletedProcess

    from liom_toolkit.segmentation.vseg.benchmark import nnunet_bridge

    input_folder = tmp_path / "imgs"
    input_folder.mkdir()
    fake_proc = CompletedProcess(
        args=["python", "-m", "nnunetv2", "nnUNetv2_predict"],
        returncode=1,
        stdout=b"",
        stderr=b"some nnunet error detail",
    )
    monkeypatch.setattr(nnunet_bridge.subprocess, "run", lambda *a, **k: fake_proc)
    monkeypatch.setenv("nnUNet_raw", str(tmp_path / "raw"))
    monkeypatch.setenv("nnUNet_preprocessed", str(tmp_path / "pre"))
    monkeypatch.setenv("nnUNet_results", str(tmp_path / "res"))

    with pytest.raises(RuntimeError, match="nnUNetv2_predict exited 1"):
        nnunet_bridge.nnunet_predict(
            input_folder=str(input_folder),
            output_folder=str(tmp_path / "out"),
            dataset_id=999,
        )


def test_nnunet_bridge_raises_on_missing_env_vars(tmp_path, monkeypatch) -> None:
    """nnunet_predict raises RuntimeError when nnUNet_* env vars are unset.

    The nnU-Net CLI requires nnUNet_raw/preprocessed/results to locate its
    datasets and plans. Missing any of them is a misconfiguration, not a
    recoverable state — the bridge raises RuntimeError naming the requirement
    instead of silently passing an empty env to the subprocess.
    """
    from liom_toolkit.segmentation.vseg.benchmark.nnunet_bridge import nnunet_predict

    input_folder = tmp_path / "imgs"
    input_folder.mkdir()
    monkeypatch.delenv("nnUNet_raw", raising=False)
    monkeypatch.delenv("nnUNet_preprocessed", raising=False)
    monkeypatch.delenv("nnUNet_results", raising=False)

    with pytest.raises(RuntimeError, match="nnUNet_raw/preprocessed/results"):
        nnunet_predict(
            input_folder=str(input_folder),
            output_folder=str(tmp_path / "out"),
            dataset_id=999,
        )


def test_nnunet_bridge_does_not_import_nnunetv2(tmp_path, monkeypatch) -> None:
    """nnunet_predict never imports nnunetv2 (torch-clobbering isolation).

    nnU-Net v2 pins its own torch/CUDA build that conflicts with the
    liom-toolkit [ai] extra's torch. The bridge runs nnU-Net as a subprocess
    in a separate venv so the liom-toolkit process never imports nnunetv2.
    Verified by checking sys.modules does not contain "nnunetv2" after the
    (mocked) call.
    """
    import sys
    from subprocess import CompletedProcess

    from liom_toolkit.segmentation.vseg.benchmark import nnunet_bridge

    sys.modules.pop("nnunetv2", None)
    input_folder = tmp_path / "imgs"
    input_folder.mkdir()
    fake_proc = CompletedProcess(
        args=["python", "-m", "nnunetv2", "nnUNetv2_predict"],
        returncode=0,
        stdout=b"",
        stderr=b"",
    )
    monkeypatch.setattr(nnunet_bridge.subprocess, "run", lambda *a, **k: fake_proc)
    monkeypatch.setenv("nnUNet_raw", str(tmp_path / "raw"))
    monkeypatch.setenv("nnUNet_preprocessed", str(tmp_path / "pre"))
    monkeypatch.setenv("nnUNet_results", str(tmp_path / "res"))

    nnunet_bridge.nnunet_predict(
        input_folder=str(input_folder),
        output_folder=str(tmp_path / "out"),
        dataset_id=999,
    )
    assert "nnunetv2" not in sys.modules, (
        "nnunet_bridge must NEVER import nnunetv2 (torch-clobbering isolation)"
    )


def test_nnunet_bridge_uses_list_argv_no_shell(tmp_path, monkeypatch) -> None:
    """nnunet_predict invokes subprocess.run with a list argv and no shell=True.

    The subprocess-injection mitigation: a list argv (no shell=True) means
    user-supplied paths cannot break out of the argv into a separate shell
    command. Captured via the call args passed to the mocked subprocess.run.
    """
    from subprocess import CompletedProcess

    from liom_toolkit.segmentation.vseg.benchmark import nnunet_bridge

    input_folder = tmp_path / "imgs"
    input_folder.mkdir()
    captured: dict = {}
    fake_proc = CompletedProcess(args=[], returncode=0, stdout=b"", stderr=b"")

    def _capture(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return fake_proc

    monkeypatch.setattr(nnunet_bridge.subprocess, "run", _capture)
    monkeypatch.setenv("nnUNet_raw", str(tmp_path / "raw"))
    monkeypatch.setenv("nnUNet_preprocessed", str(tmp_path / "pre"))
    monkeypatch.setenv("nnUNet_results", str(tmp_path / "res"))

    nnunet_bridge.nnunet_predict(
        input_folder=str(input_folder),
        output_folder=str(tmp_path / "out"),
        dataset_id=999,
    )
    assert isinstance(captured["cmd"], list), "argv must be a list (no shell=True)"
    assert captured["kwargs"].get("shell") is not True, "shell=True is forbidden"


# ---------------------------------------------------------------------------
# Full 4-contender benchmark end-to-end on synthetic data (Task 3)
# ---------------------------------------------------------------------------


def test_run_benchmark_full_4_contenders(tmp_path, monkeypatch) -> None:
    """run_benchmark scores all 4 contenders end-to-end on synthetic data.

    Constructs synthetic 2D vessel-mask PNG slices, calls ``run_benchmark``
    with all 4 contenders (Improved2D, MonaiUNet, SwinUNETR, NnUNet), with
    mocked ``train_model`` (no-op) + mocked ``subprocess.run`` for the nnU-Net
    bridge (writes fake prediction PNGs). Asserts the returned metric table
    has 4 contender keys each with the 6 gate-metric keys + reported_dice.

    The MONAI contenders actually build real MONAI models (UNet / SwinUNETR)
    and run real SlidingWindowInferer inference on the synthetic slices —
    the models are randomly initialized (train_model is mocked), so the
    predictions are random, but the test only asserts the result-table
    structure, not specific metric values.
    """
    pytest.importorskip("torch")
    pytest.importorskip("monai")
    from subprocess import CompletedProcess

    import imageio.v3 as iio
    import torch

    from liom_toolkit.segmentation.vseg.benchmark.contenders import (
        Improved2DContender,
        MonaiUnetContender,
        NnUnetContender,
        SwinUnetContender,
    )
    from liom_toolkit.segmentation.vseg.benchmark.run import run_benchmark

    # Synthetic 64×64 slices (divisible by 32 for SwinUNETR) with a horizontal
    # vessel. Written as real PNGs so the MONAI contenders can read them.
    shape = (64, 64)
    gt = np.zeros(shape, dtype=bool)
    gt[28:36, 16:48] = True
    gt_uint8 = gt.astype(np.uint8) * 255

    slice_dir = tmp_path / "slices"
    slice_dir.mkdir()
    test_slices: list[str] = []
    for i in range(2):
        p = slice_dir / f"test_{i:02d}.png"
        iio.imwrite(p, gt_uint8)
        test_slices.append(str(p))
    # Train slices (also real PNGs for the nnU-Net converter).
    train_slices: list[str] = []
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    for i in range(2):
        p = train_dir / f"train_{i:02d}.png"
        iio.imwrite(p, gt_uint8)
        train_slices.append(str(p))
        # Matching _mask.png labels for the nnU-Net converter.
        iio.imwrite(train_dir / f"train_{i:02d}_mask.png", gt.astype(np.uint8))

    # Mock train_model (no-op) for all contenders.
    monkeypatch.setattr("liom_toolkit.segmentation.vseg.training.train_model", lambda *a, **k: None)
    # Mock predict_one + VsegModel for Improved2DContender (returns perfect
    # predictions — the same pattern as the tracer slice test).
    fake_model = MagicMock()
    fake_model.eval.return_value = fake_model
    fake_model.to.return_value = fake_model
    fake_model.state_dict.return_value = {}
    mock_vseg_cls = MagicMock(return_value=fake_model)
    monkeypatch.setattr("liom_toolkit.segmentation.vseg.model.VsegModel", mock_vseg_cls)
    mock_predict_one = MagicMock(return_value=gt_uint8)
    monkeypatch.setattr("liom_toolkit.segmentation.vseg.prediction.predict_one", mock_predict_one)
    monkeypatch.setattr("liom_toolkit.segmentation.vseg.benchmark.contenders.torch", torch)

    # Mock subprocess.run for the nnU-Net bridge: when nnUNetv2_predict is
    # invoked, write fake prediction PNGs (matching the test slices) to the
    # output folder so NnUnetContender can read them back.
    def _fake_subprocess_run(cmd, **kwargs):
        # Parse -o <output_folder> from the argv.
        out_folder = cmd[cmd.index("-o") + 1]
        Path(out_folder).mkdir(parents=True, exist_ok=True)
        for i in range(len(test_slices)):
            iio.imwrite(Path(out_folder) / f"case_{i:04d}.png", gt_uint8)
        return CompletedProcess(args=cmd, returncode=0, stdout=b"", stderr=b"")

    from liom_toolkit.segmentation.vseg.benchmark import nnunet_bridge

    monkeypatch.setattr(nnunet_bridge.subprocess, "run", _fake_subprocess_run)
    monkeypatch.setenv("nnUNet_raw", str(tmp_path / "raw"))
    monkeypatch.setenv("nnUNet_preprocessed", str(tmp_path / "pre"))
    monkeypatch.setenv("nnUNet_results", str(tmp_path / "res"))

    contenders = [
        Improved2DContender(device="cpu"),
        MonaiUnetContender(device="cpu"),
        SwinUnetContender(device="cpu"),
        NnUnetContender(device="cpu", dataset_id=999),
    ]

    results = run_benchmark(
        contenders=contenders,
        split_config={
            "train_slices": train_slices,
            "test_slices": test_slices,
            "gt_masks": [gt, gt],
            "patch_size": (1, 64, 64),
            "ddp": False,
        },
        eval_config=None,
        output_dir=str(tmp_path / "bench_out"),
    )

    expected_names = {"improved_2d", "monai_unet", "monai_swinunetr", "nnunet_v2"}
    assert set(results.keys()) == expected_names, (
        f"result table must have all 4 contender keys; got {set(results.keys())}"
    )
    expected_metrics = {
        "centerline_recall",
        "caliber_stratified_recall",
        "boundary_artifact_regression",
        "spurious_thin_vessel_rate",
        "fpr_on_empty",
        "cl_dice_metric",
        "reported_dice",
    }
    for name in expected_names:
        row = results[name]
        assert expected_metrics.issubset(row.keys()), (
            f"contender {name} row must contain all 6 gate metrics + reported_dice; "
            f"got {set(row.keys())}"
        )
