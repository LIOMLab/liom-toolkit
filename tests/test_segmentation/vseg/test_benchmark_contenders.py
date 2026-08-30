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
    assert NnUnetContender(nnunet_venv_python="/usr/bin/python3").name == "nnunet_v2"
    assert isinstance(MonaiUnetContender(), Contender)
    assert isinstance(SwinUnetContender(), Contender)
    assert isinstance(NnUnetContender(nnunet_venv_python="/usr/bin/python3"), Contender)


def test_skeletal_contenders_raise_on_missing_labels() -> None:
    """The 3 wired contenders raise ValueError on missing mask labels.

    The MONAI contenders (MonaiUnetContender, SwinUnetContender) and the
    nnU-Net contender (NnUnetContender) are now wired: they train on real
    PNG slices + their ``<name>_mask.png`` labels. When a train slice has
    no matching label, they raise ValueError (no silent fallback to an
    all-zero mask — AGENTS §2). This test verifies the label-validation
    guard fires before any training/subprocess starts.
    """
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.benchmark.contenders import (
        Contender,
        MonaiUnetContender,
        NnUnetContender,
        SwinUnetContender,
    )

    monai = MonaiUnetContender()
    swin = SwinUnetContender()
    nnunet = NnUnetContender(nnunet_venv_python="/usr/bin/python3")
    assert isinstance(monai, Contender)
    assert isinstance(swin, Contender)
    assert isinstance(nnunet, Contender)
    # A nonexistent slice path → ValueError from the mask-path resolver
    # (no silent zero-mask fallback). The contenders no longer raise
    # NotImplementedError — they are wired for real training.
    with pytest.raises(ValueError, match="no matching label"):
        monai.train_and_predict(["nonexistent.png"], ["y"], "/tmp/out")
    with pytest.raises(ValueError, match="no matching label"):
        swin.train_and_predict(["nonexistent.png"], ["y"], "/tmp/out")
    with pytest.raises(ValueError, match="no matching label"):
        nnunet.train_and_predict(["nonexistent.png"], ["y"], "/tmp/out")


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
        # WR-06: train_and_predict now raises RuntimeError if the checkpoint
        # is missing after train_model. Mock the checkpoint as present and
        # torch.load returning an empty state_dict (loaded into the fake
        # model whose load_state_dict is a MagicMock no-op).
        patch.object(Path, "exists", return_value=True),
        patch.object(torch, "load", return_value={}),
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


def test_improved_2d_train_and_predict_raises_on_missing_checkpoint(tmp_path) -> None:
    """Improved2DContender.train_and_predict raises RuntimeError if checkpoint missing.

    WR-06: after train_model, if the checkpoint is missing (training failed
    silently, disk full, wrong path), the contender must raise RuntimeError
    rather than proceed with a randomly-initialized VsegModel and present
    random predictions as trained-model output (the silent-wrong-data path
    AGENTS §2 forbids).
    """
    pytest.importorskip("torch")
    import torch

    from liom_toolkit.segmentation.vseg.benchmark.contenders import Improved2DContender

    fake_model = MagicMock()
    fake_model.eval.return_value = fake_model
    fake_model.to.return_value = fake_model
    fake_model.state_dict.return_value = {}

    with (
        patch("liom_toolkit.segmentation.vseg.model.VsegModel", return_value=fake_model),
        patch("liom_toolkit.segmentation.vseg.training.train_model"),
        patch("liom_toolkit.segmentation.vseg.benchmark.contenders.torch", torch),
        # Checkpoint missing after train_model → RuntimeError.
        patch.object(Path, "exists", return_value=False),
    ):
        contender = Improved2DContender(device="cpu")
        with pytest.raises(RuntimeError, match="checkpoint not found after train_model"):
            contender.train_and_predict(
                train_slices=["train.zarr"],
                test_slices=["test_0.png"],
                output_dir=str(tmp_path / "out"),
                patch_size=(1, 64, 64),
                ddp=False,
            )


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
        # WR-06: checkpoint present + torch.load returns empty state_dict.
        patch.object(Path, "exists", return_value=True),
        patch.object(torch, "load", return_value={}),
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
            nnunet_venv_python="/usr/bin/python3",
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
        args=["nnUNetv2_predict"],
        returncode=1,
        stdout=b"",
        stderr=b"some nnunet error detail",
    )
    monkeypatch.setattr(nnunet_bridge.subprocess, "run", lambda *a, **k: fake_proc)
    monkeypatch.setattr(
        nnunet_bridge, "_nnunet_console_script", lambda py, name: f"/fake/bin/{name}"
    )
    monkeypatch.setenv("nnUNet_raw", str(tmp_path / "raw"))
    monkeypatch.setenv("nnUNet_preprocessed", str(tmp_path / "pre"))
    monkeypatch.setenv("nnUNet_results", str(tmp_path / "res"))

    with pytest.raises(RuntimeError, match="nnUNetv2_predict exited 1"):
        nnunet_bridge.nnunet_predict(
            input_folder=str(input_folder),
            output_folder=str(tmp_path / "out"),
            dataset_id=999,
            nnunet_venv_python="/usr/bin/python3",
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
            nnunet_venv_python="/usr/bin/python3",
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
        args=["nnUNetv2_predict"],
        returncode=0,
        stdout=b"",
        stderr=b"",
    )
    monkeypatch.setattr(nnunet_bridge.subprocess, "run", lambda *a, **k: fake_proc)
    monkeypatch.setattr(
        nnunet_bridge, "_nnunet_console_script", lambda py, name: f"/fake/bin/{name}"
    )
    monkeypatch.setenv("nnUNet_raw", str(tmp_path / "raw"))
    monkeypatch.setenv("nnUNet_preprocessed", str(tmp_path / "pre"))
    monkeypatch.setenv("nnUNet_results", str(tmp_path / "res"))

    nnunet_bridge.nnunet_predict(
        input_folder=str(input_folder),
        output_folder=str(tmp_path / "out"),
        dataset_id=999,
        nnunet_venv_python="/usr/bin/python3",
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
    monkeypatch.setattr(
        nnunet_bridge, "_nnunet_console_script", lambda py, name: f"/fake/bin/{name}"
    )
    monkeypatch.setenv("nnUNet_raw", str(tmp_path / "raw"))
    monkeypatch.setenv("nnUNet_preprocessed", str(tmp_path / "pre"))
    monkeypatch.setenv("nnUNet_results", str(tmp_path / "res"))

    nnunet_bridge.nnunet_predict(
        input_folder=str(input_folder),
        output_folder=str(tmp_path / "out"),
        dataset_id=999,
        nnunet_venv_python="/usr/bin/python3",
    )
    assert isinstance(captured["cmd"], list), "argv must be a list (no shell=True)"
    assert captured["kwargs"].get("shell") is not True, "shell=True is forbidden"


# ---------------------------------------------------------------------------
# Full 4-contender benchmark end-to-end on synthetic data (Task 3)
# ---------------------------------------------------------------------------


def test_run_benchmark_full_4_contenders(tmp_path, monkeypatch) -> None:
    """run_benchmark scores all 4 contenders end-to-end on synthetic data.

    The MONAI contenders (MonaiUnetContender, SwinUnetContender) train for
    real (1 epoch on tiny 32×32 synthetic slices with matching
    ``<name>_mask.png`` labels) and predict via SlidingWindowInferer. The
    nnU-Net contender's subprocess calls (plan_and_preprocess, train,
    predict) are mocked — nnU-Net runs in a separate venv on the lab box,
    not in CI — and fake prediction PNGs are written to the output folder
    so the read-back path exercises the bool conversion. Improved2DContender
    is mocked as before (train_model + predict_one + VsegModel).

    Asserts all 4 contenders produce boolean masks and score through the
    ship-gate eval-metric matrix via run_benchmark (4-key result table).
    """
    pytest.importorskip("torch")
    pytest.importorskip("monai")
    import imageio.v3 as iio
    import torch

    from liom_toolkit.segmentation.vseg.benchmark.contenders import (
        Improved2DContender,
        MonaiUnetContender,
        NnUnetContender,
        SwinUnetContender,
    )
    from liom_toolkit.segmentation.vseg.benchmark.run import run_benchmark

    # Synthetic 64×64 slices with a horizontal vessel — large enough for
    # SwinUNETR (which downsamples 5 stages: 64→32→16→8→4→2, so the
    # deepest feature map has >1 spatial element for InstanceNorm). The
    # MONAI UNet (3 strided stages: 64→32→16→8) also works at this size.
    shape = (64, 64)
    gt = np.zeros(shape, dtype=bool)
    gt[28:36, 16:48] = True
    gt_uint8 = gt.astype(np.uint8) * 255

    # Create train slices WITH matching _mask.png labels (the MONAI
    # contenders and the nnU-Net contender both require them).
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    train_slices: list[str] = []
    for i in range(4):
        p = train_dir / f"train_{i:02d}.png"
        iio.imwrite(p, gt_uint8)
        iio.imwrite(train_dir / f"train_{i:02d}_mask.png", gt_uint8)
        train_slices.append(str(p))

    # Test slices (no masks needed — the harness reads gt_masks separately).
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    test_slices: list[str] = []
    for i in range(2):
        p = test_dir / f"test_{i:02d}.png"
        iio.imwrite(p, gt_uint8)
        test_slices.append(str(p))

    # --- Mock Improved2DContender (orchestration, not compute) ---
    # train_model is mocked (no-op), so it won't create a checkpoint. Write
    # a real dummy checkpoint at the expected path so the Improved2DContender's
    # load path succeeds. VsegModel is mocked so load_state_dict is a no-op —
    # the checkpoint content doesn't matter, but torch.load must read a real
    # file. Do NOT mock torch.load globally — the MONAI contenders need the
    # real torch.load to load their trained checkpoints.
    fake_model = MagicMock()
    fake_model.eval.return_value = fake_model
    fake_model.to.return_value = fake_model
    fake_model.state_dict.return_value = {}
    mock_vseg_cls = MagicMock(return_value=fake_model)
    monkeypatch.setattr("liom_toolkit.segmentation.vseg.model.VsegModel", mock_vseg_cls)
    mock_predict_one = MagicMock(return_value=gt_uint8)
    monkeypatch.setattr("liom_toolkit.segmentation.vseg.prediction.predict_one", mock_predict_one)
    monkeypatch.setattr("liom_toolkit.segmentation.vseg.training.train_model", lambda *a, **k: None)

    bench_out = tmp_path / "bench_out"
    ckpt_path = bench_out / "files" / "checkpoint.latest.pth"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({}, str(ckpt_path))

    # --- Mock nnU-Net subprocess calls (nnU-Net runs in a separate venv) ---
    # prepare_nnunet_2d runs for real (it just copies PNGs). The three
    # subprocess calls are mocked. nnunet_predict is mocked to write fake
    # prediction PNGs to the output folder so the read-back path exercises
    # the bool conversion.
    from liom_toolkit.segmentation.vseg.benchmark import nnunet_bridge

    monkeypatch.setattr(nnunet_bridge, "nnunet_plan_and_preprocess", lambda **k: None)
    monkeypatch.setattr(nnunet_bridge, "nnunet_train", lambda **k: None)

    def _fake_nnunet_predict(input_folder, output_folder, dataset_id, **kw):
        """Write fake prediction PNGs matching the input case names."""
        out = Path(output_folder)
        out.mkdir(parents=True, exist_ok=True)
        for f in Path(input_folder).glob("*_0000.png"):
            stem = f.name[: -len("_0000.png")]
            iio.imwrite(out / f"{stem}.png", gt_uint8)

    monkeypatch.setattr(nnunet_bridge, "nnunet_predict", _fake_nnunet_predict)

    split_config = {
        "train_slices": train_slices,
        "test_slices": test_slices,
        "gt_masks": [gt, gt],
        "patch_size": (1, 64, 64),
        "ddp": False,
    }

    nnunet = NnUnetContender(device="cpu", dataset_id=999, nnunet_venv_python="/usr/bin/python3")
    contenders = [
        Improved2DContender(device="cpu"),
        MonaiUnetContender(device="cpu", epochs=1, batch_size=1),
        SwinUnetContender(device="cpu", epochs=1, batch_size=1),
        nnunet,
    ]

    results = run_benchmark(
        contenders=contenders,
        split_config=split_config,
        eval_config=None,
        output_dir=str(tmp_path / "bench_out"),
    )

    assert set(results.keys()) == {
        "improved_2d",
        "monai_unet",
        "monai_swinunetr",
        "nnunet_v2",
    }, f"result table must have all 4 contender keys; got {set(results.keys())}"
    expected_metrics = {
        "centerline_recall",
        "caliber_stratified_recall",
        "boundary_artifact_regression",
        "spurious_thin_vessel_rate",
        "fpr_on_empty",
        "cl_dice_metric",
        "reported_dice",
    }
    for name, row in results.items():
        assert expected_metrics.issubset(row.keys()), (
            f"{name} row must contain all 6 gate metrics + reported_dice; got {set(row.keys())}"
        )


# ---------------------------------------------------------------------------
# MONAI model forward must output LOGITS (no sigmoid in forward)
#
# The composite loss (_monai_composite_loss) uses DiceFocalLoss(sigmoid=True)
# + SoftclDiceLoss(sigmoid=True) — each applies sigmoid internally. The
# post-processing (_monai_predict_slices) uses
# Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)]). If a
# sigmoid is added to the model forward, the loss computes
# sigmoid(sigmoid(logits)) → wrong gradients, and post-processing
# double-sigmoids the predictions → silent wrong data (AGENTS §2). These
# tests assert the model forward outputs unbounded logits (at least one value
# outside [0, 1]); a sigmoid-in-forward regression would clamp the output to
# [0, 1] and the assertion would fail.
# ---------------------------------------------------------------------------


def test_monai_unet_forward_outputs_logits_not_probabilities() -> None:
    """MonaiUnetContender._build_model() forward outputs logits, not probabilities.

    Asserts the MONAI UNet forward must NOT apply sigmoid — the model outputs
    unbounded logits. The composite loss (DiceFocalLoss(sigmoid=True) +
    SoftclDiceLoss(sigmoid=True)) and the post-processing
    (Activations(sigmoid=True)) each apply sigmoid internally; a sigmoid in
    the model forward would compute sigmoid(sigmoid(logits)) → wrong
    gradients, and double-sigmoid the predictions → silent wrong data
    (AGENTS §2). A large-magnitude input (1e4 fill) drives logits far outside
    [0, 1] when no sigmoid clamps them; the assertion requires at least one
    output value outside [0, 1]. A sigmoid-in-forward regression would clamp
    every output to [0, 1] and fail this assertion.
    """
    pytest.importorskip("torch")
    pytest.importorskip("monai")
    import torch

    from liom_toolkit.segmentation.vseg.benchmark.contenders import MonaiUnetContender

    contender = MonaiUnetContender(device="cpu")
    model = contender._build_model()
    model.eval()
    # Large-magnitude input drives logits well outside [0, 1] when no sigmoid
    # clamps the output. (1, 1, 64, 64) float32 — divisible by 32 for the
    # strided encoder.
    x = torch.full((1, 1, 64, 64), 1e4, dtype=torch.float32)
    with torch.no_grad():
        out = model(x)
    out_np = out.detach().cpu().numpy()
    assert out_np.size > 0, "model forward must produce a non-empty output"
    assert (out_np < 0.0).any() or (out_np > 1.0).any(), (
        "MONAI UNet forward must output unbounded logits (at least one value "
        "outside [0, 1]); a sigmoid in forward would clamp to [0, 1] and "
        "cause a 2x sigmoid in the loss + post-processing."
    )


def test_monai_swinunetr_forward_outputs_logits_not_probabilities() -> None:
    """SwinUnetContender._build_model() forward outputs logits, not probabilities.

    Asserts the MONAI SwinUNETR forward must NOT apply sigmoid — the model
    outputs unbounded logits. The composite loss (DiceFocalLoss(sigmoid=True)
    + SoftclDiceLoss(sigmoid=True)) and the post-processing
    (Activations(sigmoid=True)) each apply sigmoid internally; a sigmoid in
    the model forward would compute sigmoid(sigmoid(logits)) → wrong
    gradients, and double-sigmoid the predictions → silent wrong data
    (AGENTS §2). A large-magnitude input (1e4 fill) drives logits far outside
    [0, 1] when no sigmoid clamps them; the assertion requires at least one
    output value outside [0, 1]. A sigmoid-in-forward regression would clamp
    every output to [0, 1] and fail this assertion.

    SwinUNETR uses gradient checkpointing (use_checkpoint=True); under
    torch.no_grad() checkpointing has nothing to checkpoint and may error, so
    the forward runs inside a torch.enable_grad() context (the assertion is on
    the output values, not the gradients). Spatial dims 64×64 are divisible by
    32 (SwinUNETR's requirement).
    """
    pytest.importorskip("torch")
    pytest.importorskip("monai")
    import torch

    from liom_toolkit.segmentation.vseg.benchmark.contenders import SwinUnetContender

    contender = SwinUnetContender(device="cpu")
    model = contender._build_model()
    model.eval()
    # Large-magnitude input drives logits well outside [0, 1] when no sigmoid
    # clamps the output. (1, 1, 64, 64) float32 — divisible by 32 for
    # SwinUNETR's patch embedding.
    x = torch.full((1, 1, 64, 64), 1e4, dtype=torch.float32)
    # use_checkpoint=True requires grad to be enabled (checkpointing replays
    # the forward in a backward pass); run under enable_grad even though we
    # only inspect the output values.
    with torch.enable_grad():
        out = model(x)
    out_np = out.detach().cpu().numpy()
    assert out_np.size > 0, "model forward must produce a non-empty output"
    assert (out_np < 0.0).any() or (out_np > 1.0).any(), (
        "MONAI SwinUNETR forward must output unbounded logits (at least one "
        "value outside [0, 1]); a sigmoid in forward would clamp to [0, 1] "
        "and cause a 2x sigmoid in the loss + post-processing."
    )
