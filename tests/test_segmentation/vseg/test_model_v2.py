"""Tests for the nnU-Net v2 predictor wrapper ``vseg.model_v2.NnUnetV2Model``.

``NnUnetV2Model`` is the single swap point every inference surface routes
through: it owns an ``nnunetv2.inference.predict_from_raw_data.nnUNetPredictor``
constructed EAGERLY from a trained-model directory, validates the model-dir
contract before touching nnU-Net, resolves the compute device explicitly
(nnU-Net's constructor defaults to ``cuda`` with no CPU fallback), and turns
a ``(C, Z, H, W)`` array plus per-axis spacing into the canonical 0/255
uint8 vessel mask via channel 1 of the 2-class softmax.

Two test seams:

* **Fake-predictor tests** -- the ``fake_nnunet_predictor`` fixture in the
  sibling conftest.py injects a fake
  ``nnunetv2.inference.predict_from_raw_data`` leaf module into
  ``sys.modules`` and records every call, so the wrapper's contract is
  asserted without the real nnU-Net package. These tests do NOT
  ``importorskip("nnunetv2")`` -- the point of leaf injection is that real
  nnunetv2 is never imported. They DO ``importorskip("torch")`` at the first
  body line, because model_v2.py has a module-top torch guard.
* **Real-API tracer** -- one ``@pytest.mark.slow`` test exercises the real
  nnU-Net v2 stack end-to-end on CPU (real ``ExperimentPlanner`` ->
  real ``plans.json`` -> hand-built ``fold_0`` checkpoint -> real
  ``predict_single_npy_array``) so the recorded contract is proven against
  the installed package.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Construction + device resolution (fake predictor)
# ---------------------------------------------------------------------------


@pytest.mark.ai
def test_init_initializes_predictor_eagerly(fake_nnunet_predictor, stub_nnunet_model_dir) -> None:
    """Construction validates model_dir, then builds + initializes the predictor eagerly.

    The wrapper owns the predictor as a unit: ``__init__`` must call
    ``initialize_from_trained_model_folder`` exactly once with the resolved
    model_dir, ``use_folds`` and ``checkpoint_name``, and expose
    ``.predictor`` / ``.device`` / ``.model_dir``. Eager (not lazy) init is
    the contract -- a bogus model_dir fails at construction time, not on
    first predict.
    """
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.model_v2 import NnUnetV2Model

    model = NnUnetV2Model(stub_nnunet_model_dir)

    init_calls = fake_nnunet_predictor.calls["init_calls"]
    assert len(init_calls) == 1
    assert init_calls[0]["model_dir"] == str(stub_nnunet_model_dir)
    assert init_calls[0]["use_folds"] is None
    assert init_calls[0]["checkpoint_name"] == "checkpoint_final.pth"
    assert isinstance(model.predictor, fake_nnunet_predictor.predictor_cls)
    assert model.model_dir == Path(stub_nnunet_model_dir)


@pytest.mark.ai
def test_init_device_none_resolves_to_available_device(
    fake_nnunet_predictor, stub_nnunet_model_dir
) -> None:
    """device=None resolves to cuda iff torch.cuda.is_available() else cpu.

    nnU-Net's ``nnUNetPredictor`` constructor defaults ``device`` to cuda
    unconditionally -- on a CPU-only host that default silently breaks. The
    wrapper resolves the device itself and passes it explicitly, and
    ``perform_everything_on_device`` tracks ``device.type == "cuda"`` (on-
    device preprocessing only pays off on GPU).
    """
    pytest.importorskip("torch")
    import torch

    from liom_toolkit.segmentation.vseg.model_v2 import NnUnetV2Model

    model = NnUnetV2Model(stub_nnunet_model_dir, device=None)

    expected_type = "cuda" if torch.cuda.is_available() else "cpu"
    assert model.device == torch.device(expected_type)
    ctor_kwargs = fake_nnunet_predictor.calls["ctor_kwargs"][-1]
    assert ctor_kwargs["device"] == torch.device(expected_type)
    assert ctor_kwargs["perform_everything_on_device"] is (expected_type == "cuda")


@pytest.mark.ai
def test_init_honors_explicit_device(fake_nnunet_predictor, stub_nnunet_model_dir) -> None:
    """An explicit device (str or torch.device) is honored verbatim.

    The caller owns device selection; the wrapper must forward it unchanged
    rather than re-detecting. On CPU ``perform_everything_on_device`` is
    False (nnU-Net would otherwise run the preprocessor on a device that
    gains nothing).
    """
    pytest.importorskip("torch")
    import torch

    from liom_toolkit.segmentation.vseg.model_v2 import NnUnetV2Model

    for device in ("cpu", torch.device("cpu")):
        model = NnUnetV2Model(stub_nnunet_model_dir, device=device)
        assert model.device == torch.device("cpu")
        ctor_kwargs = fake_nnunet_predictor.calls["ctor_kwargs"][-1]
        assert ctor_kwargs["device"] == torch.device("cpu")
        assert ctor_kwargs["perform_everything_on_device"] is False


# ---------------------------------------------------------------------------
# model_dir contract validation (fires BEFORE nnU-Net is touched)
# ---------------------------------------------------------------------------


@pytest.mark.ai
def test_init_rejects_nonexistent_and_non_dir_model_dir(fake_nnunet_predictor, tmp_path) -> None:
    """A model_dir that does not exist or is a file raises ValueError naming the path.

    The check fires BEFORE nnU-Net is constructed -- nnU-Net would otherwise
    raise an opaque FileNotFoundError deep inside initialize; the wrapper's
    up-front ValueError names the offending path so the caller can act.
    """
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.model_v2 import NnUnetV2Model

    missing = tmp_path / "not_here"
    with pytest.raises(ValueError, match=re.escape(str(missing))):
        NnUnetV2Model(missing)

    a_file = tmp_path / "a_file"
    a_file.write_text("not a directory")
    with pytest.raises(ValueError, match=re.escape(str(a_file))):
        NnUnetV2Model(a_file)

    # nnU-Net is never touched when the dir contract fails.
    assert fake_nnunet_predictor.calls["ctor_kwargs"] == []
    assert fake_nnunet_predictor.calls["init_calls"] == []


@pytest.mark.ai
def test_init_rejects_missing_dataset_json(fake_nnunet_predictor, tmp_path) -> None:
    """A model_dir without dataset.json raises ValueError naming dataset.json.

    ``initialize_from_trained_model_folder`` reads ``dataset.json`` for the
    label manager; a missing file must surface as a path-naming ValueError,
    not nnU-Net's raw load_json FileNotFoundError.
    """
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.model_v2 import NnUnetV2Model

    model_dir = tmp_path / "model"
    (model_dir / "fold_0").mkdir(parents=True)
    (model_dir / "plans.json").write_text("{}")
    (model_dir / "fold_0" / "checkpoint_final.pth").write_bytes(b"x")
    with pytest.raises(ValueError, match=r"dataset\.json"):
        NnUnetV2Model(model_dir)
    assert fake_nnunet_predictor.calls["init_calls"] == []


@pytest.mark.ai
def test_init_rejects_missing_plans_json(fake_nnunet_predictor, tmp_path) -> None:
    """A model_dir without plans.json raises ValueError naming plans.json."""
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.model_v2 import NnUnetV2Model

    model_dir = tmp_path / "model"
    (model_dir / "fold_0").mkdir(parents=True)
    (model_dir / "dataset.json").write_text("{}")
    (model_dir / "fold_0" / "checkpoint_final.pth").write_bytes(b"x")
    with pytest.raises(ValueError, match=r"plans\.json"):
        NnUnetV2Model(model_dir)
    assert fake_nnunet_predictor.calls["init_calls"] == []


@pytest.mark.ai
def test_init_rejects_missing_fold_checkpoint(fake_nnunet_predictor, tmp_path) -> None:
    """No fold_*/checkpoint_final.pth raises ValueError naming the checkpoint.

    Covers both subcases: no ``fold_*`` subdirectory at all, and a ``fold_0``
    directory that exists but lacks the checkpoint file -- nnU-Net's
    auto-detect would find zero usable folds and fail later.
    """
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.model_v2 import NnUnetV2Model

    for sub in ("no_folds", "empty_fold"):
        model_dir = tmp_path / sub
        model_dir.mkdir()
        (model_dir / "dataset.json").write_text("{}")
        (model_dir / "plans.json").write_text("{}")
        if sub == "empty_fold":
            (model_dir / "fold_0").mkdir()
        with pytest.raises(ValueError, match=r"checkpoint_final\.pth"):
            NnUnetV2Model(model_dir)
    assert fake_nnunet_predictor.calls["init_calls"] == []


@pytest.mark.ai
def test_init_validates_explicit_use_folds(fake_nnunet_predictor, stub_nnunet_model_dir) -> None:
    """An explicit use_folds validates exactly those fold dirs.

    ``use_folds=(1,)`` on a dir that only contains ``fold_0`` raises
    ValueError naming the missing ``fold_1`` checkpoint path; a valid
    explicit ``use_folds=(0,)`` is forwarded to nnU-Net verbatim.
    """
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.model_v2 import NnUnetV2Model

    with pytest.raises(ValueError, match="fold_1"):
        NnUnetV2Model(stub_nnunet_model_dir, use_folds=(1,))

    NnUnetV2Model(stub_nnunet_model_dir, use_folds=(0,))
    assert fake_nnunet_predictor.calls["init_calls"][-1]["use_folds"] == (0,)


@pytest.mark.ai
def test_init_rejects_empty_use_folds(fake_nnunet_predictor, stub_nnunet_model_dir) -> None:
    """An empty use_folds raises ValueError instead of loading zero checkpoints.

    An empty sequence would skip auto-detection AND pass the fold loop
    vacuously, leaving ``list_of_parameters`` empty -- inference then
    crashes opaquely on ``None.to('cpu')`` deep inside nnU-Net. The wrapper
    must reject it at construction.
    """
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.model_v2 import NnUnetV2Model

    with pytest.raises(ValueError, match="non-empty"):
        NnUnetV2Model(stub_nnunet_model_dir, use_folds=())

    assert fake_nnunet_predictor.calls["init_calls"] == []


@pytest.mark.ai
def test_init_excludes_fold_all_unless_requested(fake_nnunet_predictor, tmp_path) -> None:
    """fold_all is NOT an auto-detected fold -- a fold_all-only dir raises ValueError.

    Mirrors nnU-Net's ``auto_detect_available_folds``, which filters
    ``fold_all`` out of auto-detection. Requesting ``use_folds=("all",)``
    explicitly still works.
    """
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.model_v2 import NnUnetV2Model

    model_dir = tmp_path / "model"
    (model_dir / "fold_all").mkdir(parents=True)
    (model_dir / "dataset.json").write_text("{}")
    (model_dir / "plans.json").write_text("{}")
    (model_dir / "fold_all" / "checkpoint_final.pth").write_bytes(b"x")

    with pytest.raises(ValueError, match=r"checkpoint_final\.pth"):
        NnUnetV2Model(model_dir)

    NnUnetV2Model(model_dir, use_folds=("all",))
    assert fake_nnunet_predictor.calls["init_calls"][-1]["use_folds"] == ("all",)


@pytest.mark.ai
def test_init_rejects_out_of_range_tile_step_size(
    fake_nnunet_predictor, stub_nnunet_model_dir
) -> None:
    """tile_step_size outside (0, 1] raises ValueError naming the value.

    nnU-Net's sliding-window math steps by ``tile_step_size * tile``: a
    step > 1 leaves un-predicted gaps that Gaussian blending fills with
    near-zero-weight garbage (plausible-shaped-but-wrong), and a step <= 0
    crashes on division upstream. The bound is validated at construction,
    before the predictor is built.
    """
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.model_v2 import NnUnetV2Model

    for bad in (0.0, -0.5, 1.5, 2.0):
        with pytest.raises(ValueError, match="tile_step_size"):
            NnUnetV2Model(stub_nnunet_model_dir, tile_step_size=bad)

    # Boundary value 1.0 is valid (non-overlapping adjacent tiles).
    NnUnetV2Model(stub_nnunet_model_dir, tile_step_size=1.0)
    assert fake_nnunet_predictor.calls["ctor_kwargs"][-1]["tile_step_size"] == 1.0


# ---------------------------------------------------------------------------
# predict_proba input validation
# ---------------------------------------------------------------------------


@pytest.mark.ai
def test_predict_proba_rejects_non_ndarray_and_bad_ndim(
    fake_nnunet_predictor, stub_nnunet_model_dir
) -> None:
    """predict_proba rejects non-ndarray input and ndim != 4 with ValueError.

    Every nnunetv2 preprocessor applies the always-length-3
    ``transpose_forward`` permutation, so a (C,H,W) input crashes deep in
    ``run_case_npy`` -- the wrapper requires exactly (C,Z,H,W) and fails
    fast naming the offending shape; the predictor is never called on bad
    input.
    """
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.model_v2 import NnUnetV2Model

    model = NnUnetV2Model(stub_nnunet_model_dir)

    with pytest.raises(ValueError, match="ndarray"):
        model.predict_proba([[0.0]], (1.0,))

    for bad in (np.zeros((4, 4)), np.zeros((1, 4, 4)), np.zeros((1, 1, 2, 4, 4))):
        with pytest.raises(ValueError, match="ndim"):
            model.predict_proba(bad, (1.0,) * (bad.ndim - 1))

    assert fake_nnunet_predictor.calls["predict_calls"] == []


@pytest.mark.ai
def test_predict_proba_rejects_empty_dims(fake_nnunet_predictor, stub_nnunet_model_dir) -> None:
    """A zero channel count or a zero spatial dim raises ValueError naming the dim.

    A zero-sized axis would produce a zero-sized mask that LOOKS plausible
    downstream -- the AGENTS no-silent-wrong-data rule requires an explicit
    failure instead.
    """
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.model_v2 import NnUnetV2Model

    model = NnUnetV2Model(stub_nnunet_model_dir)

    with pytest.raises(ValueError, match="0"):
        model.predict_proba(np.zeros((0, 2, 4, 4)), (1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="0"):
        model.predict_proba(np.zeros((1, 0, 4, 4)), (1.0, 1.0, 1.0))

    assert fake_nnunet_predictor.calls["predict_calls"] == []


@pytest.mark.ai
def test_predict_proba_rejects_bad_spacing(fake_nnunet_predictor, stub_nnunet_model_dir) -> None:
    """Wrong spacing length, non-positive, or non-finite spacing raises ValueError.

    Spacing drives nnU-Net's resampling; a silently-defaulted or mis-ordered
    spacing mis-resamples the volume (the AGENTS wrong-data class). Every
    invalid spacing names the offending value.
    """
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.model_v2 import NnUnetV2Model

    model = NnUnetV2Model(stub_nnunet_model_dir)
    arr = np.zeros((1, 2, 4, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="spacing"):
        model.predict_proba(arr, (6.5, 6.5))
    with pytest.raises(ValueError, match="spacing"):
        model.predict_proba(arr, (6.5, 6.5, 6.5, 6.5))
    for bad in (
        (0.0, 6.5, 6.5),
        (-1.0, 6.5, 6.5),
        (float("inf"), 6.5, 6.5),
        (float("nan"), 6.5, 6.5),
    ):
        with pytest.raises(ValueError, match="spacing"):
            model.predict_proba(arr, bad)

    assert fake_nnunet_predictor.calls["predict_calls"] == []


# ---------------------------------------------------------------------------
# Probability -> mask contract
# ---------------------------------------------------------------------------


@pytest.mark.ai
def test_predict_proba_forwards_float32_array_and_spacing(
    fake_nnunet_predictor, stub_nnunet_model_dir
) -> None:
    """predict_single_npy_array gets the float32-cast array + {'spacing': [...]} + probs flag.

    The caller's array is passed UNMODIFIED apart from the float32 cast (no
    CLAHE / min-max -- nnU-Net normalizes per its plans), and
    ``image_properties`` carries ONLY the ``'spacing'`` key in the array's
    own spatial axis order. The return value is the probability array, not
    the labelmap.
    """
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.model_v2 import NnUnetV2Model

    model = NnUnetV2Model(stub_nnunet_model_dir)
    arr = np.arange(1 * 2 * 4 * 4, dtype=np.float64).reshape(1, 2, 4, 4)

    probs = model.predict_proba(arr, spacing=(6.5, 6.5, 6.5))

    call = fake_nnunet_predictor.calls["predict_calls"][-1]
    assert call["input_image"].dtype == np.float32
    np.testing.assert_array_equal(call["input_image"], arr.astype(np.float32))
    assert call["image_properties"] == {"spacing": [6.5, 6.5, 6.5]}
    assert call["save_or_return_probabilities"] is True
    assert probs.shape == (2, 2, 4, 4)
    assert probs.dtype == np.float32


@pytest.mark.ai
def test_predict_returns_uint8_mask_with_expected_values(
    fake_nnunet_predictor, stub_nnunet_model_dir
) -> None:
    """predict thresholds channel 1 of the 2-class softmax into a {0,255} uint8 mask.

    ``probs[1]`` IS the vessel probability for the binary vessel model;
    values above 0.5 become 255, the rest 0. The output shape equals the
    input's spatial dims positionally -- no reordering, no sorting.
    """
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.model_v2 import NnUnetV2Model

    model = NnUnetV2Model(stub_nnunet_model_dir)
    probs = np.zeros((2, 2, 4, 4), dtype=np.float32)
    probs[1, 0, 0, 0] = 0.9  # one vessel voxel
    probs[0] = 1.0 - probs[1]
    fake_nnunet_predictor.state["probs"] = probs

    mask = model.predict(np.zeros((1, 2, 4, 4), np.float32), (6.5, 6.5, 6.5))

    assert mask.dtype == np.uint8
    assert mask.shape == (2, 4, 4)
    assert set(np.unique(mask).tolist()) <= {0, 255}
    assert mask[0, 0, 0] == 255
    assert int(mask.sum()) == 255


@pytest.mark.ai
def test_predict_threshold_boundary_is_strict(fake_nnunet_predictor, stub_nnunet_model_dir) -> None:
    """The threshold is strict ``> 0.5``: a probability of exactly 0.5 maps to 0.

    A ``>=`` boundary would classify a maximally-uncertain voxel as vessel;
    the strict boundary is the locked contract.
    """
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.model_v2 import NnUnetV2Model

    model = NnUnetV2Model(stub_nnunet_model_dir)
    probs = np.zeros((2, 2, 4, 4), dtype=np.float32)
    probs[1] = 0.5
    probs[0] = 0.5
    fake_nnunet_predictor.state["probs"] = probs

    mask = model.predict(np.zeros((1, 2, 4, 4), np.float32), (6.5, 6.5, 6.5))
    assert int(mask.sum()) == 0

    # float32-aware nextafter: a float64 nextafter would round back to 0.5
    # when stored into the float32 array.
    probs[1, 0, 0, 0] = np.nextafter(np.float32(0.5), np.float32(1.0), dtype=np.float32)
    probs[0, 0, 0, 0] = 1.0 - probs[1, 0, 0, 0]
    mask = model.predict(np.zeros((1, 2, 4, 4), np.float32), (6.5, 6.5, 6.5))
    assert mask[0, 0, 0] == 255


@pytest.mark.ai
def test_predict_rejects_single_channel_probabilities(
    fake_nnunet_predictor, stub_nnunet_model_dir
) -> None:
    """A model returning fewer than 2 probability channels raises ValueError.

    A single-class softmax has no vessel channel -- reading ``probs[1]``
    would be a silent wrong-channel read. The wrapper must raise and name
    the class count.
    """
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.model_v2 import NnUnetV2Model

    model = NnUnetV2Model(stub_nnunet_model_dir)
    fake_nnunet_predictor.state["probs"] = np.zeros((1, 2, 4, 4), dtype=np.float32)

    with pytest.raises(ValueError, match=r"1 probability channel"):
        model.predict(np.zeros((1, 2, 4, 4), np.float32), (6.5, 6.5, 6.5))


@pytest.mark.ai
def test_predict_accepts_all_zero_input(fake_nnunet_predictor, stub_nnunet_model_dir) -> None:
    """An all-zero input array does NOT raise from the wrapper.

    nnU-Net's ``crop_to_nonzero`` returns the full bounding box on an
    all-zero array, so the call proceeds normally; the predict_one-level
    zero-guard lives in ``prediction.py``, not in this adapter.
    """
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.model_v2 import NnUnetV2Model

    model = NnUnetV2Model(stub_nnunet_model_dir)
    arr = np.zeros((1, 2, 4, 4), dtype=np.float32)

    mask = model.predict(arr, (6.5, 6.5, 6.5))

    assert mask.shape == (2, 4, 4)
    assert mask.dtype == np.uint8


# ---------------------------------------------------------------------------
# Class shape + source invariants
# ---------------------------------------------------------------------------


@pytest.mark.ai
def test_model_is_not_a_torch_nn_module(fake_nnunet_predictor, stub_nnunet_model_dir) -> None:
    """NnUnetV2Model does NOT subclass torch.nn.Module.

    The nnU-Net predictor owns the full inference pipeline (preprocessing,
    sliding window, ensembling); the wrapper is a leaf adapter over array and
    spacing contracts, not a torch module.
    """
    pytest.importorskip("torch")
    import torch

    from liom_toolkit.segmentation.vseg.model_v2 import NnUnetV2Model

    assert not issubclass(NnUnetV2Model, torch.nn.Module)


@pytest.mark.ai
def test_model_v2_has_no_assert_or_preprocessing_imports() -> None:
    """model_v2.py contains no ``assert`` statements and no cv2/skimage imports.

    Config-as-data invariant (same pattern as the ssl test suite):
    ``assert`` is stripped under ``python -O`` so validation must use
    ``if ...: raise``; cv2/skimage preprocessing inside the wrapper is
    forbidden -- nnU-Net normalizes per its plans, and a second
    preprocessing step would silently double-transform the input.
    """
    pytest.importorskip("torch")
    import liom_toolkit.segmentation.vseg.model_v2 as model_v2_mod

    src = Path(model_v2_mod.__file__).read_text()
    assert not re.search(r"^\s*assert\b", src, flags=re.MULTILINE), (
        "model_v2.py must not use `assert` for validation (stripped under "
        "python -O); use `if ...: raise ValueError` instead"
    )
    import_lines = [line for line in src.splitlines() if re.match(r"^\s*(import |from )", line)]
    assert not any("cv2" in line or "skimage" in line for line in import_lines), (
        "model_v2.py must not import cv2/skimage -- nnU-Net owns preprocessing"
    )


# ---------------------------------------------------------------------------
# Real nnU-Net v2 API round-trip (the tracer proof)
# ---------------------------------------------------------------------------


@pytest.mark.ai
@pytest.mark.slow
def test_real_nnunet_predictor_roundtrip_cpu(tmp_path, monkeypatch) -> None:
    """Real nnU-Net v2 API round-trip on CPU: fingerprint -> plan -> checkpoint -> predict.

    The fake-predictor tests pin the contract; this test proves the contract
    against the INSTALLED nnunetv2: a real ``ExperimentPlanner`` produces a
    real ``plans.json`` for a tiny 4-case PNG dataset written by
    ``prepare_nnunet_2d``, a network built from those plans supplies the
    ``fold_0`` checkpoint, and the wrapper drives a real
    ``initialize_from_trained_model_folder`` + ``predict_single_npy_array``
    on CPU. The assertion is the mask contract: ``(Z,H,W)`` uint8 with
    values in {0, 255}.
    """
    pytest.importorskip("torch")

    raw_dir = tmp_path / "nnUNet_raw"
    preproc_dir = tmp_path / "nnUNet_preprocessed"
    results_dir = tmp_path / "nnUNet_results"
    for directory in (raw_dir, preproc_dir, results_dir):
        directory.mkdir()
    # nnunetv2 reads the three path env vars at import time in several
    # modules, and the fingerprint extractor's spawn-Pool children re-import
    # and re-read the env -- so the env vars must be set BEFORE nnunetv2 is
    # imported, and every already-loaded nnunetv2 module's bound copies must
    # be patched too (import order is not guaranteed across the suite).
    monkeypatch.setenv("nnUNet_raw", str(raw_dir))
    monkeypatch.setenv("nnUNet_preprocessed", str(preproc_dir))
    monkeypatch.setenv("nnUNet_results", str(results_dir))
    pytest.importorskip("nnunetv2")
    patched_paths = {
        "nnUNet_raw": str(raw_dir),
        "nnUNet_preprocessed": str(preproc_dir),
        "nnUNet_results": str(results_dir),
    }
    for mod_name, mod in list(sys.modules.items()):
        if mod_name == "nnunetv2" or mod_name.startswith("nnunetv2."):
            for attr, val in patched_paths.items():
                if hasattr(mod, attr):
                    monkeypatch.setattr(mod, attr, val)

    import json

    import imageio.v3 as iio
    import torch
    from nnunetv2.experiment_planning.plan_and_preprocess_api import (
        extract_fingerprint_dataset,
        plan_experiment_dataset,
    )

    from liom_toolkit.scripts.liom_prepare_nnunet_dataset import prepare_nnunet_2d
    from liom_toolkit.segmentation.vseg.model_v2 import NnUnetV2Model
    from liom_toolkit.segmentation.vseg.ssl.pretrain import build_pretrain_network

    dataset_id = 999
    dataset_dirname = f"Dataset{dataset_id:03d}_Tiny"
    rng = np.random.default_rng(0)
    src_dir = tmp_path / "src_slices"
    src_dir.mkdir()
    image_paths, label_paths = [], []
    for i in range(4):
        img = (rng.random((32, 48)) * 200 + 30).astype(np.uint8)
        lbl = np.zeros((32, 48), dtype=np.uint8)
        lbl[8:12, 10:30] = 1
        img_path = src_dir / f"img_{i}.png"
        lbl_path = src_dir / f"img_{i}_mask.png"
        iio.imwrite(img_path, img)
        iio.imwrite(lbl_path, lbl)
        image_paths.append(str(img_path))
        label_paths.append(str(lbl_path))

    prepare_nnunet_2d(
        image_paths,
        label_paths,
        str(raw_dir / dataset_dirname),
        dataset_id=dataset_id,
        dataset_name="Tiny",
    )

    extract_fingerprint_dataset(
        dataset_id,
        num_processes=1,
        check_dataset_integrity=False,
        clean=True,
        verbose=False,
        show_progress_bar=False,
    )
    plans_dict, plans_identifier = plan_experiment_dataset(dataset_id)
    if "2d" not in plans_dict["configurations"]:
        pytest.fail(
            "ExperimentPlanner produced no 2d configuration; got "
            f"{sorted(plans_dict['configurations'])}"
        )
    dataset_json = json.loads((raw_dir / dataset_dirname / "dataset.json").read_text())
    net = build_pretrain_network(plans_dict, dataset_json, configuration="2d", device="cpu")

    model_dir = results_dir / dataset_dirname / f"nnUNetTrainer__{plans_identifier}__2d"
    (model_dir / "fold_0").mkdir(parents=True)
    (model_dir / "dataset.json").write_text(json.dumps(dataset_json))
    (model_dir / "plans.json").write_text(json.dumps(plans_dict))
    torch.save(
        {
            "network_weights": net.state_dict(),
            "trainer_name": "nnUNetTrainer",
            "init_args": {"configuration": "2d"},
            "inference_allowed_mirroring_axes": (0, 1),
        },
        str(model_dir / "fold_0" / "checkpoint_final.pth"),
    )

    model = NnUnetV2Model(str(model_dir), device="cpu")
    arr = rng.random((1, 2, 32, 48)).astype(np.float32)
    mask = model.predict(arr, spacing=(6.5, 6.5, 6.5))

    assert mask.dtype == np.uint8
    assert mask.shape == (2, 32, 48)
    assert set(np.unique(mask).tolist()) <= {0, 255}


# ---------------------------------------------------------------------------
# Barrel contract: lazy export, no eager heavy deps
# ---------------------------------------------------------------------------


def test_vseg_barrel_imports_without_torch_or_nnunetv2() -> None:
    """``import liom_toolkit.segmentation.vseg`` pulls neither torch nor nnunetv2.

    Ungated (no importorskip): the barrel must import on a core-only
    install. ``liom_toolkit.*`` is purged so the import re-runs fresh, and
    ``torch`` / ``nnunetv2`` entries are purged too -- otherwise a stale
    import by an earlier test in the same xdist worker would either mask an
    eager import inside the barrel or fake a pass. Everything is restored
    in ``finally`` (the test_imports.py purge/restore discipline). The
    ``__all__`` assertion pins the curated barrel: ``NnUnetV2Model`` is
    deliberately absent so star-import stays safe on core-only installs.
    """
    prefixes = ("liom_toolkit", "torch", "nnunetv2")
    saved = {
        name: mod
        for name, mod in sys.modules.items()
        if any(name == p or name.startswith(p + ".") for p in prefixes)
    }
    for name in saved:
        sys.modules.pop(name)
    try:
        import liom_toolkit.segmentation.vseg as vseg

        assert "torch" not in sys.modules
        assert "nnunetv2" not in sys.modules
        assert vseg.__all__ == ["predict_one", "predict_volume"]
        assert "NnUnetV2Model" not in vseg.__all__
    finally:
        for name in list(sys.modules):
            if any(name == p or name.startswith(p + ".") for p in prefixes):
                sys.modules.pop(name)
        sys.modules.update(saved)


@pytest.mark.ai
def test_vseg_nnunetv2model_resolves_lazily() -> None:
    """``vseg.NnUnetV2Model`` resolves via ``__getattr__`` to the same class object.

    The lazy export must return the identical class a direct
    ``from ...model_v2 import NnUnetV2Model`` yields (the module is cached,
    so attribute access is cheap on repeat), and an unknown attribute raises
    ``AttributeError`` naming it.
    """
    pytest.importorskip("torch")
    import liom_toolkit.segmentation.vseg as vseg
    from liom_toolkit.segmentation.vseg.model_v2 import NnUnetV2Model

    assert vseg.NnUnetV2Model is NnUnetV2Model
    with pytest.raises(AttributeError, match="bogus_attribute"):
        vseg.bogus_attribute
