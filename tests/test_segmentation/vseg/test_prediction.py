"""Tests for ``liom_toolkit/segmentation/vseg/prediction.py``.

Covers the ``predict_one`` tiling-arithmetic fix: ``n_patches_by_row`` is
computed as ``int(...)`` so the ``y1 % n_patches_by_row`` modulo is integer
arithmetic with no float rounding. This locks the tiling-path arithmetic for
the future ``patching=True`` implementation (currently raises
``NotImplementedError``).

Per AGENTS.md §5, heavy deps are mocked for orchestration tests:
``torch`` is gated with ``pytest.importorskip("torch")`` (the package's own
``try/except ImportError`` guard mirrors this), and ``do_predict`` is patched
to return a real small uint8 array so the tiling loop runs end-to-end against
real numpy arithmetic (the compute path under test) without a real model
forward pass. ``numpy``/``cv2``/``scikit-image`` are real and unmocked.

The ``torch`` importorskip and the ``predict_one`` import are deferred into a
module-scoped fixture (not module-level) so torch is NOT imported during
collection. torch spawns background threads at import, and importing it during
collection would leave the controller multi-threaded — unsafe for any
fork-based parallel runner and wasteful even under xdist (every worker that
does not run this module would still pay the import). The fixture runs once
per worker that needs it, so torch is imported lazily where it is used.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import imageio.v3 as iio
import numpy as np
import pytest


@pytest.fixture(scope="module")
def predict_one():
    """Import predict_one lazily (torch is imported only where used).

    Deferred from module-level so collection does not import torch. The
    importorskip gates the whole module: if torch is not installed, every
    test in this module skips.
    """
    pytest.importorskip("torch")  # vseg/ requires PyTorch (AGENTS §5, §9)
    from liom_toolkit.segmentation.vseg.prediction import predict_one as _po

    return _po


def _write_synthetic_image(path: str, shape=(16, 16)) -> np.ndarray:
    """Write a small non-constant uint8 PNG for predict_one to read."""
    arr = np.zeros(shape, dtype=np.uint8)
    arr[4:12, 4:12] = 200
    iio.imwrite(path, arr)
    return arr


def test_predict_one_n_patches_by_row_is_int(tmp_path, predict_one):
    """predict_one computes n_patches_by_row as int (integer tiling arithmetic).

    The tiling-path arithmetic (``y1 % n_patches_by_row``) must use integer
    modulo with no float rounding. The function is run end-to-end with
    ``do_predict`` mocked to return a real small uint8 patch, exercising the
    real numpy arithmetic in the tiling loop. The returned mask is uint8
    (0 or 255), confirming the arithmetic path completed without a
    float-modulo rounding error. Fails RED if ``n_patches_by_row`` is a
    float that triggers a rounding bug in the modulo.
    """
    img_path = str(tmp_path / "synth.png")
    _write_synthetic_image(img_path, shape=(16, 16))
    save_path = str(tmp_path / "out")

    # Mock do_predict to return a real uint8 patch matching the image shape.
    # predict_one writes the patch image to disk, reads it back, runs
    # process_image (real torch tensor conversion), then calls do_predict.
    # The mock returns a 2D uint8 array of the expected pred shape.
    def _fake_do_predict(model, patch):
        # Return a uint8 mask the same H/W as the input image.
        return np.zeros((16, 16), dtype=np.uint8)

    with patch(
        "liom_toolkit.segmentation.vseg.prediction.do_predict",
        side_effect=_fake_do_predict,
    ):
        result = predict_one(
            model=None,  # do_predict is mocked; model is never called
            img_path=img_path,
            save_path=save_path,
            dev="cpu",
        )

    # The tiling arithmetic ran end-to-end and produced a valid uint8 mask.
    assert result.dtype == np.uint8
    assert result.shape == (16, 16)
    # Mask values are 0 or 255 (the bool * 255 path).
    assert set(np.unique(result)).issubset({0, 255})


def test_predict_one_modulo_no_float_rounding(tmp_path, predict_one):
    """predict_one tiling arithmetic uses int division/modulo (no float rounding).

    ``n_patches_by_row`` must be ``int(...)`` so ``y1 % n_patches_by_row`` is
    integer arithmetic. A float ``n_patches_by_row`` would silently round
    wrong on the future tiled path. This test runs predict_one end-to-end
    (do_predict mocked) and confirms the tiling loop completes without a
    float-modulo rounding error producing a wrong-shaped or wrong-dtype
    mask. The default single-pass path (stride == H) yields
    ``n_patches_by_row == 1``; the int wrap locks the arithmetic for the
    future ``patching=True`` path.
    """
    img_path = str(tmp_path / "synth_mod.png")
    _write_synthetic_image(img_path, shape=(16, 16))
    save_path = str(tmp_path / "out_mod")

    with patch(
        "liom_toolkit.segmentation.vseg.prediction.do_predict",
        return_value=np.zeros((16, 16), dtype=np.uint8),
    ):
        result = predict_one(
            model=None,
            img_path=img_path,
            save_path=save_path,
            dev="cpu",
        )

    # Integer modulo arithmetic produced a complete, valid mask.
    assert result.shape == (16, 16)
    assert result.dtype == np.uint8
    # The segmented output file was written.
    assert (Path(save_path) / "synth_mod_segmented.png").exists()


def test_predict_one_patching_true_raises_not_implemented(tmp_path, predict_one):
    """predict_one(patching=True) raises NotImplementedError (tiling not implemented).

    The int-modulo fix locks the tiling arithmetic for this future path; the
    path itself is not implemented and must raise rather than silently
    returning plausible-shaped-but-wrong single-pass output.
    """
    img_path = str(tmp_path / "synth_patch.png")
    _write_synthetic_image(img_path, shape=(16, 16))

    with pytest.raises(NotImplementedError):
        predict_one(
            model=None,
            img_path=img_path,
            save_path=str(tmp_path / "out_patch"),
            dev="cpu",
            patching=True,
        )


@pytest.fixture
def nnunet_model(stub_nnunet_model_dir, fake_nnunet_predictor):
    """Construct a real ``NnUnetV2Model`` over the injected fake predictor.

    The ``fake_nnunet_predictor`` sys.modules leaf injection (conftest) makes
    the wrapper's function-scope ``nnunetv2`` import resolve to the recording
    stand-in, so the object under test is the real wrapper class — only the
    nnU-Net leaf is substituted (AGENTS §5 leaf-injection pattern).
    """
    pytest.importorskip("torch")  # model_v2 carries a module-top torch guard
    from liom_toolkit.segmentation.vseg.model_v2 import NnUnetV2Model

    return NnUnetV2Model(stub_nnunet_model_dir, device="cpu")


def test_predict_one_nnunet_calls_predictor_once_with_raw_input(
    tmp_path, predict_one, nnunet_model, fake_nnunet_predictor
):
    """predict_one routes an NnUnetV2Model to model.predict with raw input + spacing.

    The nnU-Net path must call ``predict_single_npy_array`` exactly once via
    the wrapper, with a ``(1, H, W)`` float32 array byte-equal to the raw PNG
    contents (no CLAHE / min-max / gaussian — nnU-Net owns normalization per
    its plans) and ``image_properties == {"spacing": [6.5, 6.5]}``. The
    persistence contract is shared with the legacy path:
    ``{stem}_segmented.png`` written under ``save_path`` and the returned
    mask is uint8 {0, 255}.
    """
    img_path = tmp_path / "nnunet.png"
    raw = _write_synthetic_image(str(img_path), shape=(16, 16))
    save_path = tmp_path / "out_nnunet"

    result = predict_one(
        model=nnunet_model,
        img_path=str(img_path),
        save_path=str(save_path),
        dev="cpu",
        spacing=(6.5, 6.5),
    )

    calls = fake_nnunet_predictor.calls["predict_calls"]
    assert len(calls) == 1
    input_image = calls[0]["input_image"]
    assert input_image.shape == (1, 16, 16)
    assert input_image.dtype == np.float32
    np.testing.assert_array_equal(input_image[0], raw.astype(np.float32))
    assert calls[0]["image_properties"] == {"spacing": [6.5, 6.5]}

    segmented = save_path / "nnunet_segmented.png"
    assert segmented.exists()
    assert result.dtype == np.uint8
    assert result.shape == (16, 16)
    assert set(np.unique(result)).issubset({0, 255})


def test_predict_one_nnunet_ignores_legacy_norm_and_dev(
    tmp_path, predict_one, nnunet_model, fake_nnunet_predictor
):
    """The nnU-Net path ignores norm/norm_param/dev (legacy-only parameters).

    ``norm_param={"a": 1}`` is a dict the legacy path would fail to index as
    ``norm_param[0]`` — surviving it proves the parameter was never touched.
    ``dev="cuda"`` on a CUDA-less host would raise inside the legacy
    ``.to(device)`` path — surviving it proves ``dev`` was never used. The
    byte-equality assertion on the predictor input proves no CLAHE/min-max
    ran regardless of the ``norm`` flag.
    """
    img_path = tmp_path / "nnunet_raw.png"
    raw = _write_synthetic_image(str(img_path), shape=(16, 16))
    save_path = tmp_path / "out_nnunet_raw"

    predict_one(
        model=nnunet_model,
        img_path=str(img_path),
        save_path=str(save_path),
        dev="cuda",
        norm_param={"a": 1},
        norm=False,
        spacing=(6.5, 6.5),
    )

    calls = fake_nnunet_predictor.calls["predict_calls"]
    assert len(calls) == 1
    np.testing.assert_array_equal(
        calls[0]["input_image"][0], raw.astype(np.float32)
    )


def test_predict_one_nnunet_requires_spacing(
    tmp_path, predict_one, nnunet_model, fake_nnunet_predictor
):
    """predict_one on an NnUnetV2Model without spacing raises ValueError.

    A PNG carries no spacing metadata, so spacing must be passed explicitly
    — silently defaulting to isotropic would mis-resample the prediction
    (silent wrong-data). The error fires before any model invocation.
    """
    img_path = tmp_path / "nnunet_nospacing.png"
    _write_synthetic_image(str(img_path), shape=(16, 16))

    with pytest.raises(ValueError, match="spacing"):
        predict_one(
            model=nnunet_model,
            img_path=str(img_path),
            save_path=str(tmp_path / "out_nnunet_nospacing"),
            dev="cpu",
        )

    assert fake_nnunet_predictor.calls["predict_calls"] == []


def test_predict_one_nnunet_all_zero_raises_before_model(
    tmp_path, predict_one, nnunet_model, fake_nnunet_predictor
):
    """predict_one keeps the all-zero ValueError guard in front of the nnU-Net path.

    nnU-Net's ``crop_to_nonzero`` falls back to the full bbox on an all-zero
    image, so without the guard the call would emit a plausible all-255 (or
    all-background) mask on garbage input. The contract guard must fire on
    the raw image BEFORE the predictor is invoked.
    """
    img_path = tmp_path / "nnunet_zeros.png"
    iio.imwrite(img_path, np.zeros((16, 16), dtype=np.uint8))

    with pytest.raises(ValueError, match="all-zero"):
        predict_one(
            model=nnunet_model,
            img_path=str(img_path),
            save_path=str(tmp_path / "out_nnunet_zeros"),
            dev="cpu",
            spacing=(6.5, 6.5),
        )

    assert fake_nnunet_predictor.calls["predict_calls"] == []


def test_predict_one_nnunet_patching_raises_not_implemented(
    tmp_path, predict_one, nnunet_model, fake_nnunet_predictor
):
    """predict_one(patching=True) raises NotImplementedError on the nnU-Net path too.

    Manual 2D tiling was never supported and the same exception (type and
    message) must surface for both model types — the guard precedes the
    type dispatch so no model call happens.
    """
    img_path = tmp_path / "nnunet_patch.png"
    _write_synthetic_image(str(img_path), shape=(16, 16))

    with pytest.raises(NotImplementedError, match="predict_volume"):
        predict_one(
            model=nnunet_model,
            img_path=str(img_path),
            save_path=str(tmp_path / "out_nnunet_patch"),
            dev="cpu",
            patching=True,
            spacing=(6.5, 6.5),
        )

    assert fake_nnunet_predictor.calls["predict_calls"] == []


def test_predict_one_legacy_model_rejects_spacing_kwarg(tmp_path, predict_one):
    """predict_one on a legacy (non-NnUnetV2) model rejects the spacing kwarg.

    ``spacing`` is an nnU-Net-only parameter: passing it with a legacy
    VsegModel must raise a ValueError naming it, not silently ignore it
    (asymmetric kwargs are never silently dropped — the error names the
    offending kwarg so the caller knows it only applies to nnU-Net models).
    """
    img_path = tmp_path / "legacy_spacing.png"
    _write_synthetic_image(str(img_path), shape=(16, 16))

    with pytest.raises(ValueError, match="spacing"):
        predict_one(
            model=None,  # legacy path: not an NnUnetV2Model instance
            img_path=str(img_path),
            save_path=str(tmp_path / "out_legacy_spacing"),
            dev="cpu",
            spacing=(6.5, 6.5),
        )
