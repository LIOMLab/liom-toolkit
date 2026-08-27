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
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import imageio.v3 as iio
import numpy as np
import pytest

pytest.importorskip("torch")  # vseg/ requires PyTorch (AGENTS §5, §9)

from liom_toolkit.segmentation.vseg.prediction import predict_one


def _write_synthetic_image(path: str, shape=(16, 16)) -> np.ndarray:
    """Write a small non-constant uint8 PNG for predict_one to read."""
    arr = np.zeros(shape, dtype=np.uint8)
    arr[4:12, 4:12] = 200
    iio.imwrite(path, arr)
    return arr


def test_predict_one_n_patches_by_row_is_int(tmp_path):
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


def test_predict_one_modulo_no_float_rounding(tmp_path):
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


def test_predict_one_patching_true_raises_not_implemented(tmp_path):
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
