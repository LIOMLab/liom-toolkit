"""Known-answer tests for ``liom_toolkit/segmentation/vseg/utils.py``.

Both tests target functions whose module (``vseg/utils.py``) has module-top
imports of ``torch`` and ``sklearn.metrics``. Per D-02 we do NOT stub those
deps; instead each test calls ``pytest.importorskip`` BEFORE importing from
``vseg.utils`` so the tests run for real on the 3.12-full CI leg (where the
``ai`` extra is installed) and cleanly skip on the 3.14-core leg (where it is
not). The importorskip must precede the ``from liom_toolkit...`` statement —
importing the module before the skips would crash on missing torch.

Coverage:

* ``calculate_metrics`` — known-answer for ``[f1, recall, accuracy, jaccard,
  precision]`` on a small binary mask pair (returns 5 floats; f1 asserted via
  ``pytest.approx``).
* ``add_patch_to_empty_array`` — inserts a 4x4 patch into an 8x8 zero array
  at coords (0, 0) with ``overlap=0``. The ``overlap=0`` boundary avoids the
  multi-branch overlap-blending logic (utils.py:134-172), keeping the
  known-answer deterministic: the patch region equals ``pred_y`` and the
  untouched region stays zero.
"""

import numpy as np
import pytest


def test_calculate_metrics_known_answer():
    """calculate_metrics returns [f1, recall, accuracy, jaccard, precision]."""
    pytest.importorskip("torch")  # vseg/utils.py module-top imports torch
    pytest.importorskip("sklearn")  # calculate_metrics uses sklearn.metrics
    from liom_toolkit.segmentation.vseg.utils import calculate_metrics

    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    result = calculate_metrics(y_true, y_pred)

    assert len(result) == 5
    # f1 is result[0]; verified value from 02-RESEARCH.md
    assert result[0] == pytest.approx(0.6667, abs=1e-3)


def test_add_patch_to_empty_array_inserts_patch():
    """add_patch_to_empty_array writes pred_y into the patch region; overlap=0
    avoids the blending branches so the untouched region stays zero."""
    pytest.importorskip("torch")  # vseg/utils.py module-top imports torch
    from liom_toolkit.segmentation.vseg.utils import add_patch_to_empty_array

    inference = np.zeros((8, 8), dtype=np.float32)
    pred_y = np.ones((4, 4), dtype=np.float32)

    result = add_patch_to_empty_array(
        inference,
        pred_y,
        coords=(0, 0),
        stride=4,
        overlap=0,
        size=(4, 4),
    )

    # Patch region (0:4, 0:4) equals pred_y (all ones)
    assert np.array_equal(result[0:4, 0:4], pred_y)
    # Untouched region (4:8, 4:8) remains all zeros
    assert np.array_equal(result[4:8, 4:8], np.zeros((4, 4), dtype=np.float32))


def test_create_patches_view_as_windows_shape_and_contents(tmp_path):
    """create_patches returns a patch grid whose shape matches
    ``skimage.util.view_as_windows`` and whose per-patch contents equal the
    corresponding window of the (cropped, scaled) source image.

    This test intentionally does NOT call ``pytest.importorskip("torch")``:
    ``create_patches`` touches only imageio / skimage.color / skimage.exposure
    / skimage.util / numpy / PIL, all of which are core deps. ``torch`` is
    ``TYPE_CHECKING``-only in ``vseg/utils.py`` (lines 16-17), so the module
    imports cleanly without the ``ai`` extra.
    """
    import imageio.v3 as iio
    from skimage.util import view_as_windows

    from liom_toolkit.segmentation.vseg.utils import create_patches, crop_image

    # Build a 32x32 uint8 image with a recognizable bright square so we can
    # assert per-patch contents, not just shape.
    img = np.zeros((32, 32), dtype=np.uint8)
    img[4:12, 4:12] = 200
    png = tmp_path / "t.png"
    iio.imwrite(str(png), img)

    size = (8, 8)
    stride = 4
    patches, img_shape, patch_shape, image_clahe = create_patches(
        str(png), size=size, stride=stride
    )

    # Reproduce the (cropped, scaled) source the same way create_patches does
    # so we can verify per-patch contents against the view_as_windows oracle.
    raw = iio.imread(str(png))
    if raw.ndim == 3:
        from skimage.color import rgb2gray
        raw = rgb2gray(raw)
    cropped = crop_image(raw, size, stride)
    scaled = (cropped / cropped.max() * 255).astype(np.uint8)
    expected = view_as_windows(scaled, size, stride)
    n_h, n_w, p_h, p_w = expected.shape

    # Patch grid shape returned by create_patches matches the oracle.
    assert patch_shape == (n_h, n_w, p_h, p_w)
    # Flat patch list length matches n_h * n_w.
    assert len(patches) == n_h * n_w
    # Each patch's contents equal the corresponding window. The downstream
    # ``patch.reshape(n_h*n_w, p_h, p_w)`` must not silently corrupt the
    # strided view (AGENTS §2): the reshape forces a copy, so contents are
    # equal to the source windows, not aliased garbage.
    for i in range(n_h):
        for j in range(n_w):
            np.testing.assert_array_equal(patches[i * n_w + j], expected[i, j])

    # The module must no longer expose a ``patchify`` attribute (the import
    # is gone after the GREEN swap).
    import liom_toolkit.segmentation.vseg.utils as u
    assert not hasattr(u, "patchify"), "vseg/utils.py still imports patchify"


def test_create_patches_non_divisible(tmp_path):
    """Non-divisible image dimensions still produce a valid patch grid:
    ``crop_image`` trims the borders so the window fits, and the call must
    not raise. Verifies the view_as_windows path handles the cropped shape
    without crashing (AGENTS §2: no silent data loss / wrong-data fallback).
    """
    import imageio.v3 as iio

    from liom_toolkit.segmentation.vseg.utils import create_patches

    # 30x30 is not evenly divisible by size=8 + stride=4; crop_image trims
    # 1px off each side -> 28x28 -> 6x6 patch grid.
    img = np.zeros((30, 30), dtype=np.uint8)
    img[10:20, 10:20] = 200
    png = tmp_path / "t30.png"
    iio.imwrite(str(png), img)

    patches, img_shape, patch_shape, _ = create_patches(
        str(png), size=(8, 8), stride=4
    )

    assert len(patches) > 0
    # patch_shape is (n_h, n_w, 8, 8); n_h == n_w == 6 for the 28x28 crop.
    assert patch_shape[2] == 8 and patch_shape[3] == 8
    assert patch_shape[0] == 6 and patch_shape[1] == 6
    assert len(patches) == patch_shape[0] * patch_shape[1]


def test_pil_max_image_pixels_is_finite():
    """vseg/utils.py sets PIL.Image.MAX_IMAGE_PIXELS to a finite integer
    (2_000_000_000), not None. None disables PIL's decompression-bomb guard
    entirely (AGENTS §2 DoS vector); the finite limit preserves the guard
    for untrusted inputs while accommodating legitimate large microscopy
    volumes.

    The autouse _reset_pil_max_image_pixels fixture in conftest.py
    guarantees the global is the package value when this test reads it.
    """
    pytest.importorskip("torch")  # vseg/utils.py module-top imports torch
    import PIL.Image

    import liom_toolkit.segmentation.vseg.utils  # noqa: F401 -- import triggers the module-top assignment

    assert PIL.Image.MAX_IMAGE_PIXELS is not None
    assert isinstance(PIL.Image.MAX_IMAGE_PIXELS, int)
    assert PIL.Image.MAX_IMAGE_PIXELS > 0
