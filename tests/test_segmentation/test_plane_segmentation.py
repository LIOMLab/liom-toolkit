"""Known-answer tests for ``liom_toolkit/segmentation/plane_segmentation.py``.

These tests close the Wave-0 test gap identified for DEP-01: before Phase 3,
``test_plane_segmentation.py`` did not exist, so the skimage 0.26 renames
(``binary_erosion`` -> ``erosion``, ``remove_small_objects(min_size=)`` ->
``max_size=``) and the ``imageio.v3`` write call sites inside
``segment_2d_image`` had no automated regression coverage.

Coverage:

* ``erode_mask`` — proves the post-rename ``erosion(...)`` call shrinks the
  foreground and never grows it (every True pixel in the output is also True
  in the input).
* ``morphology.remove_small_objects`` with ``max_size=`` — proves the rename
  preserves the intended "remove small objects" semantics: a large blob
  survives and a small blob is removed.
* ``segment_2d_image`` — end-to-end run on the ``bimodal_2d`` fixture; proves
  the full pipeline (Frangi -> threshold -> ``remove_small_objects`` ->
  ``imageio.v3.imwrite``) writes both output files without raising.

Per AGENTS section 5, ``numpy``/``scipy``/``scikit-image`` are NOT mocked --
the tests exercise the real image-processing functions on small synthetic
arrays. ``plane_segmentation.py`` imports only core deps (scipy, skimage,
imageio), so no ``pytest.importorskip`` gating is needed.
"""

import numpy as np
import pytest
from skimage import morphology

from liom_toolkit.segmentation.plane_segmentation import erode_mask, segment_2d_image


def test_erode_mask_shrinks_foreground():
    """erode_mask returns a strictly smaller foreground that is a subset of
    the input -- erosion only shrinks, never grows."""
    mask = np.zeros((32, 32), dtype=bool)
    mask[6:26, 6:26] = True  # 20x20 True square centered in a 32x32 field

    eroded = erode_mask(mask, disk_size=2)

    # Foreground pixel count strictly decreases under erosion.
    assert eroded.sum() < mask.sum()
    # Erosion only shrinks: every True pixel in the output is also True in
    # the input (the eroded set is a subset of the original set).
    assert np.all(eroded[mask == False] == False)


def test_remove_small_objects_max_size_removes_small_blobs():
    """remove_small_objects(arr, max_size=50) keeps the large blob and
    removes the small blob -- proves the max_size= rename preserves the
    'remove small objects' semantics."""
    arr = np.zeros((32, 32), dtype=bool)
    arr[2:12, 2:12] = True  # 10x10 = 100 px blob
    arr[20:22, 20:22] = True  # 2x2 = 4 px blob

    cleaned = morphology.remove_small_objects(arr, max_size=50)

    # The 100 px blob survives (it is larger than max_size).
    assert cleaned[2:12, 2:12].any()
    # The 4 px blob is removed (it is smaller than max_size).
    assert not cleaned[20:22, 20:22].any()


def test_segment_2d_image_runs_end_to_end(tmp_path, bimodal_2d):
    """segment_2d_image writes both output files (mask + vessel_mask) to
    disk via the imageio.v3 call sites, proving the full pipeline runs
    without raising."""
    output_dir = str(tmp_path) + "/"

    segment_2d_image(output_dir, bimodal_2d, "test", local_threshold=False)

    mask_path = tmp_path / "test_mask.tif"
    vessel_mask_path = tmp_path / "test_vessel_mask.tif"
    assert mask_path.exists()
    assert vessel_mask_path.exists()


def test_segment_2d_image_even_threshold_size_raises(tmp_path, bimodal_2d):
    """segment_2d_image rejects an even local_threshold_size with ValueError
    (not AssertionError, not silent under python -O).

    The validation guard is an ``if local_threshold_size % 2 == 0: raise
    ValueError(...)`` form (converted from the prior ``assert
    local_threshold_size % 2 == 1`` so it survives optimized runs). The
    f-string includes the offending value so the error is actionable.
    """
    output_dir = str(tmp_path) + "/"

    with pytest.raises(ValueError):
        segment_2d_image(
            output_dir,
            bimodal_2d,
            "test",
            local_threshold=True,
            local_threshold_size=2,
        )
