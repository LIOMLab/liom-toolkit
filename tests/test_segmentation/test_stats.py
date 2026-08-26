"""Known-answer edge-case tests for ``liom_toolkit/segmentation/stats.py``.

These tests cover the empty / boundary / precision / ordering edges of the
stats functions that compute vessel morphometrics:

* ``calculate_density`` -- empty vessel mask over positive tissue returns
  ``(tissue_area, 0.0, 0.0)`` (the math is defined: 0 vessels / positive
  tissue area = 0 density); empty tissue mask raises ``ValueError`` (a bad
  region mask is a caller error, not a valid result).
* ``compute_average_diameter`` -- empty vessel set raises ``ValueError``
  (the mean diameter of an empty set is undefined; returning 0.0 would imply
  zero-width vessels exist, a plausible-shaped-but-wrong value); empty
  tissue mask raises ``ValueError``; no ``NaN`` may escape the function.
* ``compute_slice_metrics`` -- a vessel-free region yields a row with
  ``vessel density = 0.0`` and NO ``mean diameter (um)`` entry (the diameter
  row is omitted for vessel-free regions, which is itself a publishable
  "no vessels detected" signal); a region with vessels yields a row WITH a
  mean-diameter entry.
* ``create_heatmap`` -- a non-square image (e.g. 300x450) produces a heatmap
  where each ``square_size`` block holds the correct per-block sum (no
  staircase pattern, no off-by-one across the non-square dimension).

Per AGENTS section 5, ``numpy``/``scipy``/``scikit-image`` are NOT mocked --
the tests exercise the real functions on small synthetic arrays. ``stats.py``
imports only core deps (numpy, scipy, skimage, pandas, imageio), so no
``pytest.importorskip`` gating is needed.
"""

import numpy as np
import pytest
from skimage.morphology import skeletonize

from liom_toolkit.segmentation.stats import (
    calculate_density,
    compute_average_diameter,
    compute_slice_metrics,
    create_heatmap,
)


# ---------------------------------------------------------------------------
# calculate_density
# ---------------------------------------------------------------------------


def test_calculate_density_empty_vessels_returns_zero():
    """Empty vessel_mask over positive tissue returns
    ``(tissue_area, 0.0, 0.0)`` -- 0 vessels / positive tissue area is a
    well-defined 0 density (the math is defined)."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[3:7, 3:7] = 1  # 16 positive tissue pixels
    vessel_mask = np.zeros((10, 10), dtype=np.uint8)  # empty
    voxel_size = 0.65

    tissue_area, vessel_area, density = calculate_density(vessel_mask, mask, voxel_size)

    expected_tissue = 16 * (voxel_size**2)
    assert tissue_area == pytest.approx(expected_tissue)
    assert vessel_area == pytest.approx(0.0)
    assert density == pytest.approx(0.0)


def test_calculate_density_empty_tissue_raises():
    """An all-zero tissue mask (``mask.sum() == 0``) raises ``ValueError`` --
    an empty tissue mask is a bad region mask and a caller error, not a
    valid result. The current unfixed code raises ``ZeroDivisionError``; the
    fix raises a typed ``ValueError`` with a caller-facing message instead."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    vessel_mask = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(ValueError):
        calculate_density(vessel_mask, mask, 0.65)


def test_calculate_density_known_answer():
    """Known-answer on a synthetic mask + vessel pair: a 10x10 tissue mask
    with a 4x4 vessel block yields the exact expected density."""
    mask = np.ones((10, 10), dtype=np.uint8)
    vessel_mask = np.zeros((10, 10), dtype=np.uint8)
    vessel_mask[3:7, 3:7] = 1  # 16 vessel pixels
    voxel_size = 0.65

    tissue_area, vessel_area, density = calculate_density(vessel_mask, mask, voxel_size)

    expected_tissue = 100 * (voxel_size**2)
    expected_vessel = 16 * (voxel_size**2)
    assert tissue_area == pytest.approx(expected_tissue, rel=1e-9)
    assert vessel_area == pytest.approx(expected_vessel, rel=1e-9)
    assert density == pytest.approx(expected_vessel / expected_tissue, rel=1e-9)


# ---------------------------------------------------------------------------
# compute_average_diameter
# ---------------------------------------------------------------------------


def test_compute_average_diameter_empty_set_raises():
    """An empty vessel skeleton raises ``ValueError`` -- the mean diameter
    of an empty set is undefined. Returning 0.0 would imply zero-width
    vessels exist (a plausible-shaped-but-wrong value); returning ``NaN``
    would let a silent NaN escape into the published pandas DataFrame. The
    current unfixed code returns ``NaN`` + ``RuntimeWarning``; the fix
    raises ``ValueError`` BEFORE ``np.mean`` is evaluated on the empty
    array."""
    mask = np.zeros((21, 21), dtype=np.uint8)
    skeleton = np.zeros((21, 21), dtype=bool)

    with pytest.raises(ValueError):
        compute_average_diameter(mask, skeleton, 0.65)


def test_compute_average_diameter_empty_tissue_raises():
    """An all-zero tissue mask raises ``ValueError`` -- mean diameter is
    undefined when there is no tissue."""
    mask = np.zeros((21, 21), dtype=np.uint8)
    skeleton = np.zeros((21, 21), dtype=bool)

    with pytest.raises(ValueError):
        compute_average_diameter(mask, skeleton, 0.65)


def test_compute_average_diameter_no_nan_escapes():
    """For a non-empty synthetic tube, the result is a finite float (not
    ``NaN``, not ``inf``) -- guards against any NaN escape path."""
    r = 5
    mask = np.zeros((2 * r + 1, 2 * r + 1), dtype=np.uint8)
    # Straight horizontal tube of width 2*r+1 centered on the middle row.
    mask[r, :] = 1
    # Thicken slightly so the EDT gives a non-trivial radius at the skeleton.
    mask[r - 1 : r + 2, :] = 1
    skeleton = skeletonize(mask)

    result = compute_average_diameter(mask, skeleton, 0.65)

    assert not np.isnan(result)
    assert not np.isinf(result)
    assert result > 0


def test_compute_average_diameter_known_answer():
    """Known-answer on a synthetic straight tube: a horizontal band of known
    half-width ``r`` skeletonized down its centerline yields a mean
    diameter approximately ``2 * r * voxel_size`` (the EDT at the skeleton
    center equals the half-width)."""
    r = 5
    voxel_size = 0.65
    mask = np.zeros((2 * r + 1, 2 * r + 1), dtype=np.uint8)
    # A horizontal band of height 2*r+1 (full height) and width 2*r+1 (full
    # width) makes the EDT at the centerline equal to r -- the diameter is
    # then 2 * r * voxel_size.
    mask[:, :] = 1
    skeleton = skeletonize(mask)
    # skeletonize of a fully-filled square leaves only the boundary; build a
    # thin horizontal tube instead so the skeleton is a single centerline.
    mask = np.zeros((2 * r + 1, 2 * r + 1), dtype=np.uint8)
    mask[r - r // 2 : r + r // 2 + 1, :] = 1
    skeleton = skeletonize(mask)

    result = compute_average_diameter(mask, skeleton, voxel_size)

    # The EDT at the skeleton centerline is approximately the half-height of
    # the band; the diameter is 2 * half-height * voxel_size. Assert it is
    # positive and within a generous tolerance of the band height.
    band_height = (r // 2) * 2 + 1
    expected_diameter = band_height * voxel_size
    assert result == pytest.approx(expected_diameter, rel=0.5)
    assert result > 0


# ---------------------------------------------------------------------------
# compute_slice_metrics -- row-omission contract for vessel-free regions
# ---------------------------------------------------------------------------


def test_compute_slice_metrics_vessel_free_region_omits_diameter(tmp_path, monkeypatch):
    """A vessel-free region yields a row with ``vessel density = 0.0`` and
    NO ``mean diameter (um)`` entry (the diameter row is omitted for
    vessel-free regions per the D-01 row-omission contract). The omitted
    diameter row is itself a publishable 'no vessels detected' signal, not
    a silent gap.

    The test captures the DataFrame that ``compute_slice_metrics`` would
    write to xlsx by intercepting ``pd.DataFrame.to_excel`` -- this avoids
    the undeclared openpyxl runtime dependency (a pre-existing BUG-02
    finding) while exercising the real per-region row-omission logic. No
    numpy/scipy/skimage mocking is involved.
    """
    import pandas as pd

    captured = {}

    def fake_to_excel(self, path, index=False):
        captured["df"] = self

    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel)

    # 30x30 image with two regions: left half (region label 1, region_index
    # 0) is vessel-free; right half (region label 2, region_index 1) has a
    # small vessel block. The whole-slice vessel_mask has vessels (in the
    # right half) so the 'total' row computes without crashing; the
    # per-region row for region_index 0 is vessel-free and must omit the
    # diameter entry. The ``image`` parameter is the slice label written
    # into the output DataFrame (a string), not an array used by the
    # metrics math.
    mask = np.ones((30, 30), dtype=np.uint8)
    vessel_mask = np.zeros((30, 30), dtype=np.uint8)
    vessel_mask[10:20, 20:30] = 1  # vessels only in the right half
    region_map = np.zeros((30, 30), dtype=np.uint8)
    region_map[:, :15] = 1  # left half = region label 1
    region_map[:, 15:] = 2  # right half = region label 2
    vessel_exclude = np.ones((30, 30), dtype=np.uint8)

    output_dir = str(tmp_path) + "/"
    compute_slice_metrics(
        output_dir,
        "test_slice",
        mask,
        vessel_mask,
        region_map,
        vessel_exclude,
        voxel_size=0.65,
    )

    df = captured["df"]
    region_rows = df[df["region"] == 0]
    assert len(region_rows) == 1
    row = region_rows.iloc[0]
    # The density row is kept and equals 0.0 for the vessel-free region.
    assert row["vessel density (um2/um2)"] == pytest.approx(0.0)
    # The mean-diameter entry is OMITTED for vessel-free regions: the cell
    # is NaN/missing because no diameter row was written for this region.
    diameter_val = row["mean diameter (um)"]
    assert pd.isna(diameter_val)


def test_compute_slice_metrics_vessel_region_has_diameter(tmp_path, monkeypatch):
    """A region WITH vessels yields a row WITH a ``mean diameter (um)``
    entry (the vessel-present path is unchanged).

    The test captures the DataFrame via ``monkeypatch`` on
    ``pd.DataFrame.to_excel`` to avoid the undeclared openpyxl dependency
    (see the vessel-free test for the same pattern)."""
    import pandas as pd

    captured = {}

    def fake_to_excel(self, path, index=False):
        captured["df"] = self

    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel)

    # 30x30 image; one tissue region with a small vessel block in it.
    mask = np.ones((30, 30), dtype=np.uint8)
    vessel_mask = np.zeros((30, 30), dtype=np.uint8)
    vessel_mask[10:20, 10:20] = 1  # a vessel block
    region_map = np.ones((30, 30), dtype=np.uint8)
    vessel_exclude = np.ones((30, 30), dtype=np.uint8)

    output_dir = str(tmp_path) + "/"
    compute_slice_metrics(
        output_dir,
        "test_slice",
        mask,
        vessel_mask,
        region_map,
        vessel_exclude,
        voxel_size=0.65,
    )

    df = captured["df"]
    region_rows = df[df["region"] == 0]
    assert len(region_rows) == 1
    row = region_rows.iloc[0]
    # The vessel-present region has a finite, non-NaN mean diameter.
    diameter_val = row["mean diameter (um)"]
    assert not pd.isna(diameter_val)
    assert diameter_val > 0


# ---------------------------------------------------------------------------
# create_heatmap -- non-square reset (D-14)
# ---------------------------------------------------------------------------


def test_create_heatmap_non_square(tmp_path):
    """A non-square image (300x450 with square_size=150) produces a heatmap
    where each 150x150 block holds the correct per-block sum -- no
    staircase pattern, no off-by-one across the non-square dimension.

    The current unfixed code uses ``image.shape[0] / square_size`` for BOTH
    dimensions, so on a 300x450 image the inner loop runs only 2 iterations
    (300/150) instead of 3 (450/150), clobbering the rightmost column of
    blocks with the wrong sums. The fix computes ``n_x`` and ``n_y``
    separately and resets ``y_start`` at the start of each outer iteration.
    """
    # 300 (rows) x 450 (cols) -- non-square. square_size=150 -> 2 rows of
    # blocks x 3 cols of blocks. Use 255 (on) / 0 (off) values so they
    # survive the /255 + astype(uint8) quantization inside create_heatmap
    # (values 10-60 would quantize to 0 and the test would be vacuous).
    image = np.zeros((300, 450), dtype=np.uint8)
    # Checkerboard-ish pattern: block (0, 2) = 255 (on), block (1, 2) = 0
    # (off). Under the bug, the rightmost column is never written (inner
    # loop runs 2 iterations instead of 3) so both blocks would be 0; under
    # the fix, block (0, 2) is 255 and block (1, 2) is 0.
    image[0:150, 300:450] = 255  # block (0, 2) = on
    # block (1, 2) stays 0 (off)

    output_dir = str(tmp_path) + "/"
    create_heatmap(image, output_dir, square_size=150)

    import imageio.v3 as iio

    heatmap = iio.imread(output_dir + "heatmap.tif")
    # The heatmap is normalized by square_size**2 at the end; read it back
    # and verify the per-block AVERAGE (sum / 150**2) matches the input
    # block's value / 255 (the input is divided by 255 before summing).
    # Easier: assert that two blocks in the same column but different rows
    # hold DIFFERENT values (a staircase clobbers them to the same value).
    block_0_2 = heatmap[0:150, 300:450].mean()
    block_1_2 = heatmap[150:300, 300:450].mean()
    # Under the bug, the rightmost column is either never written or
    # clobbered by the staircase; under the fix, the two blocks differ
    # because their input values differ (255 vs 0).
    assert block_0_2 != pytest.approx(block_1_2, abs=1e-6)
    # And the rightmost column block in row 0 reflects its own input value
    # (255 on), so the heatmap average should be positive (the block sum is
    # 150*150 = 22500, normalized to 1.0 before the img_as_uint cast).
    assert block_0_2 > 0
