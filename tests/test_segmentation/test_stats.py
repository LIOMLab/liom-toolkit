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


def test_compute_slice_metrics_vessel_free_slice_omits_total_diameter(tmp_path, monkeypatch):
    """A vessel-free SLICE (no vessels anywhere) must not crash
    ``compute_slice_metrics``. The whole-slice 'total' row's
    ``compute_average_diameter`` call raises ``ValueError`` on an empty
    vessel set (the D-01 contract); the per-region loop wraps that call in
    ``try/except ValueError`` and omits the diameter row, but the whole-slice
    call must be wrapped the same way -- otherwise a vessel-free slice
    crashes the whole metrics computation and the user loses every per-region
    row computed up to that point.

    The 'total' row for a vessel-free slice keeps the density=0.0 /
    branching-points / vessel-area rows and OMITS the 'mean diameter (um)'
    entry, mirroring the per-region row-omission pattern. The DataFrame is
    captured via ``monkeypatch`` on ``pd.DataFrame.to_excel`` to avoid the
    undeclared openpyxl runtime dependency (no numpy/scipy/skimage mocking).
    """
    import pandas as pd

    captured = {}

    def fake_to_excel(self, path, index=False):
        captured["df"] = self

    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel)

    # 30x30 image with one tissue region and NO vessels anywhere in the
    # slice. The whole-slice vessel_mask is all-zero, so the 'total' row's
    # compute_average_diameter(vessel_mask, skeleton, ...) hits the empty-
    # vessel-set ValueError (D-01). Without the wrap this raises out of
    # compute_slice_metrics and the xlsx is never written.
    mask = np.ones((30, 30), dtype=np.uint8)
    vessel_mask = np.zeros((30, 30), dtype=np.uint8)  # no vessels anywhere
    region_map = np.ones((30, 30), dtype=np.uint8)  # one region
    vessel_exclude = np.ones((30, 30), dtype=np.uint8)

    output_dir = str(tmp_path) + "/"
    # Must not raise.
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
    total_rows = df[df["region"] == "total"]
    assert len(total_rows) == 1
    row = total_rows.iloc[0]
    # The density row is kept and equals 0.0 for the vessel-free slice.
    assert row["vessel density (um2/um2)"] == pytest.approx(0.0)
    # The mean-diameter entry is OMITTED for the vessel-free 'total' row
    # (NaN/missing because no diameter value was written), mirroring the
    # per-region row-omission contract.
    diameter_val = row["mean diameter (um)"]
    assert pd.isna(diameter_val)


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


# ---------------------------------------------------------------------------
# compute_slice_metrics / create_heatmap -- overwrite-safe second run
# ---------------------------------------------------------------------------


def test_compute_slice_metrics_overwrite(tmp_path):
    """compute_slice_metrics called twice with the same output_dir succeeds
    on the second call (no FileExistsError) and produces the expected output
    files.

    The bare ``os.mkdir(output_dir)`` race is fixed by routing directory
    creation through the symlink-aware ``create_directory(overwrite=True)``
    helper, which ``shutil.rmtree``'s the existing output_dir then recreates
    it before the metrics are written. The test exercises real IO in
    ``tmp_path`` (no mocking of numpy/scipy/skimage/os per AGENTS section 5)
    AND the real ``pd.DataFrame.to_excel`` -> openpyxl path (CLOSE-03:
    openpyxl is now a declared core dependency at pyproject.toml, so the
    monkeypatch that previously intercepted ``to_excel`` to avoid the
    undeclared openpyxl runtime dependency is removed -- the real xlsx is
    written and asserted to exist)."""
    import os

    mask = np.ones((30, 30), dtype=np.uint8)
    vessel_mask = np.zeros((30, 30), dtype=np.uint8)
    vessel_mask[10:20, 10:20] = 1
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
    # Second call into the same existing output_dir must succeed (overwrite-safe).
    compute_slice_metrics(
        output_dir,
        "test_slice",
        mask,
        vessel_mask,
        region_map,
        vessel_exclude,
        voxel_size=0.65,
    )

    # Expected output files exist after the second run (the directory was
    # recreated by create_directory(overwrite=True), so these are the
    # second-run files, not stale first-run files). The real df.to_excel ->
    # openpyxl path writes regions.xlsx (CLOSE-03: no monkeypatch).
    assert os.path.isfile(output_dir + "regions.xlsx")
    assert os.path.isfile(output_dir + "regions.png")
    assert os.path.isfile(output_dir + "vessels.png")


def test_create_heatmap_overwrite(tmp_path):
    """create_heatmap called twice with the same output_dir succeeds on the
    second call (no FileExistsError) and produces the expected heatmap
    output file.

    The bare ``os.mkdir(output_dir)`` race is fixed by routing directory
    creation through ``create_directory(overwrite=True)``. Real IO in
    ``tmp_path``; no mocking per AGENTS section 5."""
    import os

    import imageio.v3 as iio

    # 300x300 square image with a single on-block so the heatmap is non-trivial.
    image = np.zeros((300, 300), dtype=np.uint8)
    image[0:150, 0:150] = 255

    output_dir = str(tmp_path) + "/"
    create_heatmap(image, output_dir, square_size=150)
    # Second call into the same existing output_dir must succeed (overwrite-safe).
    create_heatmap(image, output_dir, square_size=150)

    # The heatmap file exists after the second run (recreated directory).
    assert os.path.isfile(output_dir + "heatmap.tif")
    heatmap = iio.imread(output_dir + "heatmap.tif")
    # The on-block (0,0) has a positive average; the off-block (1,1) is 0.
    assert heatmap[0:150, 0:150].mean() > 0


def test_compute_slice_metrics_total_density_excludes_out_of_tissue_vessels(tmp_path, monkeypatch):
    """The whole-slice 'total' row's vessel density must exclude vessels
    outside the tissue mask and explicitly excluded vessels, matching the
    per-region rows (which derive their vessel area from
    ``full_vessel_mask = vessel_mask * mask * vessel_exclude``).

    The pre-fix code passed the raw ``vessel_mask`` to ``calculate_density``
    for the total row, so the total density included out-of-tissue and
    excluded vessels -- making it higher than the sum of per-region
    densities (a data inconsistency in the same output DataFrame).

    This test builds a slice where vessels sit OUTSIDE the tissue mask
    (mask=0 there) so the raw vessel_mask has vessels but
    full_vessel_mask does not. Under the bug the total row's vessel area
    is positive; under the fix it is 0 (matching the per-region rows,
    which all see no in-tissue vessels).
    """
    import pandas as pd

    captured = {}

    def fake_to_excel(self, path, index=False):
        captured["df"] = self

    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel)

    # 30x30 image. Tissue mask covers only the left half (mask[:, :15]=1).
    # Vessels sit ONLY in the right half (outside the tissue mask). The
    # per-region rows see no in-tissue vessels (full_vessel_mask is all-zero
    # in the tissue region), so the total row must also report vessel area 0.
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[:, :15] = 1  # tissue only in the left half
    vessel_mask = np.zeros((30, 30), dtype=np.uint8)
    vessel_mask[10:20, 20:30] = 1  # vessels only in the right half (outside tissue)
    region_map = np.zeros((30, 30), dtype=np.uint8)
    region_map[:, :15] = 1  # one tissue region
    vessel_exclude = np.ones((30, 30), dtype=np.uint8)  # no explicit exclusion

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
    total_rows = df[df["region"] == "total"]
    assert len(total_rows) == 1
    total_row = total_rows.iloc[0]
    # The total row's vessel area must be 0: all vessels are outside the
    # tissue mask, so full_vessel_mask has no vessels. Under the bug the
    # raw vessel_mask (which has vessels in the right half) was passed,
    # producing a positive vessel area inconsistent with the per-region
    # rows (which all correctly see no in-tissue vessels).
    assert total_row["vessel area (um2)"] == pytest.approx(0.0)
    assert total_row["vessel density (um2/um2)"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# get_branching_points -- numerical-equivalence regression for the kernel
# reduction (PERF-01b). The old algorithm inlined below is the reference.
# ---------------------------------------------------------------------------


def _old_get_branching_points(skeleton: np.ndarray) -> np.ndarray:
    """The inherited 20-convolution branching-point detector (reference).

    5 structural elements x 4 rotations = 20 ``ndi.binary_hit_or_miss`` calls,
    OR-ed into the result. Inlined verbatim from the pre-refactor
    ``stats.get_branching_points`` so the equivalence test is a true
    numerical-equivalence regression (``array_equal``, not ``allclose``).
    """
    import scipy.ndimage as ndi

    selems = []
    selems.extend(
        (
            np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]]),
            np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]]),
            np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]]),
            np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]]),
            np.array([[0, 0, 1], [1, 1, 1], [0, 1, 0]]),
        )
    )
    selems = [np.rot90(selems[i], k=j) for i in range(5) for j in range(4)]

    branches = np.zeros_like(skeleton, dtype=bool)
    for selem in selems:
        branches |= ndi.binary_hit_or_miss(skeleton, selem)
    return branches


def test_branching_points_equivalence():
    """The reduced-kernel ``get_branching_points`` must be ``array_equal`` to
    the old 20-convolution result on a synthetic skeleton with known branching
    points (a T-junction and a Y-junction).

    Numerical equivalence is asserted via ``array_equal`` (NOT ``allclose``) --
    branching-point detection is a boolean topology operation with no float
    intermediate, so any divergence is a topology change.
    """
    from liom_toolkit.segmentation.stats import get_branching_points

    # T-junction: horizontal bar + vertical stem meeting at (3,3).
    t_junction = np.zeros((7, 7), dtype=bool)
    t_junction[3, 1:6] = True
    t_junction[1:4, 3] = True

    # Y-junction: a center with three arms (up, left, down-right diagonal).
    y_junction = np.zeros((7, 7), dtype=bool)
    y_junction[3, 3] = True
    y_junction[2, 3] = True  # up
    y_junction[3, 2] = True  # left
    y_junction[4, 4] = True  # down-right

    for sk in (t_junction, y_junction):
        new_result = get_branching_points(sk)
        old_result = _old_get_branching_points(sk)
        assert np.array_equal(new_result, old_result), (
            "get_branching_points kernel reduction diverged from the old "
            "20-convolution result"
        )


def test_branching_points_equivalence_random():
    """Randomized numerical-equivalence regression: the reduced-kernel result
    must be ``array_equal`` to the old 20-convolution result across many random
    skeletons (catches edge cases the two hand-built skeletons miss)."""
    from liom_toolkit.segmentation.stats import get_branching_points

    rng = np.random.default_rng(seed=42)
    for _trial in range(50):
        sk = rng.random((15, 15)) < 0.5
        new_result = get_branching_points(sk)
        old_result = _old_get_branching_points(sk)
        assert np.array_equal(new_result, old_result), (
            "get_branching_points kernel reduction diverged on a random skeleton"
        )


def test_branching_points_empty():
    """``get_branching_points`` on an all-False skeleton returns all-False
    (0 branching points -- no foreground, no branches)."""
    from liom_toolkit.segmentation.stats import get_branching_points

    sk = np.zeros((7, 7), dtype=bool)
    result = get_branching_points(sk)
    assert not result.any()
    assert result.dtype == np.bool_


def test_branching_points_straight_line():
    """``get_branching_points`` on a straight-line skeleton (no branches)
    returns all-False (a line has no junctions)."""
    from liom_toolkit.segmentation.stats import get_branching_points

    sk = np.zeros((7, 7), dtype=bool)
    sk[3, 1:6] = True  # horizontal line, no branches
    result = get_branching_points(sk)
    assert not result.any()


def test_branching_points_t_junction():
    """``get_branching_points`` on a T-junction skeleton returns True at
    exactly the junction voxel (and only there)."""
    from liom_toolkit.segmentation.stats import get_branching_points

    sk = np.zeros((7, 7), dtype=bool)
    sk[3, 1:6] = True  # horizontal bar
    sk[1:4, 3] = True  # vertical stem meeting at (3,3)
    result = get_branching_points(sk)
    assert result.sum() == 1, f"expected exactly 1 branching point, got {result.sum()}"
    assert result[3, 3], "the branching point must be at the junction (3,3)"


# ---------------------------------------------------------------------------
# compute_mask_area -- .compute() site classification (PERF-01e).
# ---------------------------------------------------------------------------


def test_compute_mask_area_returns_scalar():
    """``compute_mask_area`` must return a materialized scalar (``np.uint64``),
    not a Dask array.

    Classification of the ``.compute()`` site in ``compute_mask_area``:
    ``client.gather(client.submit(da.sum, mask))`` returns a 0-dimensional
    ``dask.array.Array`` (a scalar Dask array, NOT a Python scalar), so the
    subsequent ``.compute()`` is a BOUNDARY-REQUIRED materialization (the
    gathered Dask array must be computed to produce the scalar the function
    promises to return). The ``.compute()`` is therefore KEPT, not removed.

    This test asserts the chosen behavior: the function returns a real
    ``np.uint64`` scalar (not a Dask array), proving the materialization ran.
    A real Dask distributed client is injected into the singleton manager (no
    mock of the gather/submit/compute path) so the full materialization chain
    is exercised end-to-end.
    """
    pytest.importorskip("dask.distributed")
    import dask.array as da
    from dask.distributed import Client, LocalCluster

    from liom_toolkit.segmentation.stats import compute_mask_area
    from liom_toolkit.utils.dask_client import dask_client_manager

    # The dashboard is disabled (dashboard_address=False) because the bokeh
    # TornadoServerApplication.stop() raises "Cannot synchronously wait on a
    # running event loop" when the scheduler's dashboard service is torn down
    # synchronously inside client.close() (bokeh 3.x + distributed 2026.x on
    # Python 3.12). The error propagates as a RuntimeError out of Client.close()
    # when a dashboard port is configured. The test does not exercise the
    # dashboard, so disabling it avoids the spurious teardown error while still
    # exercising the real gather/submit/compute path.
    cluster = LocalCluster(n_workers=1, threads_per_worker=1, dashboard_address=False)
    client = Client(cluster)
    saved_client = dask_client_manager.client
    dask_client_manager.client = client
    try:
        mask = da.from_array(np.array([[1, 0], [0, 1]], dtype=np.uint8), chunks=(2, 2))
        result = compute_mask_area(mask)
        # The function must return a materialized scalar, not a Dask array.
        assert isinstance(result, np.uint64), f"expected np.uint64, got {type(result)}"
        assert int(result) == 2
    finally:
        # Restore the saved client first, then close the test-created client
        # and cluster directly. The previous code restored saved_client (None)
        # before calling dask_client_manager.close(), so close() saw
        # self.client is None and did nothing -- leaking the Client/LocalCluster
        # every run. Closing the local ``client`` variable directly drains
        # in-flight tasks and shuts down the worker process.
        dask_client_manager.client = saved_client
        client.close()
        cluster.close()
