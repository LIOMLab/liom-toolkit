"""Known-answer tests for the vessel-segmentation ship-gate eval metrics.

The ship gate is a per-metric matrix (not a composite score): each of the six
metrics below is an independent pass/fail row that surfaces a distinct
silent-wrong-data failure mode that aggregate Dice hides. These tests verify
the metric implementations on small synthetic masks using the established
known-answer pattern (direct import + call + plain assert, no mocking, no I/O).

Coverage:

* ``centerline_recall`` — recall is 1.0 when the prediction covers the GT
  centerline, < 1.0 when a break is introduced; raises on empty GT / empty
  skeleton.
* ``caliber_stratified_recall`` — per-diameter-bin recall (capillary vs large
  vessel) matches expected values when both vessels are covered and when only
  the large vessel is covered; raises on empty GT / empty skeleton / no
  positive radii.
* ``boundary_artifact_regression`` — a deliberate seam drops boundary quality
  below interior quality (positive regression delta); a perfect prediction
  yields a zero delta; raises on empty input.
* ``spurious_thin_vessel_rate`` — injected false-positive thin vessels produce
  the expected count per unit volume; raises on empty input (no silent 0.0).
* ``fpr_on_empty`` — false-positive rate on GT-empty regions matches the
  injected fraction; raises on empty GT (no divide-by-zero NaN).
* ``cl_dice_metric`` — delegates to ``cldice.cl_dice``; returns 1.0 on a
  perfect match, 0.0 on both-empty skeletons, raises on non-2D/3D input.
* ``reported_dice`` — aggregate Dice via the pure-NumPy fallback (no torch
  required); the MONAI ``DiceMetric`` path is exercised in a torch-gated test
  that skips cleanly without torch/monai.

The pure-NumPy metric tests require NO torch — only the ``reported_dice``
MONAI test uses ``pytest.importorskip`` at the first line of its body (never
at module top, per the pytest #9542 importorskip-at-module-top pitfall).
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# centerline_recall
# ---------------------------------------------------------------------------


def test_centerline_recall_known_answer_perfect():
    """centerline_recall is 1.0 when the prediction covers the GT centerline.

    A 3-voxel-wide horizontal vessel skeletonises to a 1-voxel centreline;
    a prediction equal to the GT covers every centreline voxel, so the
    skeleton-intersection ratio (tprec) is exactly 1.0.
    """
    from liom_toolkit.segmentation.vseg.eval_metrics import centerline_recall

    gt = np.zeros((10, 20), dtype=bool)
    gt[3:6, :] = True  # 3-wide vessel
    predicted = gt.copy()

    assert centerline_recall(predicted, gt) == 1.0


def test_centerline_recall_known_answer_break():
    """centerline_recall drops below 1.0 when a break is introduced.

    A 1-voxel-wide horizontal line is its own skeleton (20 centreline voxels,
    no endpoint removal). Removing two columns from the prediction breaks the
    centreline coverage without costing much area overlap — the exact failure
    mode Dice hides and centreline recall exposes.
    """
    from liom_toolkit.segmentation.vseg.eval_metrics import centerline_recall

    gt = np.zeros((10, 20), dtype=bool)
    gt[5, :] = True  # 1-wide vessel, skeleton = 20 centreline voxels
    predicted = gt.copy()
    predicted[5, 9:11] = False  # break the centreline at 2 voxels

    # 18 of 20 centreline voxels covered -> 0.9
    assert centerline_recall(predicted, gt) == pytest.approx(0.9, abs=1e-6)


def test_centerline_recall_empty_gt_raises():
    """centerline_recall raises ValueError on empty GT (no centreline to recall).

    Empty GT means there is no centreline to recall — returning 0.0 would be a
    silent wrong-data fallback (AGENTS §2). The raise must precede any
    skeletonisation or division.
    """
    from liom_toolkit.segmentation.vseg.eval_metrics import centerline_recall

    gt = np.zeros((10, 20), dtype=bool)
    predicted = np.zeros((10, 20), dtype=bool)

    with pytest.raises(ValueError, match="empty"):
        centerline_recall(predicted, gt)


def test_centerline_recall_all_zero_raises():
    """centerline_recall raises ValueError on an all-zero prediction + GT pair."""
    from liom_toolkit.segmentation.vseg.eval_metrics import centerline_recall

    gt = np.zeros((10, 20), dtype=bool)
    predicted = np.zeros((10, 20), dtype=bool)

    with pytest.raises(ValueError):
        centerline_recall(predicted, gt)


# ---------------------------------------------------------------------------
# caliber_stratified_recall
# ---------------------------------------------------------------------------


def _caliber_fixture():
    """Build a GT with a small (capillary) and a large (arteriole) vessel.

    Uses ``voxel_size_um=1.0`` and ``capillary_radius_um=3.0`` so the
    distance-transform radii bin cleanly: the small circle (EDT radius ~2.24)
    falls in the capillary bin, the large circle (EDT radius ~5.10) in the
    large-vessel bin. The two circle skeletons are single centre points, so
    there are no endpoint-radius edge cases.
    """
    gt = np.zeros((40, 40), dtype=bool)
    yy, xx = np.mgrid[0:40, 0:40]
    gt[((yy - 10) ** 2 + (xx - 10) ** 2) <= 4] = True  # small circle, r=2
    gt[((yy - 28) ** 2 + (xx - 28) ** 2) <= 25] = True  # large circle, r=5
    return gt


def test_caliber_stratified_recall_known_answer_both_covered():
    """Both bins report recall 1.0 when the prediction covers both vessels."""
    from liom_toolkit.segmentation.vseg.eval_metrics import (
        caliber_stratified_recall,
    )

    gt = _caliber_fixture()
    predicted = gt.copy()

    result = caliber_stratified_recall(predicted, gt, voxel_size_um=1.0, capillary_radius_um=3.0)

    assert set(result) == {"capillary_recall", "large_vessel_recall"}
    assert result["capillary_recall"] == pytest.approx(1.0, abs=1e-6)
    assert result["large_vessel_recall"] == pytest.approx(1.0, abs=1e-6)


def test_caliber_stratified_recall_known_answer_capillary_missed():
    """Capillary recall is 0.0 while large-vessel recall is 1.0 when only the
    large vessel is covered — the "Dice 0.85 while missing every capillary"
    failure mode the metric exists to expose."""
    from liom_toolkit.segmentation.vseg.eval_metrics import (
        caliber_stratified_recall,
    )

    gt = _caliber_fixture()
    # Predict only the large vessel (miss the capillary entirely).
    yy, xx = np.mgrid[0:40, 0:40]
    predicted = (((yy - 28) ** 2 + (xx - 28) ** 2) <= 25).astype(bool)

    result = caliber_stratified_recall(predicted, gt, voxel_size_um=1.0, capillary_radius_um=3.0)

    assert result["capillary_recall"] == pytest.approx(0.0, abs=1e-6)
    assert result["large_vessel_recall"] == pytest.approx(1.0, abs=1e-6)


def test_caliber_stratified_recall_empty_gt_raises():
    """caliber_stratified_recall raises ValueError on empty GT."""
    from liom_toolkit.segmentation.vseg.eval_metrics import (
        caliber_stratified_recall,
    )

    gt = np.zeros((20, 20), dtype=bool)
    predicted = np.zeros((20, 20), dtype=bool)

    with pytest.raises(ValueError, match="empty"):
        caliber_stratified_recall(predicted, gt)


def test_caliber_stratified_recall_all_zero_raises():
    """caliber_stratified_recall raises ValueError on all-zero input."""
    from liom_toolkit.segmentation.vseg.eval_metrics import (
        caliber_stratified_recall,
    )

    gt = np.zeros((20, 20), dtype=bool)
    predicted = np.zeros((20, 20), dtype=bool)

    with pytest.raises(ValueError):
        caliber_stratified_recall(predicted, gt)


# ---------------------------------------------------------------------------
# boundary_artifact_regression
# ---------------------------------------------------------------------------


def test_boundary_artifact_regression_known_answer_seam():
    """A deliberate seam at a patch boundary drops boundary quality below
    interior quality, yielding a positive regression delta.

    A horizontal vessel crossing the vertical patch seam (x=256) is severed in
    the prediction at the seam; the boundary strip (outer ring of each patch)
    contains the break while the patch interiors remain intact.
    """
    from liom_toolkit.segmentation.vseg.eval_metrics import (
        boundary_artifact_regression,
    )

    gt = np.zeros((512, 512), dtype=bool)
    gt[100, :] = True  # horizontal vessel crossing the seam at x=256
    predicted = gt.copy()
    predicted[100, 254:259] = False  # gap at the seam

    result = boundary_artifact_regression(predicted, gt, patch_size=(256, 256))

    assert set(result) == {
        "boundary_quality",
        "interior_quality",
        "regression_delta",
    }
    assert result["boundary_quality"] < result["interior_quality"]
    assert result["regression_delta"] > 0.0


def test_boundary_artifact_regression_known_answer_perfect():
    """A perfect prediction yields equal boundary and interior quality and a
    zero regression delta — no patch-grid seam artefact."""
    from liom_toolkit.segmentation.vseg.eval_metrics import (
        boundary_artifact_regression,
    )

    gt = np.zeros((512, 512), dtype=bool)
    gt[100, :] = True
    predicted = gt.copy()

    result = boundary_artifact_regression(predicted, gt, patch_size=(256, 256))

    assert result["boundary_quality"] == pytest.approx(1.0, abs=1e-6)
    assert result["interior_quality"] == pytest.approx(1.0, abs=1e-6)
    assert result["regression_delta"] == pytest.approx(0.0, abs=1e-6)


def test_boundary_artifact_regression_empty_input_raises():
    """boundary_artifact_regression raises ValueError on empty input."""
    from liom_toolkit.segmentation.vseg.eval_metrics import (
        boundary_artifact_regression,
    )

    gt = np.zeros((256, 256), dtype=bool)
    predicted = np.zeros((256, 256), dtype=bool)

    with pytest.raises(ValueError, match="empty"):
        boundary_artifact_regression(predicted, gt, patch_size=(128, 128))


def test_boundary_artifact_regression_all_zero_raises():
    """boundary_artifact_regression raises ValueError on all-zero input."""
    from liom_toolkit.segmentation.vseg.eval_metrics import (
        boundary_artifact_regression,
    )

    gt = np.zeros((256, 256), dtype=bool)
    predicted = np.zeros((256, 256), dtype=bool)

    with pytest.raises(ValueError):
        boundary_artifact_regression(predicted, gt, patch_size=(128, 128))


# ---------------------------------------------------------------------------
# spurious_thin_vessel_rate
# ---------------------------------------------------------------------------


def test_spurious_thin_vessel_rate_known_answer():
    """Injected false-positive thin vessels produce the expected rate.

    A 5-voxel-wide GT vessel (correctly predicted) plus an 11-voxel
    false-positive 1-voxel-wide vessel elsewhere yields a spurious count of
    11 voxels over a 50x50 image (rate = 11/2500). The thick vessel's edge
    voxels are excluded because they fall within the dilated GT skeleton.
    """
    from liom_toolkit.segmentation.vseg.eval_metrics import (
        spurious_thin_vessel_rate,
    )

    gt = np.zeros((50, 50), dtype=bool)
    gt[23:28, 10:40] = True  # 5-wide thick vessel
    predicted = gt.copy()
    predicted[5, 10:21] = True  # 11-voxel false-positive thin vessel

    rate = spurious_thin_vessel_rate(predicted, gt, voxel_size_um=1.0, capillary_radius_um=1.5)

    assert rate == pytest.approx(11 / 2500, abs=1e-9)


def test_spurious_thin_vessel_rate_empty_input_raises():
    """spurious_thin_vessel_rate raises ValueError on an empty prediction
    (no silent 0.0 — the rate of an empty prediction is undefined)."""
    from liom_toolkit.segmentation.vseg.eval_metrics import (
        spurious_thin_vessel_rate,
    )

    gt = np.zeros((50, 50), dtype=bool)
    gt[25, 10:40] = True
    predicted = np.zeros((50, 50), dtype=bool)

    with pytest.raises(ValueError, match="empty"):
        spurious_thin_vessel_rate(predicted, gt)


def test_spurious_thin_vessel_rate_all_zero_raises():
    """spurious_thin_vessel_rate raises ValueError on all-zero input."""
    from liom_toolkit.segmentation.vseg.eval_metrics import (
        spurious_thin_vessel_rate,
    )

    gt = np.zeros((50, 50), dtype=bool)
    predicted = np.zeros((50, 50), dtype=bool)

    with pytest.raises(ValueError):
        spurious_thin_vessel_rate(predicted, gt)


# ---------------------------------------------------------------------------
# fpr_on_empty
# ---------------------------------------------------------------------------


def test_fpr_on_empty_known_answer():
    """FPR on GT-empty regions matches the injected false-positive fraction.

    A 10x10 GT with a single vessel voxel leaves 99 empty voxels; two
    false-positive vessel voxels in the empty region yield FPR = 2/99.
    """
    from liom_toolkit.segmentation.vseg.eval_metrics import fpr_on_empty

    gt = np.zeros((10, 10), dtype=bool)
    gt[5, 5] = True  # one GT vessel voxel -> 99 empty voxels
    predicted = gt.copy()
    predicted[1, 1] = True  # false positive in empty region
    predicted[2, 2] = True  # false positive in empty region

    assert fpr_on_empty(predicted, gt) == pytest.approx(2 / 99, abs=1e-9)


def test_fpr_on_empty_empty_gt_raises():
    """fpr_on_empty raises ValueError on empty GT (no divide-by-zero NaN).

    An all-empty GT means every voxel is "empty" — but the metric is undefined
    when there is no GT structure to define vessel-free regions against.
    Mirrors the compute_average_diameter empty-mask raise pattern.
    """
    from liom_toolkit.segmentation.vseg.eval_metrics import fpr_on_empty

    gt = np.zeros((10, 10), dtype=bool)
    predicted = np.zeros((10, 10), dtype=bool)
    predicted[1, 1] = True

    with pytest.raises(ValueError, match="empty"):
        fpr_on_empty(predicted, gt)


def test_fpr_on_empty_all_zero_raises():
    """fpr_on_empty raises ValueError on all-zero input."""
    from liom_toolkit.segmentation.vseg.eval_metrics import fpr_on_empty

    gt = np.zeros((10, 10), dtype=bool)
    predicted = np.zeros((10, 10), dtype=bool)

    with pytest.raises(ValueError):
        fpr_on_empty(predicted, gt)


# ---------------------------------------------------------------------------
# cl_dice_metric
# ---------------------------------------------------------------------------


def test_cl_dice_metric_known_answer_perfect():
    """cl_dice_metric delegates to cldice.cl_dice; a perfect match scores 1.0."""
    from liom_toolkit.segmentation.vseg.eval_metrics import cl_dice_metric

    gt = np.zeros((10, 20), dtype=bool)
    gt[3:6, :] = True
    predicted = gt.copy()

    assert cl_dice_metric(predicted, gt) == pytest.approx(1.0, abs=1e-6)


def test_cl_dice_metric_both_empty_returns_zero():
    """cl_dice_metric returns 0.0 on both-empty skeletons (documented cl_dice
    behaviour — no topology to preserve, not a raise)."""
    from liom_toolkit.segmentation.vseg.eval_metrics import cl_dice_metric

    gt = np.zeros((10, 20), dtype=bool)
    predicted = np.zeros((10, 20), dtype=bool)

    assert cl_dice_metric(predicted, gt) == 0.0


def test_cl_dice_metric_non_2d_3d_raises():
    """cl_dice_metric raises ValueError on non-2D/3D input (delegated raise)."""
    from liom_toolkit.segmentation.vseg.eval_metrics import cl_dice_metric

    gt = np.zeros((2, 2, 2, 2), dtype=bool)
    predicted = np.zeros((2, 2, 2, 2), dtype=bool)

    with pytest.raises(ValueError, match="2D or 3D"):
        cl_dice_metric(predicted, gt)


# ---------------------------------------------------------------------------
# reported_dice
# ---------------------------------------------------------------------------


def test_reported_dice_known_answer_perfect():
    """reported_dice returns 1.0 on a perfect match via the pure-NumPy fallback.

    This test requires NO torch — the fallback path is exercised when the MONAI
    DiceMetric is unavailable (the [ai] extra / monai not installed).
    """
    from liom_toolkit.segmentation.vseg.eval_metrics import reported_dice

    gt = np.zeros((10, 20), dtype=bool)
    gt[3:6, :] = True
    predicted = gt.copy()

    assert reported_dice(predicted, gt) == pytest.approx(1.0, abs=1e-6)


def test_reported_dice_known_answer_disjoint():
    """reported_dice returns 0.0 on disjoint masks (no overlap)."""
    from liom_toolkit.segmentation.vseg.eval_metrics import reported_dice

    gt = np.zeros((10, 20), dtype=bool)
    gt[0:3, 0:5] = True
    predicted = np.zeros((10, 20), dtype=bool)
    predicted[7:10, 15:20] = True  # disjoint from GT

    assert reported_dice(predicted, gt) == pytest.approx(0.0, abs=1e-6)


def test_reported_dice_both_empty_raises():
    """reported_dice raises ValueError when both masks are empty (Dice
    undefined — no silent NaN)."""
    from liom_toolkit.segmentation.vseg.eval_metrics import reported_dice

    gt = np.zeros((10, 20), dtype=bool)
    predicted = np.zeros((10, 20), dtype=bool)

    with pytest.raises(ValueError, match="empty"):
        reported_dice(predicted, gt)


def test_reported_dice_monai_dice_metric():
    """reported_dice uses the MONAI DiceMetric when torch + monai are available.

    The importorskip sits at the FIRST line of the test body (never at module
    top — pytest #9542 would otherwise skip the whole module). Skips cleanly
    without torch or monai (the [ai] extra / monai land in a later phase).
    """
    pytest.importorskip("torch")
    pytest.importorskip("monai")

    from liom_toolkit.segmentation.vseg.eval_metrics import reported_dice

    gt = np.zeros((10, 20), dtype=bool)
    gt[3:6, :] = True
    predicted = gt.copy()

    assert reported_dice(predicted, gt) == pytest.approx(1.0, abs=1e-6)
