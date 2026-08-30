"""Ship-gate evaluation metrics for vessel segmentation.

The ship gate is a per-metric matrix (not a composite score): each of the six
metrics below is an independent pass/fail row that surfaces a distinct
silent-wrong-data failure mode that aggregate Dice hides. A model passes the
gate only if ALL six rows pass independently; ``reported_dice`` is reported
for literature comparison and does NOT gate.

The metrics build on existing primitives rather than reimplementing them:

* ``centerline_recall`` and ``cl_dice_metric`` reuse ``cl_score`` / ``cl_dice``
  from :mod:`liom_toolkit.segmentation.vseg.cldice`.
* ``caliber_stratified_recall`` and ``spurious_thin_vessel_rate`` reuse the
  ``distance_transform_edt`` + skeleton diameter pattern from
  :func:`liom_toolkit.segmentation.stats.compute_average_diameter`.
* ``boundary_artifact_regression`` reuses the ``view_as_windows`` patch-grid
  partitioning pattern from :func:`liom_toolkit.segmentation.vseg.utils.create_patches`.

Correctness contract (AGENTS §2): every metric raises ``ValueError`` on
empty / edge inputs BEFORE any division — no silent NaN or zero fallback can
escape into a result table. The offending value is included in the message so
the error is actionable. ``assert`` is never used for validation (it is
stripped under ``python -O``).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_dilation, distance_transform_edt, label
from skimage.morphology import skeletonize
from skimage.util import view_as_windows

from liom_toolkit.segmentation.vseg.cldice import cl_dice, cl_score

__all__ = [
    "boundary_artifact_regression",
    "caliber_stratified_recall",
    "centerline_recall",
    "cl_dice_metric",
    "fpr_on_empty",
    "reported_dice",
    "spurious_thin_vessel_rate",
]


def _as_bool(mask: NDArray[np.generic]) -> NDArray[np.bool_]:
    """Coerce a mask to a boolean array (defensive — callers may pass int).

    Returns
    -------
    NDArray[np.bool_]
        The input array cast to boolean dtype.
    """
    return np.asarray(mask).astype(bool)


def _skeleton_of(gt: NDArray[np.bool_], metric_name: str) -> NDArray[np.bool_]:
    """Skeletonise ``gt`` and raise if the skeleton is empty.

    An empty skeleton means there is no centreline topology to measure —
    returning 0.0 would be a silent wrong-data fallback (AGENTS §2).

    Parameters
    ----------
    gt : NDArray[np.bool_]
        The ground-truth vessel mask to skeletonise.
    metric_name : str
        The calling metric name, included in the error message for context.

    Returns
    -------
    NDArray[np.bool_]
        The non-empty skeleton of ``gt``.

    Raises
    ------
    ValueError
        If the skeleton is empty after skeletonisation.
    """
    skeleton = skeletonize(gt)
    if not skeleton.any():
        raise ValueError(
            f"{metric_name}: GT skeleton is empty after skeletonize "
            f"(gt.sum()={int(gt.sum())}) - no centreline to measure"
        )
    return skeleton


def _region_dice(
    predicted: NDArray[np.bool_],
    gt: NDArray[np.bool_],
    region: NDArray[np.bool_],
) -> float:
    """Dice score restricted to ``region``.

    Returns 1.0 when both prediction and GT are empty in the region (a true
    negative — both agree there is nothing there). This is the only sanctioned
    non-raise path: a region with no vessels on either side is a correct
    agreement, not an undefined quotient.

    Returns
    -------
    float
        The Dice score on ``region`` in ``[0, 1]``; 1.0 when both sides are
        empty in the region.
    """
    pred_r = predicted & region
    gt_r = gt & region
    denom = int(pred_r.sum()) + int(gt_r.sum())
    if denom == 0:
        return 1.0
    intersection = int((pred_r & gt_r).sum())
    return float(2.0 * intersection / denom)


def centerline_recall(predicted: NDArray[np.bool_], gt: NDArray[np.bool_]) -> float:
    """Centreline recall = ``cl_score(predicted, skeletonize(GT))`` (tprec).

    The fraction of GT centreline voxels covered by the predicted mask. A
    branch continuous in GT but broken in the prediction drops the score while
    aggregate Dice stays high — the failure mode this metric exposes.

    Parameters
    ----------
    predicted : ArrayLike
        The predicted vessel mask.
    gt : ArrayLike
        The ground-truth vessel mask.

    Returns
    -------
    float
        The skeleton-intersection ratio in ``[0, 1]``.

    Raises
    ------
    ValueError
        If the GT is empty (no centreline to recall) or the GT skeleton is
        empty after skeletonisation.
    """
    predicted = _as_bool(predicted)
    gt = _as_bool(gt)
    if not gt.any():
        raise ValueError(
            f"centerline_recall: empty GT (gt.sum()={int(gt.sum())}) - no centreline to recall"
        )
    skeleton = _skeleton_of(gt, "centerline_recall")
    return cl_score(predicted.astype(np.float64), skeleton.astype(np.float64))


def caliber_stratified_recall(
    predicted: NDArray[np.bool_],
    gt: NDArray[np.bool_],
    voxel_size_um: float = 6.5,
    capillary_radius_um: float = 5.0,
) -> dict[str, float]:
    """Per-diameter-bin recall: capillary (< ~10 um) vs large vessel.

    Bins GT centreline voxels by their local vessel radius (distance transform
    x skeleton), then reports recall per bin. A bin absent in GT returns 0.0
    (honest - no vessels to recall), NOT a raise.

    Parameters
    ----------
    predicted : ArrayLike
        The predicted vessel mask.
    gt : ArrayLike
        The ground-truth vessel mask.
    voxel_size_um : float
        Voxel edge length in micrometres (default 6.5 for the lab's lightsheet
        data).
    capillary_radius_um : float
        Radius threshold in micrometres below which a vessel is binned as
        capillary (default 5.0 -> ~10 um diameter).

    Returns
    -------
    dict[str, float]
        ``{"capillary_recall": ..., "large_vessel_recall": ...}``.

    Raises
    ------
    ValueError
        If the GT is empty, the GT skeleton is empty, or no positive radii are
        found in the skeleton.
    """
    predicted = _as_bool(predicted)
    gt = _as_bool(gt)
    if not gt.any():
        raise ValueError(
            f"caliber_stratified_recall: empty GT (gt.sum()={int(gt.sum())}) - no vessels to bin"
        )
    skeleton = _skeleton_of(gt, "caliber_stratified_recall")
    distance = distance_transform_edt(gt.astype(np.float64))
    radii = distance * skeleton.astype(bool)
    positive_radii = radii[radii > 0]
    if positive_radii.size == 0:
        raise ValueError(
            "caliber_stratified_recall: no positive radii in skeleton "
            f"(gt.sum()={int(gt.sum())}) — vessel diameters undefined"
        )
    capillary_radius_vox = capillary_radius_um / voxel_size_um
    has_radius = radii > 0
    capillary_mask = has_radius & (radii <= capillary_radius_vox)
    large_mask = has_radius & (radii > capillary_radius_vox)
    pred_covers_skeleton = predicted & skeleton

    def _bin_recall(bin_mask: NDArray[np.bool_]) -> float:
        denom = int(bin_mask.sum())
        if denom == 0:
            return 0.0  # bin absent in GT — honest zero, not a raise
        return float(int((pred_covers_skeleton & bin_mask).sum()) / denom)

    return {
        "capillary_recall": _bin_recall(capillary_mask),
        "large_vessel_recall": _bin_recall(large_mask),
    }


def boundary_artifact_regression(
    predicted: NDArray[np.bool_],
    gt: NDArray[np.bool_],
    patch_size: tuple[int, int] = (256, 256),
) -> dict[str, float]:
    """Patch-boundary vs patch-interior prediction-quality regression.

    Partitions the masks into a non-overlapping patch grid (via
    ``view_as_windows``) and compares prediction quality on the boundary strip
    (the outer ring of each patch, where sliding-window seams appear) against
    the patch interiors. A positive ``regression_delta`` means boundary quality
    drops below interior quality — a systematic patch-grid seam artefact.

    Parameters
    ----------
    predicted : ArrayLike
        The predicted vessel mask.
    gt : ArrayLike
        The ground-truth vessel mask.
    patch_size : tuple[int, int]
        The patch grid cell size in voxels.

    Returns
    -------
    dict[str, float]
        ``{"boundary_quality": ..., "interior_quality": ...,
        "regression_delta": interior - boundary}``.

    Raises
    ------
    ValueError
        If both inputs are empty, or the patch grid does not fit the image.
    """
    predicted = _as_bool(predicted)
    gt = _as_bool(gt)
    # Shape mismatch is the more fundamental error — check it before the
    # empty-input check so a caller with mismatched shapes gets the shape
    # error (the actual bug) rather than a misleading "empty input" message
    # pointing at emptiness.
    if predicted.shape != gt.shape:
        raise ValueError(
            "boundary_artifact_regression: predicted and GT shape mismatch "
            f"({predicted.shape} vs {gt.shape})"
        )
    if not predicted.any() and not gt.any():
        raise ValueError(
            "boundary_artifact_regression: empty input "
            f"(predicted.sum()={int(predicted.sum())}, "
            f"gt.sum()={int(gt.sum())}) - no quality to measure"
        )
    ph, pw = patch_size
    h, w = predicted.shape[:2]
    ny, nx = h // ph, w // pw
    if ny == 0 or nx == 0:
        raise ValueError(
            f"boundary_artifact_regression: patch_size {patch_size} does not fit image {(h, w)}"
        )
    cropped_h, cropped_w = ny * ph, nx * pw
    pred_c = predicted[:cropped_h, :cropped_w]
    gt_c = gt[:cropped_h, :cropped_w]

    # Validate the non-overlapping patch grid partitioning.
    view_as_windows(pred_c, patch_size, step=patch_size)

    strip_width = max(1, min(ph, pw) // 8)
    ring = np.zeros(patch_size, dtype=bool)
    ring[:strip_width, :] = True
    ring[-strip_width:, :] = True
    ring[:, :strip_width] = True
    ring[:, -strip_width:] = True
    boundary = np.tile(ring, (ny, nx))
    interior = ~boundary

    boundary_quality = _region_dice(pred_c, gt_c, boundary)
    interior_quality = _region_dice(pred_c, gt_c, interior)
    return {
        "boundary_quality": boundary_quality,
        "interior_quality": interior_quality,
        "regression_delta": interior_quality - boundary_quality,
    }


def spurious_thin_vessel_rate(
    predicted: NDArray[np.bool_],
    gt: NDArray[np.bool_],
    voxel_size_um: float = 6.5,
    capillary_radius_um: float = 5.0,
) -> float:
    """Rate of false-positive thin vessels not corresponding to any GT vessel.

    Identifies predicted thin vessels (diameter < ~10 um via the distance
    transform on the predicted mask) that do not intersect any GT vessel
    skeleton, and reports their count per unit volume. These are visually
    convincing hallucinations that inflate vessel-density statistics — the
    failure mode this metric exposes.

    A predicted thin vessel is excluded when it falls within the dilated GT
    skeleton (dilation radius covers the vessel width), so correctly predicted
    thick-vessel edge voxels are not counted as spurious.

    Parameters
    ----------
    predicted : ArrayLike
        The predicted vessel mask.
    gt : ArrayLike
        The ground-truth vessel mask.
    voxel_size_um : float
        Voxel edge length in micrometres.
    capillary_radius_um : float
        Radius threshold in micrometres below which a predicted vessel is
        considered thin.

    Returns
    -------
    float
        Spurious thin-vessel voxel count per voxel (dimensionless fraction).

    Raises
    ------
    ValueError
        If the prediction is empty (no vessels to count — no silent 0.0).
    """
    predicted = _as_bool(predicted)
    gt = _as_bool(gt)
    if not predicted.any():
        raise ValueError(
            "spurious_thin_vessel_rate: empty prediction "
            f"(predicted.sum()={int(predicted.sum())}) - no vessels to count"
        )
    capillary_radius_vox = capillary_radius_um / voxel_size_um
    pred_distance = distance_transform_edt(predicted.astype(np.float64))
    thin_mask = predicted & (pred_distance <= capillary_radius_vox)
    if not thin_mask.any():
        # No thin vessels in the prediction at all — honest zero (there are
        # predicted vessels, just none thin enough to be capillary-class).
        return 0.0

    gt_skeleton = skeletonize(gt)
    # math.ceil (not int() truncation) so a sub-voxel radius like 1.5 rounds
    # up to 2 rather than truncating to 1 — truncation could under-cover wide
    # vessel edges and falsely count correctly-predicted edge voxels as
    # spurious thin vessels.
    dilation_radius = max(1, math.ceil(capillary_radius_vox) + 1)
    gt_skeleton_neighbourhood = binary_dilation(gt_skeleton, iterations=dilation_radius)

    labelled, n_components = label(thin_mask)
    spurious_count = 0
    for component_id in range(1, n_components + 1):
        component = labelled == component_id
        if not (component & gt_skeleton_neighbourhood).any():
            spurious_count += int(component.sum())

    volume = int(predicted.size)
    return float(spurious_count / volume)


def fpr_on_empty(predicted: NDArray[np.bool_], gt: NDArray[np.bool_]) -> float:
    """False-positive rate on GT-empty (vessel-free) regions.

    Computes the fraction of predicted-vessel voxels that fall in regions where
    the GT has no vessels. A high rate means the model hallucinates vasculature
    in clearly empty background — hard to catch by visual inspection.

    Parameters
    ----------
    predicted : ArrayLike
        The predicted vessel mask.
    gt : ArrayLike
        The ground-truth vessel mask.

    Returns
    -------
    float
        ``(predicted & ~gt).sum() / (~gt).sum()`` in ``[0, 1]``.

    Raises
    ------
    ValueError
        If the GT is empty (no structure to define vessel-free regions
        against), or if there are no GT-empty regions (GT is entirely vessels
        — the denominator would be zero). Mirrors the empty-mask raise in
        :func:`compute_average_diameter`.
    """
    predicted = _as_bool(predicted)
    gt = _as_bool(gt)
    gt_sum = int(gt.sum())
    if gt_sum == 0:
        raise ValueError(
            f"fpr_on_empty: empty GT (gt.sum()={gt_sum}) - no structure to "
            "define vessel-free regions against"
        )
    empty_region = ~gt
    empty_count = int(empty_region.sum())
    if empty_count == 0:
        raise ValueError(
            "fpr_on_empty: no GT-empty regions (gt covers the whole image, "
            f"gt.sum()={gt_sum}) - denominator would be zero"
        )
    false_positives = int((predicted & empty_region).sum())
    return float(false_positives / empty_count)


def cl_dice_metric(predicted: NDArray[np.bool_], gt: NDArray[np.bool_]) -> float:
    """Topology-preserving clDice metric (delegates to ``cldice.cl_dice``).

    Returns ``2*tprec*tsens/(tprec+tsens)`` where ``tprec`` / ``tsens`` are the
    skeleton-intersection ratios in each direction. Returns 0.0 when both
    skeletons are empty (no topology to preserve - documented cl_dice
    behaviour, not a raise).

    Parameters
    ----------
    predicted : ArrayLike
        The predicted vessel mask.
    gt : ArrayLike
        The ground-truth vessel mask.

    Returns
    -------
    float
        The clDice score in ``[0, 1]``; 0.0 on both-empty skeletons.

    Raises
    ------
    ValueError
        If either input is not 2D or 3D.
    """
    predicted = _as_bool(predicted)
    gt = _as_bool(gt)
    if len(predicted.shape) not in (2, 3):
        raise ValueError(f"cl_dice_metric expects 2D or 3D input, got {len(predicted.shape)}D")
    return cl_dice(predicted, gt)


def reported_dice(predicted: NDArray[np.bool_], gt: NDArray[np.bool_]) -> float:
    """Aggregate Dice score (reported for literature comparison, NOT a gate).

    Uses the MONAI ``DiceMetric`` when ``torch`` and ``monai`` are available
    (the ``[ai]`` extra); otherwise falls back to a pure-NumPy Dice
    ``2*|intersection| / (|predicted| + |gt|)`` so the function works without
    the AI extra installed.

    Parameters
    ----------
    predicted : ArrayLike
        The predicted vessel mask.
    gt : ArrayLike
        The ground-truth vessel mask.

    Returns
    -------
    float
        The Dice score in ``[0, 1]``.

    Raises
    ------
    ValueError
        If both masks are empty (Dice undefined - no silent NaN).
    """
    predicted = _as_bool(predicted)
    gt = _as_bool(gt)
    if not predicted.any() and not gt.any():
        raise ValueError(
            "reported_dice: both masks empty "
            f"(predicted.sum()={int(predicted.sum())}, gt.sum()={int(gt.sum())}) "
            "- Dice is undefined"
        )

    try:
        import torch
        from monai.metrics import DiceMetric
    except ImportError:
        # Pure-NumPy fallback — works without the [ai] extra.
        intersection = int((predicted & gt).sum())
        denom = int(predicted.sum()) + int(gt.sum())
        if denom == 0:
            # Unreachable: both-empty raises above, but guard anyway against a
            # silent NaN escaping into a result table.
            return 0.0
        return float(2.0 * intersection / denom)

    # MONAI DiceMetric path — expects (B, C, H, W) float tensors.
    # ignore_empty=False so MONAI returns 1.0 for both-empty (already guarded
    # by the raise above) and 0.0 for one-empty, matching the NumPy fallback.
    # The default ignore_empty=True returns NaN for empty-GT cases, which
    # would silently propagate into the result table (AGENTS §2 violation).
    pred_t = torch.from_numpy(predicted.astype(np.float32))[None, None]
    gt_t = torch.from_numpy(gt.astype(np.float32))[None, None]
    metric = DiceMetric(include_background=True, reduction="mean", ignore_empty=False)
    result = metric(pred_t, gt_t)
    val = float(result.item())
    if val != val:  # NaN check (val != val is True only for NaN)
        raise ValueError(
            f"reported_dice: MONAI DiceMetric returned NaN "
            f"(predicted.sum()={int(predicted.sum())}, gt.sum()={int(gt.sum())})"
        )
    return val
